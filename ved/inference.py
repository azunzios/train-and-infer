"""Sliding-window inference over long videos (runs fine on an RTX 4070, 12 GB)."""
from __future__ import annotations

import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
import torch

from .model import load_export
from .postprocess import format_segments, frame_scores_to_segments, windows_to_frame_scores
from .video_io import iter_frames, video_info

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def to_model_input(batch_uint8: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 -> (B, 3, T, H, W) normalized float."""
    x = batch_uint8.permute(0, 4, 1, 2, 3).float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class Detector:
    def __init__(self, ckpt_path: str, device: Optional[str] = None, fp16: bool = True,
                 class_names=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment. Install a CUDA PyTorch build or use --device cpu.")
        self.model, self.meta = load_export(ckpt_path, self.device)
        if class_names is not None:
            previous = self.meta["class_names"]
            names = list(class_names)
            if len(names) != len(previous) or len(set(names)) != len(names) or any(not n.strip() for n in names):
                raise ValueError("class_names must contain one unique name per model output, in checkpoint order.")
            self.meta["thresholds"] = dict(zip(names, (self.meta["thresholds"][n] for n in previous)))
            self.meta["class_names"] = names
            self.meta["display_names"] = dict(zip(names, names))
            self.model.output_labels = dict(zip(names, names))
            self.model.output_display = dict(zip(names, names))
        self.fp16 = fp16 and self.device.startswith("cuda")
        if self.fp16:
            self.model.half()
        self.class_names = self.meta["class_names"]
        self.display = self.meta.get("display_names", {})
        self.fps = float(self.meta["fps"])
        self.win = int(self.meta["num_frames"])
        self.size = int(self.meta["img_size"])
        self.thresholds = dict(self.meta.get("thresholds", {}))
        self._mean = MEAN.to(self.device)
        self._std = STD.to(self.device)
        self.last_stats = {}

    def resolve_thresholds(self, thresholds=None):
        resolved = {}
        aliases = {f"event_{i + 1}": name for i, name in enumerate(self.class_names)}
        for key, value in (thresholds or {}).items():
            name = key if key in self.class_names else aliases.get(key)
            if name is None:
                raise ValueError(f"Unknown class {key!r}; use {', '.join(self.class_names)} or event_N aliases.")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Thresholds must be finite numbers between 0 and 1.")
            if name in resolved and resolved[name] != value:
                raise ValueError(f"Conflicting thresholds for {name}.")
            resolved[name] = value
        return self.thresholds | resolved

    @torch.inference_mode()
    def _run(self, windows):
        x = torch.from_numpy(np.stack(windows)).to(self.device, non_blocking=True)
        x = x.permute(0, 4, 1, 2, 3).float()
        x.div_(255.0).sub_(self._mean).div_(self._std)
        if self.fp16:
            x = x.half()
        return torch.sigmoid(self.model(x).float()).cpu().numpy()

    def _batches(self, video_path, stride, batch_size, first, end_sec):
        """Decode on one producer; retain at most one batch ahead of inference."""
        buf = deque(maxlen=self.win)
        pending, starts = [], []
        n, last_start = 0, -1
        frames = iter_frames(video_path, self.fps, self.size, first / self.fps, end_sec)
        try:
            for frame in frames:
                buf.append(frame)
                n += 1
                s = n - self.win
                if s >= 0 and s % stride == 0:
                    pending.append(np.stack(buf))
                    starts.append(first + s)
                    last_start = s
                if len(pending) >= batch_size:
                    yield pending, starts, first + n
                    pending, starts = [], []
            if n == 0:
                raise IOError(f"No frames decoded from {video_path} in the selected interval")
            if n < self.win:
                pending.append(np.stack(list(buf) + [buf[-1]] * (self.win - n)))
                starts.append(first)
            elif last_start != n - self.win:
                pending.append(np.stack(buf))
                starts.append(first + n - self.win)
            if pending:
                yield pending, starts, first + n
        finally:
            frames.close()

    def predict_windows(self, video_path: str, stride_sec: float = 0.5, batch_size: int = 8,
                        verbose: bool = True, prefetch: bool = True,
                        start_sec: float = 0, end_sec: float | None = None):
        if not math.isfinite(batch_size) or batch_size < 1 or int(batch_size) != batch_size:
            raise ValueError("batch_size must be a positive integer.")
        if not math.isfinite(stride_sec) or not 0 < stride_sec <= self.win / self.fps:
            raise ValueError("stride_sec must be positive and no longer than one model window.")
        stride = max(1, int(round(stride_sec * self.fps)))
        if not math.isfinite(start_sec) or start_sec < 0 or (end_sec is not None and
                (not math.isfinite(end_sec) or end_sec <= start_sec)):
            raise ValueError("Invalid inference interval.")
        first = int(math.ceil(start_sec * self.fps - 1e-6))
        starts, probs = [], []
        n = first
        _, _, dur = video_info(video_path)
        t0 = time.perf_counter()
        batches = self._batches(video_path, stride, batch_size, first, end_sec)
        executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        future = executor.submit(next, batches, None) if executor else None
        try:
            while True:
                item = future.result() if executor else next(batches, None)
                if item is None:
                    break
                if executor:
                    future = executor.submit(next, batches, None)
                pending, pending_starts, n = item
                probs.append(self._run(pending))
                starts.extend(pending_starts)
                if verbose and dur > 0:
                    print(f"\r  processed {(n - first) / self.fps:7.1f}s "
                          f"({((n - first) / self.fps) / max(time.perf_counter() - t0, 1e-6):.1f}x realtime)",
                          end="", flush=True)
        finally:
            if executor:
                future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
            batches.close()
        elapsed = time.perf_counter() - t0
        self.last_stats = {"seconds": round(elapsed, 4), "windows": len(starts),
                           "video_seconds": (n - first) / self.fps,
                           "realtime_factor": round((n - first) / self.fps / max(elapsed, 1e-6), 3)}
        if verbose:
            print(f"\r  done: {(n - first) / self.fps:.1f}s in {elapsed:.2f}s "
                  f"({self.last_stats['realtime_factor']:.1f}x realtime, {len(starts)} windows)" + " " * 10)
        return np.array(starts), np.concatenate(probs, 0), n

    def detect(self, video_path: str, stride_sec: float = 0.5, agg: str = "mean",
               thresholds: Optional[Dict[str, float]] = None, low_ratio: float = 0.6,
               smooth_sec: float = 1.0, merge_gap: float = 2.0, min_dur: float = 1.0,
               batch_size: int = 8, verbose: bool = True, prefetch: bool = True,
               start_sec: float = 0, end_sec: float | None = None,
               scan: str = "full", gate_ratio: float = 0.5, refine_padding: float = 2.0):
        thr = self.resolve_thresholds(thresholds)
        if agg not in {"mean", "max"} or scan not in {"full", "adaptive"}:
            raise ValueError("Use agg=mean/max and scan=full/adaptive.")
        if not 0 <= low_ratio <= 1 or any(not math.isfinite(v) or v < 0
                                         for v in (smooth_sec, merge_gap, min_dur, refine_padding)):
            raise ValueError("Invalid post-processing or padding settings.")
        if not math.isfinite(gate_ratio) or not 0 < gate_ratio <= 1:
            raise ValueError("gate_ratio must be greater than 0 and at most 1.")
        if not math.isfinite(stride_sec) or not 0 < stride_sec <= self.win / self.fps:
            raise ValueError("stride_sec must be positive and no longer than one model window.")
        t0 = time.perf_counter()
        first = int(math.ceil(start_sec * self.fps - 1e-6)) if math.isfinite(start_sec) else 0
        coarse_stride = self.win / self.fps if scan == "adaptive" else stride_sec
        starts, probs, n = self.predict_windows(video_path, coarse_stride, batch_size, verbose,
                                                prefetch, start_sec, end_sec)
        predicted_windows = len(starts)
        refined_ranges = []
        if scan == "adaptive" and max(1, int(round(stride_sec * self.fps))) < self.win:
            # This gates on existing event scores; it is not a separate scene classifier.
            gate = np.array([thr[name] * gate_ratio for name in self.class_names])
            pad = self.win + int(math.ceil(refine_padding * self.fps))
            for s in starts[np.any(probs >= gate, axis=1)]:
                lo, hi = max(first, int(s) - pad), min(n, int(s) + self.win + pad)
                if refined_ranges and lo <= refined_ranges[-1][1]:
                    refined_ranges[-1][1] = max(refined_ranges[-1][1], hi)
                else:
                    refined_ranges.append([lo, hi])
            parts_s, parts_p = [starts], [probs]
            for lo, hi in refined_ranges:
                refined_s, refined_p, _ = self.predict_windows(
                    video_path, stride_sec, batch_size, verbose, prefetch, lo / self.fps, hi / self.fps)
                predicted_windows += len(refined_s)
                parts_s.append(refined_s)
                parts_p.append(refined_p)
            all_s, all_p = np.concatenate(parts_s), np.concatenate(parts_p)
            # Prefer refined predictions for duplicate windows; never weight duplicates twice.
            _, indices = np.unique(all_s[::-1], return_index=True)
            indices = len(all_s) - 1 - indices
            order = indices[np.argsort(all_s[indices])]
            starts, probs = all_s[order], all_p[order]
        frame_scores = windows_to_frame_scores(starts, probs, n, self.win, agg)
        # Restrict post-processing to the requested interval, including threshold=0.
        segs = frame_scores_to_segments(frame_scores[first:n], self.fps, self.class_names, thr,
                                        low_ratio, smooth_sec, merge_gap, min_dur)
        for segment in segs:
            segment.start = round(segment.start + first / self.fps, 3)
            segment.end = round(segment.end + first / self.fps, 3)
        elapsed = time.perf_counter() - t0
        stats = {"scan": scan, "seconds": round(elapsed, 4), "windows": predicted_windows,
                 "unique_windows": len(starts), "video_seconds": (n - first) / self.fps,
                 "realtime_factor": round((n - first) / self.fps / max(elapsed, 1e-6), 3),
                 "analyzed_range": [first / self.fps, n / self.fps],
                 "refined_ranges": [[lo / self.fps, hi / self.fps] for lo, hi in refined_ranges],
                 "device": self.device, "fp16": self.fp16, "batch_size": batch_size,
                 "stride_sec": stride_sec, "prefetch": prefetch,
                 "gate_ratio": gate_ratio if scan == "adaptive" else None,
                 "refine_padding": refine_padding if scan == "adaptive" else None,
                 "agg": agg, "smooth_sec": smooth_sec, "merge_gap": merge_gap,
                 "min_duration": min_dur, "low_ratio": low_ratio}
        if verbose:
            print(f"Inference total: {elapsed:.2f}s, {stats['realtime_factor']:.1f}x realtime, "
                  f"{predicted_windows} windows ({scan})")
        return {"video": video_path, "segments": segs, "frame_scores": frame_scores,
                "fps": self.fps, "thresholds": thr, "performance": stats,
                "output_labels": getattr(self.model, "output_labels", {}),
                "output_display": getattr(self.model, "output_display", {})}

    def report(self, result) -> str:
        return format_segments(result["segments"], self.display)


def save_result(result, json_path: str, display: Optional[Dict[str, str]] = None):
    labels = result.get("output_labels", {})
    original_display = result.get("output_display", {})
    segments = []
    for segment in result["segments"]:
        name = labels.get(segment.label, segment.label)
        segments.append(dict(segment.to_dict(), label=name,
                             display_name=original_display.get(name, (display or {}).get(name, name))))
    out = {"video": result["video"],
           "thresholds": {labels.get(name, name): value for name, value in result["thresholds"].items()},
           "segments": segments, "performance": result.get("performance", {})}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def plot_timeline(result, class_names, path: Optional[str] = None, display=None):
    import matplotlib.pyplot as plt

    sc = result["frame_scores"]
    t = np.arange(len(sc)) / result["fps"]
    fig, axes = plt.subplots(len(class_names), 1, figsize=(14, 2.4 * len(class_names)), sharex=True)
    axes = np.atleast_1d(axes)
    for c, (ax, name) in enumerate(zip(axes, class_names)):
        ax.plot(t, sc[:, c], lw=0.8)
        thr = result["thresholds"].get(name, 0.5)
        ax.axhline(thr, color="r", ls="--", lw=0.8)
        for s in result["segments"]:
            if s.label == name:
                ax.axvspan(s.start, s.end, color="orange", alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_ylabel((display or {}).get(name, name))
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120)
    return fig
