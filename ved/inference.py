"""Sliding-window inference over long videos (runs fine on an RTX 4070, 12 GB)."""
from __future__ import annotations

import json
import math
import time
from collections import deque
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
    def __init__(self, ckpt_path: str, device: Optional[str] = None, fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.meta = load_export(ckpt_path, self.device)
        self.fp16 = fp16 and self.device.startswith("cuda")
        if self.fp16:
            self.model.half()
        self.class_names = self.meta["class_names"]
        self.display = self.meta.get("display_names", {})
        self.fps = float(self.meta["fps"])
        self.win = int(self.meta["num_frames"])
        self.size = int(self.meta["img_size"])
        self.thresholds = dict(self.meta.get("thresholds", {}))

    @torch.no_grad()
    def _run(self, windows):
        x = torch.from_numpy(np.stack(windows)).to(self.device, non_blocking=True)
        x = to_model_input(x)
        if self.fp16:
            x = x.half()
        return torch.sigmoid(self.model(x).float()).cpu().numpy()

    def predict_windows(self, video_path: str, stride_sec: float = 0.5, batch_size: int = 8,
                        verbose: bool = True):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not math.isfinite(stride_sec) or not 0 < stride_sec <= self.win / self.fps:
            raise ValueError("stride_sec must be positive and no longer than one model window.")
        stride = max(1, int(round(stride_sec * self.fps)))
        buf = deque(maxlen=self.win)
        pending, pending_starts = [], []
        starts, probs = [], []
        n = 0
        _, _, dur = video_info(video_path)
        t0 = time.time()
        last_start = -1
        for frame in iter_frames(video_path, self.fps, self.size):
            buf.append(frame)
            n += 1
            s = n - self.win
            if s >= 0 and s % stride == 0:
                pending.append(np.stack(buf)); pending_starts.append(s); last_start = s
            if len(pending) >= batch_size:
                probs.append(self._run(pending)); starts += pending_starts
                pending, pending_starts = [], []
                if verbose and dur > 0:
                    print(f"\r  processed {n / self.fps:7.1f}s / {dur:.1f}s "
                          f"({(n / self.fps) / max(time.time() - t0, 1e-6):.1f}x realtime)",
                          end="", flush=True)
        if n == 0:
            raise IOError(f"No frames decoded from {video_path}")
        if n < self.win:  # very short video: pad by repeating the last frame
            frames = list(buf) + [buf[-1]] * (self.win - n)
            pending.append(np.stack(frames)); pending_starts.append(0)
        elif last_start != n - self.win:  # make sure the tail is covered
            pending.append(np.stack(buf)); pending_starts.append(n - self.win)
        if pending:
            probs.append(self._run(pending)); starts += pending_starts
        if verbose:
            print(f"\r  done: {n / self.fps:.1f}s of video in {time.time() - t0:.1f}s" + " " * 20)
        return np.array(starts), np.concatenate(probs, 0), n

    def detect(self, video_path: str, stride_sec: float = 0.5, agg: str = "mean",
               thresholds: Optional[Dict[str, float]] = None, low_ratio: float = 0.6,
               smooth_sec: float = 1.0, merge_gap: float = 2.0, min_dur: float = 1.0,
               batch_size: int = 8, verbose: bool = True):
        if thresholds:
            if set(thresholds) - set(self.class_names):
                raise ValueError("Threshold keys must use event_1, event_2, etc.")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values()):
                raise ValueError("Thresholds must be finite numbers between 0 and 1.")
        starts, probs, n = self.predict_windows(video_path, stride_sec, batch_size, verbose)
        frame_scores = windows_to_frame_scores(starts, probs, n, self.win, agg)
        thr = dict(self.thresholds)
        thr.update(thresholds or {})
        segs = frame_scores_to_segments(frame_scores, self.fps, self.class_names, thr,
                                        low_ratio, smooth_sec, merge_gap, min_dur)
        return {"video": video_path, "segments": segs, "frame_scores": frame_scores,
                "fps": self.fps, "thresholds": thr,
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
           "segments": segments}
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
