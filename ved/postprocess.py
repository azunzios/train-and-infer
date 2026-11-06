"""Turn sliding-window probabilities into merged time segments.

Pipeline (per class):
  1. Overlap-add: every frame gets the mean (or max) probability of all
     windows that cover it  -> a frame-level score curve.
  2. Temporal smoothing (moving average) to remove single-window spikes.
  3. Hysteresis thresholding: a segment is a connected region where
     score >= low that contains at least one frame with score >= high.
     This keeps boundaries stable without flickering on/off.
  4. Gap merging: segments of the same class separated by <= merge_gap
     seconds are joined (the event was briefly occluded / low score).
  5. Minimum duration filter: drop segments shorter than min_dur seconds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Segment:
    label: str
    start: float
    end: float
    peak: float
    mean: float

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d


def windows_to_frame_scores(starts: Sequence[int], probs: np.ndarray, num_frames_total: int,
                            window_len: int, agg: str = "mean") -> np.ndarray:
    """starts: window start frame indices; probs: (num_windows, C)."""
    C = probs.shape[1]
    T = max(num_frames_total, 1)
    if agg == "max":
        out = np.zeros((T, C), dtype=np.float32)
        for s, p in zip(starts, probs):
            e = min(s + window_len, T)
            out[s:e] = np.maximum(out[s:e], p)
        return out
    acc = np.zeros((T, C), dtype=np.float64)
    cnt = np.zeros((T, 1), dtype=np.float64)
    for s, p in zip(starts, probs):
        e = min(s + window_len, T)
        acc[s:e] += p
        cnt[s:e] += 1
    cnt[cnt == 0] = 1
    return (acc / cnt).astype(np.float32)


def smooth(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return scores
    kernel = np.ones(k) / k
    pad = k // 2
    out = np.empty_like(scores)
    for c in range(scores.shape[1]):
        x = np.pad(scores[:, c], (pad, k - 1 - pad), mode="edge")
        out[:, c] = np.convolve(x, kernel, mode="valid")
    return out


def hysteresis_regions(s: np.ndarray, high: float, low: float):
    above = s >= low
    regions, i, n = [], 0, len(s)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if s[i:j].max() >= high:
                regions.append((i, j))
            i = j
        else:
            i += 1
    return regions


def merge_regions(regions, max_gap_frames: int):
    if not regions:
        return []
    merged = [list(regions[0])]
    for s, e in regions[1:]:
        if s - merged[-1][1] <= max_gap_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


def frame_scores_to_segments(scores: np.ndarray, fps: float, class_names: List[str],
                             thresholds: Dict[str, float], low_ratio: float = 0.6,
                             smooth_sec: float = 1.0, merge_gap: float = 2.0,
                             min_dur: float = 1.0) -> List[Segment]:
    sm = smooth(scores, max(1, int(round(smooth_sec * fps))))
    segs: List[Segment] = []
    for c, name in enumerate(class_names):
        high = float(thresholds.get(name, 0.5))
        low = high * low_ratio
        regions = hysteresis_regions(sm[:, c], high, low)
        regions = merge_regions(regions, int(round(merge_gap * fps)))
        for s, e in regions:
            if (e - s) / fps < min_dur:
                continue
            seg_scores = sm[s:e, c]
            segs.append(Segment(name, round(s / fps, 2), round(e / fps, 2),
                                round(float(seg_scores.max()), 3), round(float(seg_scores.mean()), 3)))
    segs.sort(key=lambda x: (x.start, x.label))
    return segs


def fmt_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def format_segments(segs: List[Segment], display: Dict[str, str] | None = None) -> str:
    if not segs:
        return "No events detected."
    lines = []
    for s in segs:
        name = (display or {}).get(s.label, s.label)
        lines.append(f"[{fmt_time(s.start)} -> {fmt_time(s.end)}] Detected {name} "
                     f"from {s.start:.2f}s until {s.end:.2f}s "
                     f"(duration {s.duration:.2f}s, peak {s.peak:.2f}, mean {s.mean:.2f})")
    return "\n".join(lines)
