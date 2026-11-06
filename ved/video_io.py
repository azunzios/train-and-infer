"""Video decoding: sample frames at a fixed FPS and resize them to a square."""
from __future__ import annotations

from typing import Iterator, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}


def video_info(path: str) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1e-3 or fps > 240:
        fps = 30.0
    return fps, n, (n / fps if n > 0 else 0.0)


def iter_frames(path: str, target_fps: float, size: int) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames (size x size x 3) sampled at `target_fps`.

    Frame k corresponds to timestamp k / target_fps seconds.
    Frames are squashed (not cropped) to keep the whole scene visible.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3 or src_fps > 240:
        src_fps = 30.0
    step = 1.0 / target_fps
    next_t = 0.0
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / src_fps
        idx += 1
        if t + 1e-6 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # emit the same frame several times if the source FPS < target FPS
        while next_t <= t + 1e-6:
            yield frame
            next_t += step
    cap.release()


def read_frames(path: str, target_fps: float, size: int) -> np.ndarray:
    frames = list(iter_frames(path, target_fps, size))
    if not frames:
        raise IOError(f"No frames decoded from: {path}")
    return np.stack(frames)
