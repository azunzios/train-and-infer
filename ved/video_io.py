"""Video decoding: sample frames at a fixed FPS and resize them to a square."""
from __future__ import annotations

from typing import Iterator, Tuple
import math
import os
from pathlib import Path
import tempfile

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


def iter_frames(path: str, target_fps: float, size: int,
                start_sec: float = 0, end_sec: float | None = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames (size x size x 3) sampled at `target_fps`.

    Frame k corresponds to timestamp k / target_fps seconds.
    Frames are squashed (not cropped) to keep the whole scene visible.
    """
    if not math.isfinite(start_sec) or start_sec < 0 or (end_sec is not None and
            (not math.isfinite(end_sec) or end_sec <= start_sec)):
        raise ValueError("Invalid decoding interval.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3 or src_fps > 240:
        src_fps = 30.0
    step = 1.0 / target_fps
    next_t = start_sec
    idx = max(0, int(math.ceil(start_sec * src_fps - 1e-6)))
    try:
        if idx and not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
            raise IOError("Video backend cannot seek to the requested interval.")
        while True:
            t = idx / src_fps
            if end_sec is not None and next_t >= end_sec - 1e-6:
                break
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Repeat frames if the source FPS is below the model FPS.
            while next_t <= t + 1e-6 and (end_sec is None or next_t < end_sec - 1e-6):
                yield frame
                next_t += step
    finally:
        cap.release()


def read_frames(path: str, target_fps: float, size: int) -> np.ndarray:
    frames = list(iter_frames(path, target_fps, size))
    if not frames:
        raise IOError(f"No frames decoded from: {path}")
    return np.stack(frames)


def export_segment(video_path: str, output_path: str, start: float, end: float) -> str:
    """Save one detected interval using OpenCV (video only, without audio)."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Invalid clip interval.")
    output = Path(output_path).resolve()
    if output == Path(video_path).resolve():
        raise ValueError("The clip output must differ from the source video.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clip-", dir=output.parent) as temporary:
        destination = Path(temporary) / "clip.mp4"
        cap = cv2.VideoCapture(video_path)
        writer = None
        written = 0
        try:
            if not cap.isOpened():
                raise IOError("Cannot open the source video for clipping.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Cannot determine source video FPS.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first, last = int(math.floor(start * fps)), int(math.ceil(end * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            if not writer.isOpened():
                raise IOError("Cannot create the output MP4.")
            for _ in range(first, last):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if not written or not destination.is_file() or destination.stat().st_size == 0:
            raise IOError("The selected interval produced no video frames.")
        os.replace(destination, output)
    return str(output)
