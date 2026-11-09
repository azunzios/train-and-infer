"""Download a video interval with yt-dlp and produce a browser-friendly MP4."""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit


def parse_time(value: str | float) -> float:
    """Accept nonnegative seconds, MM:SS, or HH:MM:SS (including decimals)."""
    try:
        parts = str(value).strip().split(":")
        if not 1 <= len(parts) <= 3 or any(not p.strip() for p in parts):
            raise ValueError
        numbers = [float(p) for p in parts]
        if any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError
        if len(numbers) > 1 and any(n >= 60 for n in numbers[1:]):
            raise ValueError
        if any(n != int(n) for n in numbers[:-1]):
            raise ValueError
        return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Use nonnegative seconds, MM:SS, or HH:MM:SS.") from None


def time_range(start_time="00:00:00", end_time="") -> tuple[float, float | None]:
    start = parse_time(start_time)
    end = parse_time(end_time) if str(end_time).strip() else None
    if end is not None and end <= start:
        raise ValueError("end_time must be greater than start_time.")
    return start, end


def download_video(url: str, output: str | Path, start_time="00:00:00", end_time="",
                   max_height: int = 720, cookies_file: str = "") -> Path:
    """Fetch one interval, then normalize codecs and discard source metadata.

    yt-dlp delegates interval extraction to FFmpeg. Some sources may still need
    to download more than the requested interval. End time is absolute in the
    original video; an empty end time means the remainder of the video.
    """
    start, end = time_range(start_time, end_time)
    if urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc:
        raise ValueError("Provide an HTTP(S) video URL.")
    if max_height < 1:
        raise ValueError("max_height must be positive.")
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} is missing. Install FFmpeg and add its bin folder to PATH.")
    if cookies_file and not Path(cookies_file).is_file():
        raise FileNotFoundError("The configured cookies file does not exist.")

    from yt_dlp import YoutubeDL
    from yt_dlp.utils import download_range_func

    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("The downloaded video output must end in .mp4.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".video-", dir=output.parent) as temp:
        temp = Path(temp)
        options = {
            "format": (f"bv[height<={max_height}]+ba/"
                       f"b[height<={max_height}]/b"),
            "outtmpl": str(temp / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "overwrites": True,
            "merge_output_format": "mkv",
        }
        if start > 0 or end is not None:
            options["download_ranges"] = download_range_func(None, [(start, end or math.inf)])
            options["force_keyframes_at_cuts"] = True
        if cookies_file:
            options["cookiefile"] = str(Path(cookies_file).resolve())
        # Deno is enabled by default; also support a recent Node installation.
        if shutil.which("node"):
            options["js_runtimes"] = {"deno": {}, "node": {}}
        print("Downloading the selected video interval...", flush=True)
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
            if not info or info.get("_type") in {"playlist", "multi_video"}:
                raise ValueError("Provide a single video URL rather than a playlist.")
            downloader.process_ie_result(info, download=True)
        sources = [p for p in temp.glob("source.*")
                   if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".ts"}]
        if len(sources) != 1:
            raise RuntimeError("Download did not produce one usable video file.")
        converted = temp / "clip.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(sources[0]),
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "fast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
            "-map_metadata", "-1", "-map_chapters", "-1", str(converted),
        ]
        subprocess.run(command, check=True)
        if not converted.is_file() or converted.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce a nonempty video.")
        os.replace(converted, output)
    return output
