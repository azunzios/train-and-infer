"""Download a video interval and run an exported model without training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

from ved.media import download_video, time_range


def run_pipeline(url: str, output="outputs/check", start_time="00:00:00", end_time="",
                 ckpt="", model_repo="azunzios/learning", model_file="model.pt",
                 batch_size=8, stride=0.5, thresholds=None, smooth=1.0, merge_gap=2.0,
                 min_duration=1.0, low_ratio=0.6, max_height=720, fp16=True,
                 device=None, cookies_file="", token=None):
    """Return paths to an MP4, JSON, timeline, and ZIP of those three files."""
    from ved.inference import Detector, plot_timeline, save_result

    start, end = time_range(start_time, end_time)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    output = Path(output).expanduser().resolve()
    if ckpt:
        checkpoint = Path(ckpt).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("The configured checkpoint file does not exist.")
    else:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(hf_hub_download(model_repo, model_file, token=token))
    detector = Detector(str(checkpoint), device=device, fp16=fp16)
    print(f"Device: {detector.device}; classes: {', '.join(detector.class_names)}")
    # Validate options before spending time downloading a video.
    if not math.isfinite(stride) or not 0 < stride <= detector.win / detector.fps:
        raise ValueError("stride must be positive and no longer than one model window.")
    if not 0 <= low_ratio <= 1 or any(not math.isfinite(v) or v < 0
                                     for v in (smooth, merge_gap, min_duration)):
        raise ValueError("Invalid post-processing settings.")
    if thresholds and (set(thresholds) - set(detector.class_names)
                       or any(not math.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values())):
        raise ValueError("Use known event identifiers and thresholds between 0 and 1.")
    output.mkdir(parents=True, exist_ok=True)
    video = download_video(url, output / "clip.mp4", start_time, end_time, max_height, cookies_file)
    result = detector.detect(
        str(video), batch_size=batch_size, stride_sec=stride, thresholds=thresholds,
        smooth_sec=smooth, merge_gap=merge_gap, min_dur=min_duration, low_ratio=low_ratio,
    )
    print(detector.report(result))
    result_path = output / "result.json"
    timeline_path = output / "timeline.png"
    save_result(result, str(result_path), detector.display)
    # The ordinary start/end fields refer to clip.mp4; preserve source offsets too.
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["source_range"] = {"start": start, "end": end}
    for segment in data["segments"]:
        segment["source_start"] = round(start + segment["start"], 3)
        segment["source_end"] = round(start + segment["end"], 3)
    result_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    figure = plot_timeline(result, detector.class_names, str(timeline_path), detector.display)
    import matplotlib.pyplot as plt

    plt.close(figure)
    archive = output / "inference_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in (video, result_path, timeline_path):
            bundle.write(path, arcname=path.name)
    return {"video": video, "json": result_path, "timeline": timeline_path, "archive": archive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="single video URL supported by yt-dlp")
    parser.add_argument("--start-time", default="00:00:00", help="seconds, MM:SS, or HH:MM:SS")
    parser.add_argument("--end-time", default="", help="absolute source end time; empty = end of video")
    parser.add_argument("--output", default="outputs/check", help="directory for all inference outputs")
    parser.add_argument("--ckpt", default="", help="local export; empty = download from Hugging Face")
    parser.add_argument("--model-repo", default="azunzios/learning")
    parser.add_argument("--model-file", default="model.pt")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--thr", nargs="*", default=[], help="e.g. event_1=0.6 event_2=0.5")
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--merge-gap", type=float, default=2.0)
    parser.add_argument("--min-dur", type=float, default=1.0)
    parser.add_argument("--low-ratio", type=float, default=0.6)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--fp32", action="store_true", help="disable fp16 on CUDA")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--cookies-file", default="", help="optional Netscape cookies file")
    args = parser.parse_args()
    try:
        thresholds = {key: float(value) for key, value in (item.split("=", 1) for item in args.thr)}
        paths = run_pipeline(
            args.url, args.output, args.start_time, args.end_time, args.ckpt,
            args.model_repo, args.model_file, args.batch, args.stride, thresholds,
            args.smooth, args.merge_gap, args.min_dur, args.low_ratio, args.max_height,
            not args.fp32, args.device, args.cookies_file,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
