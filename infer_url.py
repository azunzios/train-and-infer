"""Download a video interval and run inference, without creating an archive."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from ved.cli import add_options, checkpoint_path, parsed_thresholds, profile_options
from ved.media import download_video, time_range


def run_pipeline(url: str, output="outputs/check", start_time="00:00:00", end_time="",
                 ckpt="", model_repo="azunzios/learning", model_file="model.pt",
                 batch_size=None, stride=None, thresholds=None, smooth=1.0, merge_gap=2.0,
                 min_duration=1.0, low_ratio=0.6, max_height=720, fp16=True,
                 device=None, cookies_file="", token=None, profile="balanced",
                 plot=True, best_clip=True, prefetch=True, scan="full", gate_ratio=0.5,
                 refine_padding=2.0, class_names=None, agg="mean", verbose=True):
    """Return video/JSON/config paths, plus optional timeline and best segment."""
    import math

    from ved.inference import Detector
    from ved.outputs import write_outputs

    start, end = time_range(start_time, end_time)
    checkpoint = checkpoint_path(ckpt, model_repo, model_file, token)
    detector = Detector(checkpoint, device=device, fp16=fp16, class_names=class_names)
    batch_size, stride = profile_options(profile, batch_size, stride, detector.win / detector.fps)
    effective_thresholds = detector.resolve_thresholds(thresholds)
    # Validate options before downloading a potentially large video.
    if not math.isfinite(batch_size) or batch_size < 1 or int(batch_size) != batch_size:
        raise ValueError("batch_size must be a positive integer.")
    if not math.isfinite(stride) or not 0 < stride <= detector.win / detector.fps:
        raise ValueError("stride must be positive and no longer than one model window.")
    if (not 0 <= low_ratio <= 1 or not 0 < gate_ratio <= 1
            or any(not math.isfinite(v) or v < 0 for v in (smooth, merge_gap, min_duration, refine_padding))
            or scan not in {"full", "adaptive"} or agg not in {"mean", "max"}):
        raise ValueError("Invalid inference settings.")
    print(f"Device: {detector.device}; classes: {', '.join(detector.class_names)}")
    print(f"Thresholds: {effective_thresholds}")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = download_video(url, output / "clip.mp4", start_time, end_time, max_height, cookies_file)
    result = detector.detect(
        str(video), batch_size=batch_size, stride_sec=stride, thresholds=thresholds,
        smooth_sec=smooth, merge_gap=merge_gap, min_dur=min_duration, low_ratio=low_ratio,
        prefetch=prefetch, scan=scan, gate_ratio=gate_ratio, refine_padding=refine_padding,
        agg=agg, verbose=verbose,
    )
    print(detector.report(result))
    return write_outputs(result, detector, output, plot=plot, best_clip=best_clip,
                         source_offset=start, source_range={"start": start, "end": end},
                         checkpoint=checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="single video URL supported by yt-dlp")
    parser.add_argument("--output", default="outputs/check")
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--cookies-file", default="", help="optional Netscape cookies file")
    add_options(parser)
    args = parser.parse_args(argv)
    try:
        paths = run_pipeline(
            args.url, args.output, args.start_time, args.end_time, args.ckpt,
            args.model_repo, args.model_file, args.batch, args.stride, parsed_thresholds(args.thr),
            args.smooth, args.merge_gap, args.min_dur, args.low_ratio, args.max_height,
            not args.fp32, args.device, args.cookies_file, profile=args.profile,
            plot=args.plot, best_clip=args.best_clip, prefetch=args.prefetch,
            scan=args.scan, gate_ratio=args.gate_ratio, refine_padding=args.refine_padding,
            class_names=args.class_names, agg=args.agg, verbose=not args.quiet,
        )
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
