"""Run video event detection from a terminal, on local files or a video URL."""
import argparse
from pathlib import Path
import subprocess

from ved.cli import add_options, checkpoint_path, detection_options, parsed_thresholds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--video", nargs="+", help="one or more local video files")
    sources.add_argument("--url", help="one video URL supported by yt-dlp")
    parser.add_argument("--out-dir", "--output", dest="output", default="results")
    parser.add_argument("--max-height", type=int, default=720, help="URL download height")
    parser.add_argument("--cookies-file", default="")
    add_options(parser)
    args = parser.parse_args(argv)
    try:
        if args.url:
            from infer_url import run_pipeline

            paths = run_pipeline(
                args.url, output=args.output, start_time=args.start_time, end_time=args.end_time,
                ckpt=args.ckpt, model_repo=args.model_repo, model_file=args.model_file,
                batch_size=args.batch, stride=args.stride, thresholds=parsed_thresholds(args.thr),
                smooth=args.smooth, merge_gap=args.merge_gap, min_duration=args.min_dur,
                low_ratio=args.low_ratio, max_height=args.max_height, fp16=not args.fp32,
                device=args.device, cookies_file=args.cookies_file, profile=args.profile,
                plot=args.plot, best_clip=args.best_clip, prefetch=args.prefetch,
                scan=args.scan, gate_ratio=args.gate_ratio, refine_padding=args.refine_padding,
                class_names=args.class_names, agg=args.agg, verbose=not args.quiet,
            )
            for name, path in paths.items():
                print(f"{name}: {path}")
            return
        from ved.inference import Detector
        from ved.outputs import write_outputs

        videos = [Path(video).expanduser().resolve() for video in args.video]
        if any(not video.is_file() for video in videos):
            raise FileNotFoundError("One or more input videos do not exist.")
        if len({video.stem for video in videos}) != len(videos):
            raise ValueError("Input videos must have distinct filename stems to avoid overwriting outputs.")
        checkpoint = checkpoint_path(args.ckpt, args.model_repo, args.model_file)
        detector = Detector(checkpoint, device=args.device, fp16=not args.fp32, class_names=args.class_names)
        options = detection_options(args, detector)
        print(f"Device: {detector.device}; classes: {', '.join(detector.class_names)}")
        print(f"Thresholds: {detector.resolve_thresholds(options['thresholds'])}")
        for video in videos:
            print(f"\nVideo: {video}")
            result = detector.detect(str(video), **options)
            print(detector.report(result))
            paths = write_outputs(result, detector, args.output, stem=video.stem,
                                  plot=args.plot, best_clip=args.best_clip, checkpoint=checkpoint)
            for name, path in paths.items():
                print(f"{name}: {path}")
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
