"""Options shared by local-file and URL command-line entry points."""
from pathlib import Path

from .media import time_range


def checkpoint_path(ckpt="", model_repo="azunzios/learning", model_file="model.pt", token=None):
    if ckpt:
        path = Path(ckpt).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("The configured checkpoint file does not exist.")
        return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_repo, model_file, token=token)


def add_options(parser):
    import argparse

    parser.add_argument("--ckpt", default="", help="local model.pt; otherwise download from the Hub")
    parser.add_argument("--model-repo", default="azunzios/learning")
    parser.add_argument("--model-file", default="model.pt")
    parser.add_argument("--class-names", nargs="+", help="optional labels in checkpoint output order")
    parser.add_argument("--profile", choices=["balanced", "fast", "fastest"], default="balanced",
                        help="balanced: stride 0.5/batch 8; fast: 1/16; fastest: 2/16 (less temporal detail)")
    parser.add_argument("--batch", type=int, default=None, help="override profile batch size")
    parser.add_argument("--stride", type=float, default=None, help="override window stride in seconds")
    parser.add_argument("--scan", choices=["full", "adaptive"], default="full",
                        help="adaptive: coarse scan then dense refinement near candidate events; may miss brief events")
    parser.add_argument("--gate-ratio", type=float, default=0.5, help="adaptive gate = class threshold * ratio")
    parser.add_argument("--refine-padding", type=float, default=2.0, help="extra context seconds for adaptive refinement")
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                        help="decode the next batch while the model runs")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--thr", nargs="*", default=[], help="CLASS=0.6; original labels and event_N aliases accepted")
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--merge-gap", type=float, default=2.0)
    parser.add_argument("--min-dur", type=float, default=1.0)
    parser.add_argument("--low-ratio", type=float, default=0.6)
    parser.add_argument("--start-time", default="0", help="analyze from seconds, MM:SS or HH:MM:SS")
    parser.add_argument("--end-time", default="", help="absolute source end; empty = remainder")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--fp32", action="store_true", help="disable FP16 on CUDA")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--best-clip", action=argparse.BooleanOptionalAction, default=True,
                        help="export the segment with highest peak, then mean; video only")
    parser.add_argument("--quiet", action="store_true", help="hide progress (results and paths still printed)")


def parsed_thresholds(items):
    result = {}
    for item in items:
        key, value = item.split("=", 1)
        if key in result:
            raise ValueError(f"Duplicate threshold: {key}")
        result[key] = float(value)
    return result


def profile_options(profile="balanced", batch=None, stride=None, window_sec=2.0):
    defaults = {"balanced": (8, 0.5), "fast": (16, 1.0), "fastest": (16, 2.0)}
    if profile not in defaults:
        raise ValueError("Unknown inference profile.")
    default_batch, default_stride = defaults[profile]
    return (default_batch if batch is None else batch,
            min(default_stride, window_sec) if stride is None else stride)


def detection_options(args, detector):
    batch, stride = profile_options(args.profile, args.batch, args.stride, detector.win / detector.fps)
    start, end = time_range(args.start_time, args.end_time)
    return dict(batch_size=batch, stride_sec=stride, thresholds=parsed_thresholds(args.thr),
                agg=args.agg, smooth_sec=args.smooth, merge_gap=args.merge_gap, min_dur=args.min_dur,
                low_ratio=args.low_ratio, prefetch=args.prefetch, verbose=not args.quiet,
                start_sec=start, end_sec=end, scan=args.scan, gate_ratio=args.gate_ratio,
                refine_padding=args.refine_padding)
