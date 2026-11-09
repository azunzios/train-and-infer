"""Command-line inference (designed for a consumer GPU such as an RTX 4070).

Example:
    python infer.py --ckpt export/model.pt --video my_video.mp4 --plot
"""
import argparse
import os

from ved.inference import Detector, plot_timeline, save_result


def main():
    ap = argparse.ArgumentParser(description="Detect and localize events in a video.")
    ap.add_argument("--ckpt", required=True, help="exported model.pt")
    ap.add_argument("--video", required=True, nargs="+", help="one or more video files")
    ap.add_argument("--stride", type=float, default=0.5, help="window stride in seconds")
    ap.add_argument("--agg", choices=["mean", "max"], default="mean",
                    help="how overlapping windows are combined per frame")
    ap.add_argument("--smooth", type=float, default=1.0, help="moving-average length (s)")
    ap.add_argument("--merge-gap", type=float, default=2.0, help="merge segments closer than this (s)")
    ap.add_argument("--min-dur", type=float, default=1.0, help="drop segments shorter than this (s)")
    ap.add_argument("--low-ratio", type=float, default=0.6, help="hysteresis low = high * ratio")
    ap.add_argument("--thr", nargs="*", default=[], help="override thresholds, e.g. event_1=0.6")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                    help="default: CUDA when available, otherwise CPU")
    ap.add_argument("--fp32", action="store_true", help="disable fp16")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--plot", action="store_true", help="save a score timeline PNG")
    args = ap.parse_args()

    try:
        thr = {k: float(v) for k, v in (t.split("=", 1) for t in args.thr)}
    except ValueError:
        ap.error("--thr must use event_1=0.6 event_2=0.5 format")
    det = Detector(args.ckpt, device=args.device, fp16=not args.fp32)
    print(f"device={det.device}  classes={det.class_names}  thresholds={det.thresholds | thr}")
    os.makedirs(args.out_dir, exist_ok=True)
    for video in args.video:
        print(f"\n=== {video}")
        res = det.detect(video, stride_sec=args.stride, agg=args.agg, thresholds=thr,
                         low_ratio=args.low_ratio, smooth_sec=args.smooth,
                         merge_gap=args.merge_gap, min_dur=args.min_dur, batch_size=args.batch)
        print(det.report(res))
        stem = os.path.splitext(os.path.basename(video))[0]
        save_result(res, os.path.join(args.out_dir, stem + ".json"), det.display)
        if args.plot:
            plot_timeline(res, det.class_names, os.path.join(args.out_dir, stem + ".png"), det.display)


if __name__ == "__main__":
    main()
