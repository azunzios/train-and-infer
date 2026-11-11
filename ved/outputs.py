"""Write reviewable inference outputs without creating archives."""
import hashlib
import json
from pathlib import Path


def write_outputs(result, detector, output, stem="result", plot=True, best_clip=True,
                  source_offset=0, source_range=None, checkpoint=""):
    from .inference import plot_timeline, save_result
    from .video_io import export_segment

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = Path(result["video"]).resolve()
    paths = {"video": video, "json": output / f"{stem}.json"}
    save_result(result, str(paths["json"]), detector.display)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    if source_range is not None:
        data["source_range"] = source_range
        for segment in data["segments"]:
            segment["source_start"] = round(source_offset + segment["start"], 3)
            segment["source_end"] = round(source_offset + segment["end"], 3)
    data["best_clip"] = None
    if best_clip and result["segments"]:
        index = max(range(len(result["segments"])),
                    key=lambda i: (result["segments"][i].peak, result["segments"][i].mean))
        best = result["segments"][index]
        name = "best_clip" if stem == "result" else f"{stem}_best_clip"
        clip = output / f"{name}.mp4"
        if clip == video:
            clip = output / f"{name}_detection.mp4"
        export_segment(str(video), str(clip), best.start, best.end)
        paths["clip"] = clip
        data["best_clip"] = dict(data["segments"][index], file=str(clip))
    paths["json"].write_text(json.dumps(data, indent=2), encoding="utf-8")
    if plot:
        paths["timeline"] = output / f"{stem if stem != 'result' else 'timeline'}.png"
        figure = plot_timeline(result, detector.class_names, str(paths["timeline"]), detector.display)
        import matplotlib.pyplot as plt

        plt.close(figure)
    if checkpoint:
        import cv2
        import torch

        with open(checkpoint, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        paths["config"] = output / f"{stem if stem != 'result' else 'run'}_config.json"
        config = {"checkpoint_sha256": digest, "model_settings": detector.meta,
                  "effective_thresholds": result["thresholds"],
                  "performance": result.get("performance", {}),
                  "versions": {"torch": torch.__version__, "opencv": cv2.__version__}}
        paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths
