"""Checkpoint labels and backwards-compatible event aliases."""
from __future__ import annotations


def inference_metadata(meta: dict, num_classes: int) -> dict:
    """Retain the checkpoint's output order, labels and calibrated thresholds."""
    result = neutral_metadata(meta, num_classes)
    names = list(meta["class_names"])
    result.update(class_names=names,
                  display_names={name: meta.get("display_names", {}).get(name, name)
                                 for name in names},
                  thresholds={name: float(meta.get("thresholds", {}).get(name, 0.5))
                              for name in names})
    return result


def neutral_metadata(meta: dict, num_classes: int) -> dict:
    """Keep output order and calibrated thresholds from an existing export.

    Only inference settings are retained. Historical training configuration and
    display text are intentionally omitted from the returned metadata.
    """
    previous = list(meta["class_names"])
    if num_classes < 1 or len(previous) != num_classes or len(set(previous)) != num_classes:
        raise ValueError("Checkpoint class metadata does not match the output head.")
    names = [f"event_{i + 1}" for i in range(num_classes)]
    thresholds = meta.get("thresholds", {})
    return {
        "class_names": names,
        "display_names": {name: f"Event {i + 1}" for i, name in enumerate(names)},
        "thresholds": {name: float(thresholds.get(old, 0.5))
                       for old, name in zip(previous, names)},
        "fps": float(meta["fps"]),
        "num_frames": int(meta["num_frames"]),
        "img_size": int(meta["img_size"]),
    }
