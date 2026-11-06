"""Build notebooks/train_colab.ipynb from the source files in this repo.

The notebook is self-contained: it writes the `ved` package and `infer.py`
with %%writefile, so you only need to upload the .ipynb to Colab.

    python tools/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(True)})


def writefile(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {rel.replace(os.sep, '/')}\n{body}")


# --------------------------------------------------------------------------- #
md("""
# Video event detection + temporal localization (VideoMAEv2-S, small data)

**Pipeline**
1. Download the dataset (Google Drive) and the pretrained **VideoMAEv2 ViT-Small** (distilled from ViT-g, Kinetics-710).
2. Decode every 8 s clip once at 8 FPS / 224 px and cache it.
3. Train a 2-output multi-label classifier (`body_shaking`, `squirting`) with **multiple-instance learning**:
   each clip gives K 2-second windows; the clip label supervises a smooth-max (LogSumExp) over the windows,
   so the model learns *which* window contains the event without needing timestamps.
4. 5-fold cross-validation (split by clip / group) -> out-of-fold metrics + calibrated thresholds.
5. Train the final model on all data and export a small fp16 checkpoint (~45 MB).
6. Sliding-window inference + segment merging -> `Detected X from a s until b s`.

Training: RTX 6000 (48 GB) / A100. Inference: any consumer GPU (RTX 4070 is plenty) via `infer.py`.
""")

md("## 0. Environment")
code("""
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install "gdown>=5.1" huggingface_hub opencv-python-headless scikit-learn matplotlib
WORK = "/content/work" if IN_COLAB else os.path.abspath("work")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
os.makedirs("ved", exist_ok=True)
print("working dir:", os.getcwd())
""")

md("## 1. Write the code package (`ved/`) and `infer.py`")
for f in ["ved/__init__.py", "ved/model.py", "ved/video_io.py", "ved/postprocess.py",
          "ved/inference.py", "ved/train.py", "infer.py"]:
    writefile(f)

md("## 2. Configuration")
code("""
# ---- data -----------------------------------------------------------------
DATA_SOURCE = "gdown_folder"   # "gdown_folder" | "drive_mount" | "gdown_zip"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/17uHSFRmn8QXLMbj5Tq8-qf474_DOd95O?usp=drive_link"
DRIVE_MOUNT_PATH = "/content/drive/MyDrive/dataset"   # used if DATA_SOURCE == "drive_mount"
ZIP_URL = ""                                          # used if DATA_SOURCE == "gdown_zip"
DATA_DIR = os.path.join(WORK, "data")
CACHE_DIR = os.path.join(WORK, "cache")
OUT_DIR = os.path.join(WORK, "export")

# If clips were cut from the same source video, give them a shared group so they
# never end up in both train and validation. Regex with ONE capture group applied to
# the file name, e.g. r"^(.*)_clip\\d+$". None = every clip is its own group.
GROUP_REGEX = None

# ---- pretrained model ------------------------------------------------------
HF_REPO = "OpenGVLab/VideoMAE2"
HF_FILE = "distill/vit_s_k710_dl_from_giant.pth"

# ---- training --------------------------------------------------------------
CFG = dict(
    class_names=["body_shaking", "squirting"],
    folder_to_class={"og": "body_shaking", "sq": "squirting"},
    display_names={"body_shaking": "body shaking", "squirting": "squirting"},
    fps=8, num_frames=16, img_size=224,   # 16 frames @ 8 fps = 2 s window (matches K710 sampling)
    k_train=4, k_eval=7,                  # MIL windows per 8 s clip
    batch_size=8,                         # clips per step (x k_train windows)
    epochs=30, warmup_epochs=3,
    backbone_lr=1.5e-4, head_lr=1e-3, layer_decay=0.75, weight_decay=0.05,
    drop_path=0.1, head_dropout=0.3,
    freeze_blocks=4,                      # freeze patch-embed + first N of 12 blocks (0 = full fine-tune)
    lse_r=5.0, grad_clip=1.0, num_workers=4,
)
N_FOLDS = 5
RUN_CV = True          # set False to skip CV and only train the final model
SEED = 42
""")

md("""
## 3. Download the dataset

`gdown` can only list **50 files per Drive folder**. If a subfolder has more than 50 clips
(og / sq have 80), use one of the alternatives:
* **`drive_mount`**: open the shared link -> *Add shortcut to Drive*, then set `DRIVE_MOUNT_PATH`.
* **`gdown_zip`**: zip the `dataset` folder, share the zip, set `ZIP_URL`.

The cell checks the counts and stops if a folder looks truncated.
""")
code("""
import gdown, glob, zipfile
from ved.video_io import VIDEO_EXTS

if DATA_SOURCE == "gdown_folder":
    gdown.download_folder(DRIVE_FOLDER_URL, output=DATA_DIR, quiet=True, remaining_ok=True)
    data_base = DATA_DIR
elif DATA_SOURCE == "drive_mount":
    from google.colab import drive
    drive.mount("/content/drive")
    data_base = DRIVE_MOUNT_PATH
elif DATA_SOURCE == "gdown_zip":
    os.makedirs(DATA_DIR, exist_ok=True)
    zpath = gdown.download(ZIP_URL, os.path.join(WORK, "dataset.zip"), quiet=False, fuzzy=True)
    zipfile.ZipFile(zpath).extractall(DATA_DIR)
    data_base = DATA_DIR
else:
    raise ValueError(DATA_SOURCE)

truncated = []
for root, dirs, files in os.walk(data_base):
    n = sum(os.path.splitext(f)[1].lower() in VIDEO_EXTS for f in files)
    if n:
        print(f"{n:5d}  {root}")
    if n == 50 and DATA_SOURCE == "gdown_folder":
        truncated.append(root)
if truncated:
    raise RuntimeError("These folders have exactly 50 files -> gdown hit its listing limit:\\n"
                       + "\\n".join(truncated) + "\\nUse DATA_SOURCE='drive_mount' or 'gdown_zip'.")
""")

md("""
## 4. Download the pretrained VideoMAEv2 ViT-S

The Hugging Face repo `OpenGVLab/VideoMAE2` may be **gated**: open
https://huggingface.co/OpenGVLab/VideoMAE2, accept the terms, create a read token and
store it as the Colab secret `HF_TOKEN` (key icon in the left sidebar) or env var `HF_TOKEN`.
""")
code("""
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN")
if token is None and IN_COLAB:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
PRETRAINED = hf_hub_download(HF_REPO, HF_FILE, token=token, local_dir=os.path.join(WORK, "pretrained"))
print(PRETRAINED, f"{os.path.getsize(PRETRAINED) / 1e6:.1f} MB")

# sanity check: the checkpoint loads into our backbone
from ved.model import build_from_pretrained
_ = build_from_pretrained(PRETRAINED, num_classes=2); del _
""")

md("## 5. Scan dataset + cache frames")
code("""
import numpy as np, random, torch
from ved.train import find_dataset_root, scan_dataset, summarize, build_cache

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
root = find_dataset_root(data_base)
print("dataset root:", root)
items = scan_dataset(root, CFG["folder_to_class"], CFG["class_names"], GROUP_REGEX)
summarize(items, CFG["class_names"])
items = build_cache(items, CACHE_DIR, CFG["fps"], CFG["img_size"], workers=8)
lens = np.array([it["n_frames"] for it in items]) / CFG["fps"]
print(f"clip length: min {lens.min():.1f}s  median {np.median(lens):.1f}s  max {lens.max():.1f}s")
""")

md("""
## 6. 5-fold cross-validation (for the thesis numbers + threshold calibration)

Each fold trains from the pretrained checkpoint. Out-of-fold (OOF) clip scores are
collected for every clip, then per-class AP / AUC / P / R / F1 are computed and the
F1-optimal threshold per class is stored for inference.

*Note:* the OOF predictions come from the best epoch on each validation fold, which is
slightly optimistic. For a strict number, keep a separate held-out test set.
""")
code("""
import json, pandas as pd
from ved.train import make_folds, train_model, clip_metrics

os.makedirs(OUT_DIR, exist_ok=True)
Y = np.stack([it["y"] for it in items])
thresholds = {n: 0.5 for n in CFG["class_names"]}
best_epochs, all_hist = [], []

if RUN_CV:
    oof = np.zeros_like(Y)
    for k, (tr, va) in enumerate(make_folds(items, N_FOLDS, SEED)):
        print(f"\\n===== fold {k + 1}/{N_FOLDS} =====")
        model, hist, best = train_model(CFG, PRETRAINED, [items[i] for i in tr], [items[i] for i in va],
                                        tag=f"[f{k + 1}]")
        oof[va] = best["probs"]
        best_epochs.append(best["epoch"])
        for h in hist:
            all_hist.append(dict(h, fold=k + 1))
        del model; torch.cuda.empty_cache()

    metrics = clip_metrics(Y, oof, CFG["class_names"])
    thresholds = {n: metrics[n]["threshold"] for n in CFG["class_names"]}
    print("\\n===== OOF clip-level metrics =====")
    print(pd.DataFrame({n: metrics[n] for n in CFG["class_names"]}).round(3))
    print("mAP:", round(metrics["mAP"], 4), " best epochs per fold:", best_epochs)
    neg = Y.sum(1) == 0
    for c, n in enumerate(CFG["class_names"]):
        print(f"false-positive rate on negative clips [{n}]: {(oof[neg, c] >= thresholds[n]).mean():.3f}")
    pd.DataFrame(all_hist).to_csv(os.path.join(OUT_DIR, "cv_history.csv"), index=False)
    pd.DataFrame(oof, columns=CFG["class_names"]).assign(
        path=[it["path"] for it in items], **{f"y_{n}": Y[:, i] for i, n in enumerate(CFG["class_names"])}
    ).to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)
    json.dump(metrics, open(os.path.join(OUT_DIR, "cv_metrics.json"), "w"), indent=2)
""")
code("""
# training curves + PR curves
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
if RUN_CV:
    h = pd.DataFrame(all_hist)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    for f, g in h.groupby("fold"):
        ax[0].plot(g.epoch, g.loss, label=f"fold {f}"); ax[1].plot(g.epoch, g.val_mAP, label=f"fold {f}")
    ax[0].set_title("train loss"); ax[1].set_title("val mAP"); ax[0].legend()
    for c, n in enumerate(CFG["class_names"]):
        p, r, _ = precision_recall_curve(Y[:, c], oof[:, c])
        ax[2].plot(r, p, label=f"{n} (AP {metrics[n]['AP']:.2f})")
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].set_title("OOF PR curve"); ax[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "cv_curves.png"), dpi=120); plt.show()
""")

md("## 7. Train the final model on all clips and export")
code("""
from ved.model import save_export
final_epochs = int(round(np.mean(best_epochs))) if best_epochs else CFG["epochs"]
final_epochs = max(final_epochs, 10)
print("final training epochs:", final_epochs, " thresholds:", thresholds)
model, hist, _ = train_model(CFG, PRETRAINED, items, None, epochs=final_epochs, tag="[final]")
meta = dict(class_names=CFG["class_names"], display_names=CFG["display_names"], fps=CFG["fps"],
            num_frames=CFG["num_frames"], img_size=CFG["img_size"], thresholds=thresholds,
            pretrained=HF_FILE, train_clips=len(items), epochs=final_epochs, cfg=CFG)
ckpt_path = os.path.join(OUT_DIR, "model.pt")
save_export(model.cpu(), ckpt_path, meta)
print("saved", ckpt_path, f"{os.path.getsize(ckpt_path) / 1e6:.1f} MB")
del model; torch.cuda.empty_cache()
""")

md("""
## 8. Inference on a long video

Set `TEST_VIDEO` to any video (upload it to Colab or put it on Drive). By default a
demo video is built by concatenating a few dataset clips so you can see the merging work.

Post-processing knobs: `stride_sec` (window hop), `smooth_sec`, `merge_gap` (join segments
closer than this), `min_dur`, `low_ratio` (hysteresis), per-class `thresholds`.
""")
code("""
from ved.inference import Detector, plot_timeline, save_result
TEST_VIDEO = ""   # e.g. "/content/drive/MyDrive/test_video.mp4"

if not TEST_VIDEO:
    import cv2
    rng = np.random.default_rng(0)
    pick = [items[i] for i in rng.choice(len(items), 6, replace=False)]
    TEST_VIDEO = os.path.join(WORK, "demo_concat.mp4")
    vw = cv2.VideoWriter(TEST_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), CFG["fps"], (CFG["img_size"],) * 2)
    t = 0.0
    for it in pick:
        arr = np.load(it["cache"])
        lab = [n for n, v in zip(CFG["class_names"], it["y"]) if v] or ["negative"]
        print(f"  {t:6.2f}s - {t + len(arr) / CFG['fps']:6.2f}s  ground truth clip label: {lab}")
        for fr in arr:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        t += len(arr) / CFG["fps"]
    vw.release()
    print("(demo clips are from the training set, so this only checks the pipeline, not accuracy)\\n")

det = Detector(ckpt_path)
res = det.detect(TEST_VIDEO, stride_sec=0.5, agg="mean", smooth_sec=1.0,
                 merge_gap=2.0, min_dur=1.0, low_ratio=0.6)
print(det.report(res))
save_result(res, os.path.join(OUT_DIR, "demo_result.json"), det.display)
plot_timeline(res, det.class_names, os.path.join(OUT_DIR, "demo_timeline.png"), det.display); plt.show()
""")

md("## 9. Package the export for the RTX 4070 machine")
code("""
import shutil
pkg = os.path.join(WORK, "inference_package")
shutil.rmtree(pkg, ignore_errors=True)
os.makedirs(pkg)
shutil.copytree("ved", os.path.join(pkg, "ved"))
shutil.copy("infer.py", pkg)
for f in os.listdir(OUT_DIR):
    shutil.copy(os.path.join(OUT_DIR, f), pkg)
zip_path = shutil.make_archive(os.path.join(WORK, "inference_package"), "zip", pkg)
print("created", zip_path)
if IN_COLAB:
    from google.colab import drive, files
    try:
        drive.mount("/content/drive")
        shutil.copy(zip_path, "/content/drive/MyDrive/")
        print("copied to MyDrive/inference_package.zip")
    except Exception as e:
        print("Drive copy failed:", e); files.download(zip_path)
""")

nb = {"cells": cells, "metadata": {
    "accelerator": "GPU", "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
out = os.path.join(ROOT, "notebooks", "train_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, f"({len(cells)} cells)")
