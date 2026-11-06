# Video Event Detector (VideoMAEv2-S + MIL)

This project detects **where** two events happen in long videos (`body_shaking` and `squirting`). It trains on short labelled clips, and you don't need timestamps.

```
[00:12.50 -> 00:18.00] Detected body shaking from 12.50s until 18.00s (duration 5.50s, peak 0.93, mean 0.81)
[01:02.00 -> 01:06.50] Detected squirting from 62.00s until 66.50s (duration 4.50s, peak 0.88, mean 0.74)
```

## Repo layout

| Path | Purpose |
|---|---|
| `notebooks/train_colab.ipynb` | **Self-contained** notebook: download → cache → CV → final training → export → inference demo |
| `ved/model.py` | VideoMAEv2 ViT backbone (loads the official checkpoint) + multi-label head |
| `ved/video_io.py` | Decode videos at a fixed FPS (8 fps, 224×224) |
| `ved/train.py` | Dataset scan, frame cache, MIL training, layer-wise LR decay, CV, metrics |
| `ved/inference.py` | Streaming sliding-window inference (low RAM, fp16) |
| `ved/postprocess.py` | Merges windows into segments: overlap-add → smoothing → hysteresis → gap merge → min-duration filter |
| `infer.py` | Command-line inference for the RTX 4070 |
| `tools/build_notebook.py` | Rebuilds the notebook from the source files (`python tools/build_notebook.py`) |

## Dataset layout expected

```
dataset/
  positive/
    og/   -> body_shaking clips (~8 s)
    sq/   -> squirting clips   (~8 s)
  negative/
    fp/ ic/ ns/   -> all merged as negative
```

> **gdown limit:** `gdown.download_folder` can only list **50 files per folder**. `og` and `sq` have 80 clips each, so they will be truncated. The notebook detects this and stops. You can fix it in one of two ways:
> * Set `DATA_SOURCE="drive_mount"`. Open the shared link, click *Add shortcut to Drive*, and point `DRIVE_MOUNT_PATH` at it.
> * Set `DATA_SOURCE="gdown_zip"`. Zip the `dataset` folder, share the zip, and set `ZIP_URL`.

## Method

1. **Backbone:** VideoMAEv2 **ViT-S**, distilled from ViT-g on Kinetics-710 (`OpenGVLab/VideoMAE2`, file `distill/vit_s_k710_dl_from_giant.pth`). It has 22M parameters and takes 16 frames at 224 px. The HF repo may be gated: accept the terms and set the `HF_TOKEN` secret.
2. **Input:** 16 frames sampled at 8 fps, so each window covers 2 s. This matches the K710 pre-training sampling (about stride 4 at 30 fps).
3. **Multiple-instance learning:** each 8 s clip gives K=4 random 2 s windows. The clip-level label supervises a LogSumExp (smooth-max) pooling of the window logits. A positive clip only needs *some* window to fire. A negative clip pushes *every* window down. This gives a window-level detector without timestamp labels.
4. **Small-data regime:**
   * Patch-embed and the first 4 of 12 blocks are frozen.
   * Layer-wise LR decay is 0.75, drop-path 0.1, head dropout 0.3.
   * Augmentation: random resized crops, flips and brightness/contrast jitter. Temporal jitter comes from the random window positions.
   * Classes are re-weighted with `pos_weight = sqrt(neg/pos)`.
5. **Evaluation:** 5-fold `StratifiedGroupKFold` gives out-of-fold AP, AUC, P, R and F1 per class, plus the false-positive rate on negative clips. The F1-optimal thresholds are saved into the exported model.
6. **Final model:** trained on all clips for the mean best epoch from CV, then exported as fp16 (~45 MB).

### Merging algorithm (`ved/postprocess.py`)
1. A 2 s window slides with a 0.5 s stride and produces per-window probabilities.
2. **Overlap-add:** each 1/8 s frame gets the mean probability of all windows covering it, which makes a smooth curve. Use `--agg max` for higher recall but wider boundaries.
3. A **moving average** over 1 s removes single-window spikes.
4. **Hysteresis:** a segment is a region where the score stays at or above `low = 0.6 × threshold` and peaks at or above `threshold`. Boundaries stay stable and don't flicker.
5. **Gap merge:** same-class segments less than or equal to `merge_gap` (2 s) apart are joined.
6. **Minimum duration:** segments shorter than `min_dur` (1 s) are dropped.
7. Each segment is reported with its start, end, duration, peak and mean score.

## Training (RTX 6000 48 GB / A100)
Open `notebooks/train_colab.ipynb` in Colab (or Jupyter on the school machine) and run it top to bottom. The last cell builds `inference_package.zip`, which contains `ved/`, `infer.py`, `model.pt` and the CV reports.

## Inference (RTX 4070)
```bash
pip install torch opencv-python numpy matplotlib   # install the CUDA build of torch
unzip inference_package.zip -d pkg && cd pkg
python infer.py --ckpt model.pt --video long_video.mp4 --plot
# tune: --stride 0.5 --merge-gap 2 --min-dur 1 --smooth 1 --thr squirting=0.6 body_shaking=0.5
```
Results are written to `results/<video>.json` and `results/<video>.png`. ViT-S in fp16 uses less than 2 GB of VRAM. Video decoding (OpenCV, CPU) is usually the bottleneck.

## Notes for the thesis
* **Leakage:** if several clips come from the same source video, set `GROUP_REGEX` so they share a fold. Otherwise CV scores will be inflated.
* **Threshold mismatch:** thresholds are calibrated on clip scores (max over windows). The timeline uses smoothed scores, so you may need to tune `--thr` and `--low-ratio` on a few full-length validation videos.
* **Long-video evaluation:** for the strongest results chapter, annotate start and end times on 5–10 held-out full videos. Then report segment-level precision and recall (or mAP at tIoU 0.3/0.5) and false alarms per hour.
* **Ablations to report:** frozen vs. partial vs. full fine-tune (`freeze_blocks` = 12 / 4 / 0), window length, `k_train`, and `agg` mean vs. max.
* **Ethics:** get approval from your ethics committee. Use only legally sourced content with verified adult performers, and don't redistribute the dataset.
