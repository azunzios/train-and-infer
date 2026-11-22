# Inferensi Video di Google Colab

Aktifkan GPU melalui **Runtime -> Change runtime type -> GPU**, lalu jalankan sel berikut.

```python
# @title Inferensi Video
# @markdown Hanya satu file video lokal yang diproses.

# @markdown ## Input dan Output
video_path = "/content/video.mp4" # @param {type:"string"}
output_dir = "/content/results" # @param {type:"string"}
start_time = "0" # @param {type:"string"}
end_time = "" # @param {type:"string"}

# @markdown ## Model
local_checkpoint = "" # @param {type:"string"}
model_repo = "azunzios/learning" # @param {type:"string"}
model_file = "model.pt" # @param {type:"string"}
class_names_override = "" # @param {type:"string"}

# @markdown ## Performa
device = "auto" # @param ["auto", "cuda", "cpu"]
use_fp16 = True # @param {type:"boolean"}
profile = "balanced" # @param ["balanced", "fast", "fastest"]
batch_size = None # @param {type:"raw"}
stride_sec = None # @param {type:"raw"}
prefetch = True # @param {type:"boolean"}

# @markdown ## Pemindaian
scan_mode = "full" # @param ["full", "adaptive"]
gate_ratio = 0.5 # @param {type:"number"}
refine_padding_sec = 2.0 # @param {type:"number"}

# @markdown ## Threshold dan Post-processing
event_1_threshold = -1.0 # @param {type:"number"}
event_2_threshold = -1.0 # @param {type:"number"}
smooth_sec = 1.0 # @param {type:"number"}
merge_gap_sec = 2.0 # @param {type:"number"}
min_duration_sec = 1.0 # @param {type:"number"}
low_ratio = 0.6 # @param {type:"number"}
aggregation = "mean" # @param ["mean", "max"]

# @markdown ## Tampilan
plot_timeline = True # @param {type:"boolean"}
export_best_clip = True # @param {type:"boolean"}
preview_video = True # @param {type:"boolean"}
quiet = False # @param {type:"boolean"}

import shlex
import subprocess
import sys
from pathlib import Path

# Siapkan proyek dan dependensi inferensi.
repo_dir = Path("/content/train-and-infer")
if not repo_dir.exists():
    subprocess.run([
        "git", "clone",
        "https://github.com/azunzios/train-and-infer.git",
        str(repo_dir),
    ], check=True)

import torch
if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("Runtime GPU belum aktif. Pilih GPU lalu jalankan ulang sel.")

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", str(repo_dir / "requirements-inference.txt"),
], check=True)

video = Path(video_path).expanduser().resolve()
if not video.is_file():
    raise FileNotFoundError(f"File video tidak ditemukan: {video}")

args = [
    sys.executable, str(repo_dir / "infer.py"),
    "--video", str(video),
    "--out-dir", output_dir,
    "--model-repo", model_repo,
    "--model-file", model_file,
    "--profile", profile,
    "--scan", scan_mode,
    "--gate-ratio", str(gate_ratio),
    "--refine-padding", str(refine_padding_sec),
    "--smooth", str(smooth_sec),
    "--merge-gap", str(merge_gap_sec),
    "--min-dur", str(min_duration_sec),
    "--low-ratio", str(low_ratio),
    "--start-time", start_time,
    "--end-time", end_time,
    "--agg", aggregation,
]

if local_checkpoint.strip():
    args += ["--ckpt", local_checkpoint.strip()]
if class_names_override.strip():
    names = [name.strip() for name in class_names_override.split(",") if name.strip()]
    if names:
        args += ["--class-names", *names]
if batch_size not in (None, ""):
    args += ["--batch", str(int(batch_size))]
if stride_sec not in (None, ""):
    args += ["--stride", str(float(stride_sec))]
if device in ("cuda", "cpu"):
    args += ["--device", device]
if not use_fp16:
    args += ["--fp32"]
if not prefetch:
    args += ["--no-prefetch"]
if not plot_timeline:
    args += ["--no-plot"]
if not export_best_clip:
    args += ["--no-best-clip"]
if quiet:
    args += ["--quiet"]

thresholds = []
for index, value in enumerate((event_1_threshold, event_2_threshold), start=1):
    if value != -1:
        if not 0 <= value <= 1:
            raise ValueError(f"event_{index}_threshold harus -1 atau antara 0 dan 1.")
        thresholds.append(f"event_{index}={value}")
if thresholds:
    args += ["--thr", *thresholds]

print("Menjalankan:")
print(" ".join(shlex.quote(str(arg)) for arg in args))
subprocess.run(args, cwd=repo_dir, check=True)

if preview_video and export_best_clip:
    from IPython.display import Video, display

    stem = video.stem
    clip = Path(output_dir).expanduser().resolve() / f"{stem}_best_clip.mp4"
    if not clip.is_file():
        clip = Path(output_dir).expanduser().resolve() / "best_clip.mp4"
    if clip.is_file():
        display(Video(str(clip), embed=True))
```
