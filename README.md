# Inferensi Video di Google Colab

Jalankan sel berikut di Google Colab. Aktifkan GPU lewat **Runtime -> Change runtime type -> GPU**.

```python
# @title Inferensi Video - Satu Sel Colab
# @markdown Pilih GPU: Runtime -> Change runtime type -> GPU.

# @markdown ## Sumber Video
source_mode = "URL" # @param ["URL", "Local"]
video_url = "https://example.com/video.mp4" # @param {type:"string"}
video_paths = "/content/video1.mp4;/content/video2.mp4" # @param {type:"string"}
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

# @markdown ## Unduhan URL
max_height = 720 # @param [360, 480, 720, 1080, 1440]
cookies_file = "" # @param {type:"string"}

# @markdown ## Output
plot_timeline = True # @param {type:"boolean"}
export_best_clip = True # @param {type:"boolean"}
preview_video = True # @param {type:"boolean"}
download_results = False # @param {type:"boolean"}
quiet = False # @param {type:"boolean"}

import os
import shlex
import subprocess
import sys
from pathlib import Path

# Persiapan repository
repo_dir = Path("/content/train-and-infer")
if not repo_dir.exists():
    subprocess.run([
        "git", "clone",
        "https://github.com/azunzios/train-and-infer.git",
        str(repo_dir)
    ], check=True)

os.chdir(repo_dir)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", "requirements-inference.txt"
], check=True)

import torch
if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("Runtime GPU belum aktif. Pilih GPU lalu jalankan ulang sel.")

if not output_dir.strip():
    raise ValueError("output_dir tidak boleh kosong.")

args = [
    sys.executable, "infer.py",
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
    "--max-height", str(max_height),
    "--agg", aggregation,
]

if source_mode == "URL":
    args += ["--url", video_url]
else:
    videos = [p.strip() for p in video_paths.split(";") if p.strip()]
    if not videos:
        raise ValueError("Isi minimal satu path video pada video_paths.")
    args += ["--video", *videos]

if local_checkpoint.strip():
    args += ["--ckpt", local_checkpoint.strip()]
if class_names_override.strip():
    args += ["--class-names", *[name.strip() for name in class_names_override.split(",") if name.strip()]
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
if cookies_file.strip():
    args += ["--cookies-file", cookies_file.strip()]

thresholds = []
for index, value in enumerate((event_1_threshold, event_2_threshold), start=1):
    if value != -1:
        if not 0 <= value <= 1:
            raise ValueError(f"event_{index}_threshold harus -1 atau antara 0 sampai 1.")
        thresholds += [f"event_{index}={value}"]

if thresholds:
    args += ["--thr", *thresholds]

print("Menjalankan:")
print(" ".join(shlex.quote(str(arg)) for arg in args))
subprocess.run(args, check=True)

# Tampilkan timeline dan klip terbaik
from IPython.display import Image, Video, display

output_path = Path(output_dir)
timeline = output_path / "timeline.png"
best_clip = output_path / "best_clip.mp4"

if timeline.is_file():
    display(Image(filename=str(timeline)))
if preview_video and best_clip.is_file():
    display(Video(str(best_clip), embed=True))

# Unduh seluruh hasil sebagai ZIP
if download_results:
    from google.colab import files
    zip_path = Path("/content/inference_results.zip")
    subprocess.run(["zip", "-r", "-q", str(zip_path), str(output_path)], check=True)
    files.download(str(zip_path))
```
