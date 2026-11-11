"""Build a self-contained, single-cell Colab notebook for local files or wget."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap


def build_notebook():
    root = Path(__file__).resolve().parents[1]
    package_sources = {
        name: (root / name).read_text(encoding="utf-8")
        for name in ("ved/__init__.py", "ved/labels.py", "ved/model.py",
                     "ved/video_io.py", "ved/postprocess.py", "ved/inference.py")
    }
    source = textwrap.dedent('''
        # @title Inferensi video — satu sel
        # @markdown Pilih **Runtime → Change runtime type → GPU**, isi formulir, lalu jalankan sel ini.
        # @markdown Upload video ke panel Files, atau masukkan URL langsung ke berkas video untuk diunduh dengan **wget**.
        video_url = "/content/test.mp4" # @param {type:"string"}
        output_dir = "/content/outputs/check" # @param {type:"string"}
        # @markdown Seluruh video diproses. Folder output yang sama akan mengganti hasil sebelumnya.
        # @markdown **Model** — kosongkan checkpoint lokal untuk unduh otomatis dari Hub.
        model_repo = "azunzios/learning" # @param {type:"string"}
        model_file = "model.pt" # @param {type:"string"}
        local_checkpoint = "" # @param {type:"string"}
        # @markdown **Inferensi** — mulai dengan batch 8; turunkan ke 4/2/1 jika VRAM penuh.
        batch_size = 8 # @param [1, 2, 4, 8, 16, 32] {type:"raw"}
        stride_sec = 0.5 # @param {type:"number"}
        use_fp16 = True # @param {type:"boolean"}
        require_gpu = True # @param {type:"boolean"}
        # @markdown Threshold **-1** memakai kalibrasi bawaan model.
        event_1_threshold = -1.0 # @param {type:"number"}
        event_2_threshold = -1.0 # @param {type:"number"}
        smooth_sec = 1.0 # @param {type:"number"}
        merge_gap_sec = 2.0 # @param {type:"number"}
        min_duration_sec = 1.0 # @param {type:"number"}
        low_ratio = 0.6 # @param {type:"number"}
        # @markdown Preview menyematkan seluruh video; aktifkan hanya untuk video pendek.
        preview_video = False # @param {type:"boolean"}
        download_results = True # @param {type:"boolean"}

        import importlib
        import hashlib
        import json
        import math
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys
        import tempfile
        from urllib.parse import urlsplit
        import zipfile

        if not video_url.strip():
            raise ValueError("Isi path video lokal atau URL langsung ke file video.")
        if batch_size < 1 or batch_size != int(batch_size):
            raise ValueError("Batch harus bilangan bulat minimal 1.")
        if not math.isfinite(stride_sec) or stride_sec <= 0:
            raise ValueError("Stride harus berupa angka positif.")
        if not 0 <= low_ratio <= 1 or any(not math.isfinite(v) or v < 0
                                         for v in (smooth_sec, merge_gap_sec, min_duration_sec)):
            raise ValueError("Pengaturan post-processing tidak valid.")
        thresholds = {}
        for index, value in enumerate((event_1_threshold, event_2_threshold), 1):
            if not math.isfinite(value) or (value != -1 and not 0 <= value <= 1):
                raise ValueError("Threshold harus -1 atau angka antara 0 dan 1.")
            if value != -1:
                thresholds[f"event_{index}"] = value

        # Dependency dipasang otomatis; PyTorch bawaan Colab tetap digunakan jika sudah memenuhi versi.
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "torch>=2.1", "numpy",
            "opencv-python-headless", "matplotlib", "huggingface_hub",
        ], check=True)
        import torch
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("Aktifkan runtime GPU, lalu jalankan ulang sel ini.")

        # Paket inferensi disertakan di notebook sehingga tidak perlu clone repo.
        package_sources = __PACKAGE_SOURCES__
        runtime_dir = (Path("/content") if Path("/content").is_dir() else Path.cwd() / ".cache") / "video_event_runtime"
        for relative, body in package_sources.items():
            target = runtime_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        for module_name in list(sys.modules):
            if module_name == "ved" or module_name.startswith("ved."):
                del sys.modules[module_name]
        if str(runtime_dir) in sys.path:
            sys.path.remove(str(runtime_dir))
        sys.path.insert(0, str(runtime_dir))
        importlib.invalidate_caches()
        from huggingface_hub import hf_hub_download
        from ved.inference import Detector, save_result, plot_timeline

        checkpoint = local_checkpoint.strip()
        if checkpoint:
            if not Path(checkpoint).expanduser().is_file():
                raise FileNotFoundError("Checkpoint lokal tidak ditemukan.")
            checkpoint = str(Path(checkpoint).expanduser().resolve())
        else:
            token = os.environ.get("HF_TOKEN")
            if token is None:
                try:
                    from google.colab import userdata
                    token = userdata.get("HF_TOKEN")
                except Exception:
                    token = None
            checkpoint = hf_hub_download(model_repo, model_file, token=token)
        detector = Detector(checkpoint, device="cuda" if require_gpu else None, fp16=use_fp16)
        if stride_sec > detector.win / detector.fps:
            raise ValueError("Stride tidak boleh lebih panjang dari jendela model.")
        if set(thresholds) - set(detector.class_names):
            raise ValueError("Threshold mengacu pada kelas yang tidak tersedia di model.")
        print("Device:", detector.device, "| Kelas:", ", ".join(detector.class_names))
        effective_thresholds = dict(detector.thresholds)
        effective_thresholds.update(thresholds)
        print("Threshold efektif:", json.dumps(effective_thresholds))
        print("Threshold bawah:", json.dumps({name: value * low_ratio
                                              for name, value in effective_thresholds.items()}))
        print("Stride:", stride_sec, "| Smooth:", smooth_sec,
              "| Merge gap:", merge_gap_sec, "| Durasi minimum:", min_duration_sec)
        if any(value == 0 for value in effective_thresholds.values()):
            print("PERHATIAN: threshold 0 meloloskan semua skor. Gunakan -1 pada formulir untuk kalibrasi model.")
        if detector.device.startswith("cuda"):
            print("GPU:", torch.cuda.get_device_name(0))

        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        source = video_url.strip()
        if urlsplit(source).scheme.lower() in {"http", "https"}:
            if not shutil.which("wget"):
                if not Path("/content").is_dir():
                    raise RuntimeError("Pasang wget dan tambahkan ke PATH terlebih dahulu.")
                subprocess.run(["apt-get", "update", "-qq"], check=True)
                subprocess.run(["apt-get", "install", "-y", "-qq", "wget"], check=True)
            suffix = Path(urlsplit(source).path).suffix.lower()
            if suffix not in {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}:
                suffix = ".mp4"
            video_path = out / ("input_video" + suffix)
            with tempfile.TemporaryDirectory(prefix=".download-", dir=out) as temp:
                temporary = Path(temp) / ("video" + suffix)
                print("Mengunduh video dengan wget...", flush=True)
                subprocess.run(["wget", "-O", str(temporary), "--", source], check=True)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("Unduhan tidak menghasilkan video.")
                os.replace(temporary, video_path)
        else:
            video_path = Path(source).expanduser().resolve()
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError("Video belum ditemukan atau kosong. Upload file ke panel Files.")

        import cv2
        from ved.video_io import video_info
        source_fps, source_frames, source_duration = video_info(str(video_path))
        with open(checkpoint, "rb") as checkpoint_file:
            checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        run_config = {
            "checkpoint_sha256": checkpoint_sha256,
            "model_settings": detector.meta,
            "effective_thresholds": effective_thresholds,
            "threshold_overrides": thresholds,
            "settings": {"batch_size": int(batch_size), "stride_sec": stride_sec,
                         "fp16": detector.fp16, "device": detector.device,
                         "agg": "mean", "smooth_sec": smooth_sec, "merge_gap": merge_gap_sec,
                         "min_duration": min_duration_sec, "low_ratio": low_ratio},
            "video": {"path": str(video_path), "bytes": video_path.stat().st_size,
                      "fps": source_fps, "frames": source_frames, "duration": source_duration},
            "versions": {"torch": torch.__version__, "opencv": cv2.__version__},
        }
        config_path = out / "run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print("Checkpoint SHA256:", checkpoint_sha256)

        result = detector.detect(
            str(video_path), batch_size=int(batch_size), stride_sec=stride_sec,
            thresholds=thresholds, smooth_sec=smooth_sec, merge_gap=merge_gap_sec,
            min_dur=min_duration_sec, low_ratio=low_ratio,
        )
        print(detector.report(result))
        paths = {"video": video_path, "json": out / "result.json",
                 "config": config_path,
                 "timeline": out / "timeline.png", "archive": out / "inference_results.zip"}
        save_result(result, str(paths["json"]), detector.display)
        figure = plot_timeline(result, detector.class_names, str(paths["timeline"]), detector.display)
        import matplotlib.pyplot as plt
        plt.close(figure)
        with zipfile.ZipFile(paths["archive"], "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(video_path, arcname="video" + video_path.suffix)
            archive.write(paths["json"], arcname="result.json")
            archive.write(paths["timeline"], arcname="timeline.png")
            archive.write(paths["config"], arcname="run_config.json")
        print(json.dumps(json.loads(paths["json"].read_text(encoding="utf-8")), indent=2))
        print("Hasil tersimpan:", out)

        from IPython.display import Image, Video, display
        display(Image(filename=str(paths["timeline"])))
        if preview_video:
            display(Video(str(video_path), embed=True))
        if download_results:
            try:
                from google.colab import files
            except ImportError:
                print("ZIP siap:", paths["archive"])
            else:
                files.download(str(paths["archive"]))
    ''').strip().replace("__PACKAGE_SOURCES__", repr(package_sources))
    notebook = {
        "cells": [{"cell_type": "code", "metadata": {"cellView": "form"},
                   "source": source.splitlines(keepends=True),
                   "execution_count": None, "outputs": []}],
        "metadata": {
            "accelerator": "GPU", "colab": {"provenance": [], "name": "infer_colab.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    target = root / "notebooks" / "infer_colab.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {target.name} (1 cell)")


if __name__ == "__main__":
    build_notebook()
