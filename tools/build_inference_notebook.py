"""Generate the inference-only Colab notebook with editable form fields."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap


def build_notebook():
    cells = []

    def cell(kind, source, form=False):
        item = {
            "cell_type": kind,
            "metadata": {"cellView": "form"} if form else {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """
        # Inferensi video — tanpa training

        Model: [`azunzios/learning`](https://huggingface.co/azunzios/learning).
        Label hasil menggunakan `event_1`, `event_2`, dan seterusnya sesuai urutan output model.

        1. Pilih **Runtime → Change runtime type → GPU** (misalnya T4).
        2. Jalankan setup, isi formulir, lalu jalankan inferensi dan sel hasil.
        3. Video diunduh dengan **yt-dlp**, dipotong sesuai waktu sumber dengan **FFmpeg**,
           lalu dianalisis memakai checkpoint yang sudah dilatih.

        `start_time` dan `end_time` adalah waktu absolut pada video sumber, bukan durasi.
        Contoh: `00:01:00` sampai `00:01:30` menghasilkan klip 30 detik.
        Kosongkan `end_time` untuk memproses sampai akhir. Untuk pemeriksaan pertama,
        gunakan klip pendek. Sumber tertentu mungkin tetap memerlukan unduhan lebih panjang.
    """)
    cell("code", '''
        # @title 1. Setup lingkungan (jalankan sekali)
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys

        REPO = Path("/content/train-and-infer")
        if not (REPO / ".git").is_dir():
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/azunzios/train-and-infer.git", str(REPO)], check=True)
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "-r", str(REPO / "requirements-inference.txt")], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                        "yt-dlp[default]"], check=True)
        # Runtime JavaScript diperlukan untuk dukungan YouTube melalui yt-dlp.
        os.environ["PATH"] = str(Path.home() / ".deno" / "bin") + os.pathsep + os.environ["PATH"]
        if not shutil.which("deno"):
            subprocess.run(["bash", "-o", "pipefail", "-c",
                            "curl -fsSL https://deno.land/install.sh | sh -s -- --yes"], check=True)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA tersedia:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        print("Setup selesai. Isi formulir pada sel berikutnya.")
    ''')
    cell("code", '''
        # @title 2. Formulir inferensi
        # @markdown **Video dan waktu sumber** — detik, MM:SS, atau HH:MM:SS.
        video_url = "" # @param {type:"string"}
        start_time = "00:00:00" # @param {type:"string"}
        end_time = "00:00:30" # @param {type:"string"}
        output_dir = "/content/outputs/check" # @param {type:"string"}
        max_height = 720 # @param [360, 480, 720, 1080] {type:"raw"}
        # @markdown **Model** — kosongkan checkpoint lokal untuk unduh otomatis dari Hub.
        model_repo = "azunzios/learning" # @param {type:"string"}
        model_file = "model.pt" # @param {type:"string"}
        local_checkpoint = "" # @param {type:"string"}
        # @markdown **Inferensi** — mulai dengan batch 8; turunkan ke 4/2/1 jika VRAM penuh.
        batch_size = 8 # @param [1, 2, 4, 8, 16, 32] {type:"raw"}
        stride_sec = 0.5 # @param {type:"number"}
        use_fp16 = True # @param {type:"boolean"}
        require_gpu = True # @param {type:"boolean"}
        # @markdown Nilai **-1** memakai threshold kalibrasi dari model.
        event_1_threshold = -1.0 # @param {type:"number"}
        event_2_threshold = -1.0 # @param {type:"number"}
        smooth_sec = 1.0 # @param {type:"number"}
        merge_gap_sec = 2.0 # @param {type:"number"}
        min_duration_sec = 1.0 # @param {type:"number"}
        low_ratio = 0.6 # @param {type:"number"}
        # @markdown Opsional: path cookies Netscape yang sudah diunggah ke runtime.
        cookies_file = "" # @param {type:"string"}
        # @markdown Preview menyematkan video ke notebook; gunakan untuk klip pendek.
        preview_video = False # @param {type:"boolean"}
        download_results = True # @param {type:"boolean"}
    ''', form=True)
    cell("code", '''
        # @title 3. Unduh video dan jalankan inferensi
        import json
        import math
        import os
        import torch
        from infer_url import run_pipeline

        if not video_url.strip():
            raise ValueError("Isi video_url pada formulir terlebih dahulu.")
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("Aktifkan runtime GPU, lalu jalankan ulang notebook.")
        thresholds = {}
        for index, value in enumerate((event_1_threshold, event_2_threshold), 1):
            if not math.isfinite(value) or (value != -1 and not 0 <= value <= 1):
                raise ValueError("Threshold harus -1 atau angka antara 0 dan 1.")
            if value != -1:
                thresholds[f"event_{index}"] = value
        # Untuk repo privat, tambahkan HF_TOKEN sebagai Colab Secret (ikon kunci).
        token = os.environ.get("HF_TOKEN")
        if token is None and not local_checkpoint.strip():
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            except Exception:
                token = None
        paths = run_pipeline(
            video_url.strip(), output=output_dir, start_time=start_time, end_time=end_time,
            ckpt=local_checkpoint.strip(), model_repo=model_repo, model_file=model_file,
            batch_size=batch_size, stride=stride_sec, thresholds=thresholds,
            smooth=smooth_sec, merge_gap=merge_gap_sec, min_duration=min_duration_sec,
            low_ratio=low_ratio, max_height=max_height, fp16=use_fp16,
            device="cuda" if require_gpu else None, cookies_file=cookies_file.strip(), token=token,
        )
        result = json.loads(paths["json"].read_text(encoding="utf-8"))
        print(json.dumps(result, indent=2))
        print("Hasil tersimpan:", paths["archive"])
    ''')
    cell("code", '''
        # @title 4. Lihat hasil dan unduh ZIP
        from IPython.display import Image, Video, display
        from google.colab import files

        if "paths" not in globals():
            raise RuntimeError("Jalankan sel inferensi sampai selesai terlebih dahulu.")
        display(Image(filename=str(paths["timeline"])))
        if preview_video:
            display(Video(str(paths["video"]), embed=True))
        print("ZIP berisi clip.mp4, result.json, dan timeline.png.")
        print("Waktu start/end di JSON relatif terhadap klip; source_start/source_end relatif ke sumber.")
        if download_results:
            files.download(str(paths["archive"]))
    ''')
    cell("markdown", """
        **Catatan:** hasil deteksi mengikuti bobot dan threshold model; gunakan video berlabel
        untuk mengukur akurasi. Runtime Colab dapat terputus dan file `/content` bersifat sementara.
        Simpan ZIP sebelum sesi berakhir. Menjalankan ulang dengan `output_dir` yang sama akan
        mengganti hasil sebelumnya. Jika sumber meminta login, unggah cookies Netscape lewat
        panel Files dan isi `cookies_file`; jangan commit cookies ke GitHub.
    """)
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU", "colab": {"provenance": [], "name": "infer_colab.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    target = Path(__file__).resolve().parents[1] / "notebooks" / "infer_colab.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {target.name} ({len(cells)} cells)")


if __name__ == "__main__":
    build_notebook()
