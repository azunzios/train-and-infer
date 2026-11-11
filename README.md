# Video Event Detection

Training dan inferensi video menggunakan VideoMAEv2. Proses dan grafik menggunakan
nama netral `event_1`, `event_2`, dan seterusnya. Hasil JSON memakai kembali nama
asli checkpoint. Urutan output dan threshold kalibrasi tetap dipertahankan.

## Colab: langsung inferensi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azunzios/train-and-infer/blob/main/notebooks/infer_colab.ipynb)

1. Buka notebook di atas dan pilih **Runtime → Change runtime type → GPU**.
2. Upload video ke panel Files, lalu isi `video_url` dengan path seperti
   `/content/test.mp4`. Alternatifnya, masukkan URL langsung ke berkas video.
3. Atur `output_dir`, batch, dan threshold opsional pada formulir `@param`, lalu
   jalankan **satu-satunya sel**. Setup, inferensi, grafik, dan unduh hasil berjalan
   dalam sel yang sama.

Notebook menyertakan kode inferensi sehingga dapat diunggah langsung ke Colab
tanpa clone repo. Checkpoint diunduh otomatis dari
[`azunzios/learning`](https://huggingface.co/azunzios/learning), atau gunakan
`local_checkpoint` untuk model yang sudah diunggah. Tidak perlu dataset,
checkpoint backbone tambahan, atau training ulang.

Path lokal langsung diproses. Untuk URL, notebook menggunakan **wget** dan
memerlukan link langsung ke video, bukan halaman pemutar video. Seluruh video
diproses; formulir tidak menyediakan pemotongan waktu atau perubahan resolusi.
Alur Colab ini tidak memasang atau menjalankan yt-dlp, FFmpeg CLI, atau Deno.
Preview bersifat opsional dan mengikuti format/codec file asli; gunakan MP4 yang
kompatibel dengan browser jika ingin menampilkan video di notebook.

Untuk repo model privat, simpan token baca sebagai Colab Secret bernama `HF_TOKEN`
dan izinkan akses notebook. Jangan memasukkan token ke sel kode yang akan dibagikan.
Notebook tidak membuat ZIP. `result.json` mengembalikan label dan threshold ke
nama asli checkpoint; tampilan proses dan grafik tetap menggunakan nama netral.
Hanya satu segmen dengan confidence `peak` tertinggi yang dipotong menjadi
`best_clip.mp4`; jika peak sama, dipilih skor mean tertinggi. Klip dibuat melalui
OpenCV dan tidak menyertakan audio. Jika tidak ada deteksi, klip tidak dibuat.
Unduhan otomatis mencakup JSON dan satu klip tersebut. Grafik `timeline.png` dan
`run_config.json` tetap tersimpan di folder output.
Catatan konfigurasi memuat hash checkpoint, threshold efektif, pengaturan video,
dan versi library untuk membandingkan dua run. Threshold formulir `-1` memakai
kalibrasi model; nilai `0` meloloskan semua skor. Simpan hasil sebelum runtime
berakhir karena berkas `/content` bersifat sementara.

## Instalasi lokal untuk RTX 4070

**Ya, PyTorch dengan dukungan CUDA diperlukan untuk inferensi di RTX 4070.**
Gunakan Python **3.11 atau 3.12 64-bit**, driver NVIDIA yang mendukung build CUDA
yang dipilih, dan environment Python terpisah. Contoh berikut memakai wheel CUDA
12.8. Perbarui driver NVIDIA terlebih dahulu; periksa GPU dengan `nvidia-smi`.

| Komponen | Kebutuhan |
| --- | --- |
| PyTorch (`torch`) | Wajib; pasang build CUDA agar GPU digunakan. |
| Driver NVIDIA | Wajib untuk inferensi GPU. |
| CUDA Toolkit / cuDNN terpisah | Tidak diperlukan untuk menjalankan wheel PyTorch siap pakai; runtime disediakan paket. |
| FFmpeg dan ffprobe | Dipakai CLI `infer_url.py` untuk unduh/potong/konversi; Colab satu sel menggunakan wget. |
| NumPy, OpenCV, Matplotlib, huggingface_hub | Dipasang lewat `requirements-inference.txt`. |
| yt-dlp + runtime JavaScript | Dipakai CLI `infer_url.py`; tidak diperlukan oleh notebook inferensi satu sel. |
| torchvision, torchaudio, TensorFlow | Tidak dipakai oleh kode inferensi ini. |

### Windows PowerShell

Pasang Python dan Git jika belum tersedia. Pasang FFmpeg dan Deno:

```powershell
winget install --id Gyan.FFmpeg --exact
winget install --id DenoLand.Deno --exact
```

Buka PowerShell baru agar PATH diperbarui, lalu:

```powershell
git clone https://github.com/azunzios/train-and-infer.git
cd train-and-infer
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU belum tersedia')"
ffmpeg -version
ffprobe -version
deno --version
```

Tidak perlu mengaktifkan venv; semua perintah memakai interpreter `.venv` langsung.
Hasil pemeriksaan CUDA harus `True` dan nama GPU sesuai kartu Anda. Jika `False`,
periksa driver, interpreter yang dipakai, dan apakah wheel yang terpasang adalah
build CUDA. Pilihan wheel resmi lainnya tersedia di
[panduan instalasi PyTorch](https://pytorch.org/get-started/locally/).

### Linux / Ubuntu

Setelah driver NVIDIA dan Python 3.11/3.12 tersedia:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg python3-venv
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"
git clone https://github.com/azunzios/train-and-infer.git
cd train-and-infer
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU belum tersedia')"
```

### Unduh video lalu inferensi

Contoh PowerShell, dengan model diunduh otomatis dari Hugging Face:

```powershell
.\.venv\Scripts\python.exe infer_url.py --url "URL_VIDEO" --start-time "00:01:00" --end-time "00:01:30" --output "outputs/check" --batch 8 --device cuda
```

Di Linux gunakan `.venv/bin/python` sebagai interpreter. Pilihan tambahan:

```powershell
.\.venv\Scripts\python.exe infer_url.py --help
.\.venv\Scripts\python.exe infer_url.py --url "URL_VIDEO" --start-time 60 --end-time 90 --output "outputs/check2" --ckpt "models/model.pt" --thr event_1=0.6 event_2=0.5 --batch 4 --device cuda
```

Jika perlu autentikasi Hub, gunakan `hf auth login` dari venv atau atur `HF_TOKEN`
di environment. Untuk video yang memerlukan cookies, tambahkan
`--cookies-file "path/to/session.cookies.txt"`. FFmpeg berada di PATH, bukan paket
Python bernama `ffmpeg`. yt-dlp menggunakan FFmpeg untuk interval dan konversi;
sumber tertentu tetap dapat memerlukan unduhan lebih panjang dari intervalnya.

Folder `--output` berisi:

| Berkas | Isi |
| --- | --- |
| `clip.mp4` | Video interval terpilih, H.264/AAC untuk kompatibilitas preview. |
| `result.json` | Segmen per kelas, threshold, skor peak/mean, dan offset sumber. |
| `timeline.png` | Grafik skor per kelas dan segmen terdeteksi. |
| `inference_results.zip` | Ketiga berkas di atas untuk diunduh. |

`start`/`end` pada segmen JSON dihitung dari awal `clip.mp4`.
`source_start`/`source_end` menambahkan offset `start_time` agar lokasi pada sumber
jelas. Penentuan waktu mengikuti hasil ekstraksi/decoding dan resolusi frame model,
bukan jaminan anotasi tepat pada setiap frame sumber. Menjalankan ulang dengan
folder output sama mengganti hasil sebelumnya.

### Inferensi video lokal yang sudah tersedia

```powershell
.\.venv\Scripts\hf.exe download azunzios/learning model.pt --local-dir models
.\.venv\Scripts\python.exe infer.py --ckpt models/model.pt --video "video.mp4" --out-dir results --batch 8 --device cuda --plot
```

Mode ini tidak mengunduh atau memotong video. `infer.py` menghasilkan JSON dan,
dengan `--plot`, PNG. FP16 digunakan otomatis di CUDA; tambahkan `--fp32` untuk
menonaktifkannya. Mulai dengan batch **8** pada RTX 4070; jika kehabisan VRAM,
turunkan menjadi **4**, **2**, atau **1**. Pemakaian memori dan kecepatan bergantung
pada video dan konfigurasi model. Resolusi input jaringan mengikuti checkpoint,
sehingga unduhan 4K tidak diperlukan untuk pemeriksaan awal.

## Training dan pemeliharaan notebook

Notebook training tetap tersedia di [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).
Untuk dependency training lokal gunakan `python -m pip install -r requirements.txt`.
Folder dataset lama `positive/og` dan `positive/sq` dipetakan ke `event_1` dan
`event_2` sesuai urutan output. Checkpoint ekspor baru menggunakan metadata netral;
checkpoint lama dinormalisasi oleh loader saat inferensi.

Setelah mengubah sumber, bangun ulang notebook:

```bash
python tools/build_notebook.py
python tools/build_inference_notebook.py
```

Pemeriksaan regresi (termasuk video sintetis yang dilayani lewat HTTP lokal,
unduh/potong FFmpeg, inferensi CPU, dan isi ZIP):

```bash
python -m unittest discover -s tests -v
```

Referensi resmi: [PyTorch](https://pytorch.org/get-started/locally/),
[yt-dlp](https://github.com/yt-dlp/yt-dlp#usage-and-options),
[runtime JavaScript yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/EJS),
[FFmpeg](https://ffmpeg.org/download.html), dan
[instalasi Deno](https://docs.deno.com/runtime/getting_started/installation/).
