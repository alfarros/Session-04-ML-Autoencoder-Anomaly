# Session 04: Unsupervised Anomaly Detection using Deep Autoencoder

Sistem pendeteksi anomali (*Garbage In*) otomatis tanpa label awal berbasis PyTorch Deep Autoencoder untuk melakukan audit kualitas data sensor secara otonom.

## 1. Dari Materi Pemateri vs Pengembangan Mandiri
* **Materi Pemateri (Taniza Tajuddin):** Membahas taksonomi Machine Learning, jenis-jenis *Unsupervised Learning*, serta bahaya prinsip *"Garbage In, Garbage Out"* dalam sistem cerdas.
* **Pengembangan Mandiri:** Membangun model *Deep Autoencoder* (PyTorch Encoder-Decoder) yang dilatih secara *Unsupervised* untuk mendeteksi data kotor/terdistorsi berdasarkan error rekonstruksi fitur (*Reconstruction MSE*).

## 2. Masalah dan Tujuan
Menyaring data kotor (*Garbage In*) secara otomatis tanpa bantuan label manual sangat krusial agar pipeline ML tidak rusak. Penelitian ini bertujuan memanfaatkan arsitektur Autoencoder untuk mengisolasi sampel anomali melalui ambang batas kesalahan rekonstruksi.

## 3. Spesifikasi Dataset
* **Sumber:** Dataset Sintetik (`dataset/ml_quality_synthetic.csv`) yang disintesis melalui skrip `dataset/generate_dataset.py`.
* **Ukuran:** 2.000 sampel audit data sensor.
* **Fitur Input:** `suhu`, `kelembapan`, `voltase`, `rasio_noise`, `latency`, `throughput`.
* **Target (Hanya untuk Evaluasi Test):** `is_anomaly` (Binary: `0` = Normal (85%), `1` = Anomali/Garbage (15%)).

## 4. Metodologi & Pencegahan Leakage
1. **Unsupervised Training:** Autoencoder dilatih **hanya menggunakan sampel data normal** dari data Train, sehingga model terbiasa merekonstruksi data bersih.
2. **Pencegahan Data Leakage:** Split Train/Val/Test (70/15/15) dilakukan di awal. `StandardScaler` di-fit **hanya** pada data Train normal.
3. **Threshold Selection:** Ambang batas *Reconstruction MSE* untuk memisahkan Normal dan Anomali ditentukan secara dinamis pada data Validation.
4. **Reproducibility:** Eksperimen diulang pada 3 Seed acak (`42`, `123`, `999`).

## 5. Arsitektur / Alur Sistem
[Input Sensor (6 Fitur)] ──► [Encoder: 6 -> 16 -> 8 -> Bottleneck (3)]
   │
[Output Rekonstruksi (6)] ◄── [Decoder: 3 -> 8 -> 16 -> 6]──►[Hitung Reconstruction MSE] ──► [Thresholding Validation] ──► [Prediksi: Normal / Anomali]

## 6. Cara Menjalankan Perintah
**Di Vscode (PowerShell / Command Prompt):**
```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python dataset/generate_dataset.py

# 4. Jalankan eksperimen Baseline (Random Forest)
python src/01_baseline.py

# 5. Jalankan eksperimen Deep Learning (PyTorch)
python src/02_train_dl.py