import os
import numpy as np
import pandas as pd

# Menyetel seed agar dataset sintetik dapat diulang dengan hasil persis sama
np.random.seed(42)

def generate_quality_sensor_data(n_samples=2000):
    """
    Membuat dataset sintetik parameter kualitas sensor / data audit.
    Target (y): 0 jika Data Normal/Bersih (85%), 1 jika Data Anomali/Garbage (15%).
    Catatan: Target 'y' HANYA digunakan untuk evaluasi akhir (Unsupervised).
    """
    print(f"Membuat {n_samples} sampel data sintetik audit kualitas sensor...")
    
    n_anomalies = int(n_samples * 0.15)
    n_normals = n_samples - n_anomalies

    # 1. Generate Data Normal (Distribusi Gaussian Terkontrol)
    suhu_n = np.random.normal(loc=25.0, scale=2.0, size=n_normals)
    kelembapan_n = np.random.normal(loc=60.0, scale=5.0, size=n_normals)
    voltase_n = np.random.normal(loc=220.0, scale=3.0, size=n_normals)
    rasio_noise_n = np.random.normal(loc=0.05, scale=0.01, size=n_normals)
    latency_n = np.random.normal(loc=12.0, scale=1.5, size=n_normals)
    throughput_n = np.random.normal(loc=100.0, scale=8.0, size=n_normals)

    # 2. Generate Data Anomali / Garbage (Nilai ekstrem, terdistorsi, atau rusak)
    suhu_a = np.random.uniform(low=45.0, high=90.0, size=n_anomalies)
    kelembapan_a = np.random.uniform(low=10.0, high=30.0, size=n_anomalies)
    voltase_a = np.random.uniform(low=150.0, high=190.0, size=n_anomalies)
    rasio_noise_a = np.random.uniform(low=0.30, high=0.80, size=n_anomalies)
    latency_a = np.random.uniform(low=80.0, high=300.0, size=n_anomalies)
    throughput_a = np.random.uniform(low=5.0, high=30.0, size=n_anomalies)

    # Gabungkan Data Normal dan Anomali
    X_normal = np.column_stack([suhu_n, kelembapan_n, voltase_n, rasio_noise_n, latency_n, throughput_n])
    X_anomaly = np.column_stack([suhu_a, kelembapan_a, voltase_a, rasio_noise_a, latency_a, throughput_a])

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normals + [1] * n_anomalies)

    # Acak posisi data agar tidak terkumpul di bawah
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    df = pd.DataFrame(X, columns=['suhu', 'kelembapan', 'voltase', 'rasio_noise', 'latency', 'throughput'])
    df['is_anomaly'] = y

    os.makedirs('dataset', exist_ok=True)
    filepath = os.path.join('dataset', 'ml_quality_synthetic.csv')
    df.to_csv(filepath, index=False)
    print(f"Dataset kualitas data sintetik berhasil disimpan di: {filepath}")
    print(f"Distribusi Anomali:\n{df['is_anomaly'].value_counts(normalize=True)}\n")

if __name__ == '__main__':
    generate_quality_sensor_data()