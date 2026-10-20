import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from common import set_seed, load_and_preprocess_data

SEEDS = [42, 123, 999]
DATA_PATH = os.path.join('dataset', 'ml_quality_synthetic.csv')
DEVICE = torch.device('cpu')

# Arsitektur Deep Autoencoder untuk Rekonstruksi Fitur
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        # Encoder: Memampatkan dimensi data ke bottleneck tersembunyi
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3) # Bottleneck layer (3 dimensi)
        )
        # Decoder: Mengrekonstruksi data kembali ke dimensi awal
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def run_dl_experiments():
    print("=== MENJALANKAN EXPERIMEN DEEP LEARNING (PyTorch Autoencoder) ===")
    
    metrics_summary = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    last_cm = None
    last_history = None

    for seed in SEEDS:
        set_seed(seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = load_and_preprocess_data(DATA_PATH, seed=seed)

        # Murni Unsupervised: Training Autoencoder HANYA menggunakan data normal dari X_train
        X_train_normal = X_train[y_train == 0]

        train_ds = TensorDataset(torch.FloatTensor(X_train_normal), torch.FloatTensor(X_train_normal))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = DeepAutoencoder(input_dim=X_train.shape[1]).to(DEVICE)

        # Loss function menggunakan Mean Squared Error (MSE) antara Input dan Hasil Rekonstruksi
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

        best_val_loss = float('inf')
        patience = 12
        patience_counter = 0
        best_model_weights = None

        history = {'train_loss': [], 'val_loss': []}

        # Training Loop
        for epoch in range(120):
            model.train()
            train_loss = 0.0
            for X_b, _ in train_loader:
                X_b = X_b.to(DEVICE)
                optimizer.zero_grad()
                recon_b = model(X_b)
                loss = criterion(recon_b, X_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)
            
            train_loss /= len(train_loader.dataset)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, _ in val_loader:
                    X_b = X_b.to(DEVICE)
                    recon_b = model(X_b)
                    loss = criterion(recon_b, X_b)
                    val_loss += loss.item() * X_b.size(0)
            
            val_loss /= len(val_loader.dataset)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Early Stopping Check berdasarkan Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_model_weights)
        last_history = history

        # --- TUNING THRESHOLD REKONSTRUKSI MSE PADA VALIDATION SET ---
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            recon_val = model(X_val_tensor)
            # Hitung MSE tiap sampel: jika MSE tinggi -> Data tidak cocok dengan pola normal -> Anomali
            val_mse = torch.mean((X_val_tensor - recon_val) ** 2, dim=1).cpu().numpy()

        best_thresh = np.percentile(val_mse, 85)
        best_val_f1 = 0.0
        for thresh in np.linspace(val_mse.min(), val_mse.max(), 50):
            score = f1_score(y_val, (val_mse >= thresh).astype(int), zero_division=0)
            if score > best_val_f1:
                best_val_f1 = score
                best_thresh = thresh

        # --- EVALUASI AKHIR PADA TEST SET ---
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
            recon_test = model(X_test_tensor)
            test_mse = torch.mean((X_test_tensor - recon_test) ** 2, dim=1).cpu().numpy()
            test_preds = (test_mse >= best_thresh).astype(int)

        metrics_summary['acc'].append(accuracy_score(y_test, test_preds))
        metrics_summary['prec'].append(precision_score(y_test, test_preds, zero_division=0))
        metrics_summary['rec'].append(recall_score(y_test, test_preds, zero_division=0))
        metrics_summary['f1'].append(f1_score(y_test, test_preds, zero_division=0))
        metrics_summary['auc'].append(roc_auc_score(y_test, test_mse))
        
        last_cm = confusion_matrix(y_test, test_preds)

    print("\n--- HASIL EVALUASI DEEP LEARNING (3 SEED) ---")
    for metric, values in metrics_summary.items():
        print(f"{metric.upper():<5}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(last_history['train_loss'], label='Train Loss')
    plt.plot(last_history['val_loss'], label='Val Loss')
    plt.title('Kurva Training & Validation Reconstruction Loss (Autoencoder)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'dl_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(last_cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
    plt.title('Deep Autoencoder Confusion Matrix')
    plt.ylabel('Kenyataan (True)')
    plt.xlabel('Prediksi (Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'dl_cm.png'))
    plt.close()
    print("Grafik kurva loss dan confusion matrix disimpan di direktori results/\n")

if __name__ == '__main__':
    run_dl_experiments()