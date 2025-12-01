import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib

from src.config import load_config
from src.dataio.preprocess import build_master
from src.dataio.window import make_windows
from src.models.dcenn import TinyDCENN

class WeatherDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

class ELM(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, ridge=1e-2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, hidden)*0.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden)*0.5, requires_grad=False)
        self.ridge = ridge
        self.beta = None
    def fit(self, X, Y):
        H = torch.tanh(X @ self.W + self.b)
        I = torch.eye(H.shape[1]).to(X.device)
        self.beta = torch.linalg.solve(H.T @ H + self.ridge * I, H.T @ Y)
    def predict(self, X):
        return torch.tanh(X @ self.W + self.b) @ self.beta

def run(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data Prep
    print("--- 1. Data Preparation ---")
    train_df, val_df, test_df = build_master(cfg)
    
    inputs = cfg["features"]["input_features"]
    targets = cfg["features"]["target_features"]
    ctx, hz = cfg["features"]["context_hours"], cfg["features"]["horizon_hours"]
    
    # Robust Scaling
    all_features = sorted(list(set(inputs + targets)))
    main_scaler = StandardScaler()
    
    # Fit on Train, Apply to Val/Test
    train_df_scaled = train_df.copy()
    val_df_scaled   = val_df.copy()
    test_df_scaled  = test_df.copy()
    
    train_df_scaled[all_features] = main_scaler.fit_transform(train_df[all_features])
    val_df_scaled[all_features]   = main_scaler.transform(val_df[all_features])
    test_df_scaled[all_features]  = main_scaler.transform(test_df[all_features])
    
    # Y Scaler for inverse transform later
    y_scaler = StandardScaler()
    y_scaler.fit(train_df[targets])
    
    # 2. Windowing
    print("--- 2. Creating Windows ---")
    Xtr, Ytr, _ = make_windows(train_df_scaled, inputs, targets, ctx, hz)
    Xva, Yva, _ = make_windows(val_df_scaled,   inputs, targets, ctx, hz) # <--- ADDED VAL
    Xte, _, te_idx = make_windows(test_df_scaled, inputs, targets, ctx, hz)
    
    # 3. Train dCeNN Encoder
    print("--- 3. Training dCeNN Encoder ---")
    enc = TinyDCENN(len(inputs), cfg["training"]["encoder"]["latent_channels"]).to(device)
    output_dim = hz * len(targets)
    head = nn.Linear(cfg["training"]["encoder"]["latent_channels"], output_dim).to(device)
    
    optim = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=0.001)
    
    # Datasets
    train_ds = WeatherDataset(Xtr, Ytr)
    val_ds   = WeatherDataset(Xva, Yva) # <--- ADDED VAL DS
    
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    
    criterion = nn.L1Loss()

    for ep in range(cfg["training"]["encoder"]["epochs"]):
        # Train Loop
        enc.train()
        train_loss_sum = 0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {ep+1} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            z = enc(xb)
            pred = head(z).reshape(yb.shape)
            loss = criterion(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            train_loss_sum += loss.item()
            
        # Validation Loop (No Grad)
        enc.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                z = enc(xb)
                pred = head(z).reshape(yb.shape)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item()
        
        # PRINT BOTH LOSSES
        t_loss = train_loss_sum / len(train_dl)
        v_loss = val_loss_sum / len(val_dl)
        print(f"  > Epoch {ep+1}: train_loss={t_loss:.4f}  val_loss={v_loss:.4f}")

    # 4. Fit ELM Heads
    print("--- 4. Fitting ELMs (Reasoning Prep) ---")
    enc.eval()
    with torch.no_grad():
        # Get latents for Train and Test
        # (We don't strictly need Val latents for ELM fitting unless we tuned hyperparameters)
        Ztr_flat = []
        for i in range(0, len(Xtr), 256):
            Ztr_flat.append(enc(torch.from_numpy(Xtr[i:i+256]).to(device)).cpu().numpy())
        Ztr_flat = np.concatenate(Ztr_flat)
        
        Zte_flat = []
        for i in range(0, len(Xte), 256):
            Zte_flat.append(enc(torch.from_numpy(Xte[i:i+256]).to(device)).cpu().numpy())
        Zte_flat = np.concatenate(Zte_flat)

    preds_scaled = np.zeros((len(Xte), hz, len(targets)))
    
    for i, name in enumerate(targets):
        print(f"  Fitting {name}...")
        elm = ELM(Ztr_flat.shape[1], hz).to(device)
        elm.fit(torch.from_numpy(Ztr_flat).to(device), torch.from_numpy(Ytr[:,:,i]).to(device))
        p = elm.predict(torch.from_numpy(Zte_flat).to(device)).cpu().numpy()
        preds_scaled[:,:,i] = p

    # 5. Inverse Scale & Save
    print("--- 5. Saving Raw Predictions ---")
    N, H, C = preds_scaled.shape
    flat = preds_scaled.reshape(-1, C)
    inverse = y_scaler.inverse_transform(flat).reshape(N, H, C)
    
    cols = []
    for h in range(hz):
        for i, name in enumerate(targets):
            cols.append(f"{name}+h{h+1}")
            
    out_data = []
    for h in range(hz):
        out_data.append(inverse[:, h, :])
    out_data = np.hstack(out_data)
    
    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_out = pd.DataFrame(out_data, index=te_idx, columns=cols)
    df_out.to_parquet(out_dir / "raw_weather.parquet")
    print("Done.")

if __name__ == "__main__":
    run("configs/weather_full.yaml")