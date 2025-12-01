import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib
import gc

from src.config import load_config
from src.dataio.window import make_windows
from src.models.dcenn import TinyDCENN

class FinanceDataset(Dataset):
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
        # Force inputs to Float32 to prevent Double vs Float crash
        X = X.float()
        Y = Y.float()
        
        H = torch.tanh(X @ self.W + self.b)
        I = torch.eye(H.shape[1]).to(X.device)
        self.beta = torch.linalg.solve(H.T @ H + self.ridge * I, H.T @ Y)
        
        del H, I
        torch.cuda.empty_cache()

    def predict(self, X):
        X = X.float() # Ensure float32
        return torch.tanh(X @ self.W + self.b) @ self.beta

def run(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- 1. Data Prep ---")
    csv_path = Path(cfg["paths"]["engineered_csv"])
    df = pd.read_csv(csv_path)
    ts_col = cfg["columns"]["timestamp"]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    
    split = cfg["time"]["split"]
    train_df = df.loc[:split["train_until"]].copy()
    val_df   = df.loc[split["train_until"]:split["val_until"]].copy()
    test_df  = df.loc[split["val_until"]:].copy()
    
    inputs = cfg["features"]["input_features"]
    targets = cfg["features"]["target_features"]
    ctx, hz = cfg["features"]["context_hours"], cfg["features"]["horizon_hours"]
    
    # Scaling
    print("Scaling...")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_scaler.fit(train_df[inputs])
    y_scaler.fit(train_df[targets])
    
    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(y_scaler, out_dir / "y_scaler.joblib")
    
    train_df[inputs] = x_scaler.transform(train_df[inputs])
    val_df[inputs]   = x_scaler.transform(val_df[inputs])
    test_df[inputs]  = x_scaler.transform(test_df[inputs])
    
    print("--- 2. Windows ---")
    Xtr, Ytr, _ = make_windows(train_df, inputs, targets, ctx, hz)
    Xva, Yva, _ = make_windows(val_df, inputs, targets, ctx, hz)
    Xte, _, te_idx = make_windows(test_df, inputs, targets, ctx, hz)
    
    print("Freeing DataFrame memory...")
    del df, train_df, val_df, test_df
    gc.collect()

    def scale_targets(Y_arr, scaler):
        N, H, C = Y_arr.shape
        Y_flat = Y_arr.reshape(-1, C)
        Y_scaled = scaler.transform(Y_flat).reshape(N, H, C)
        return Y_scaled.astype(np.float32)

    Ytr_scaled = scale_targets(Ytr, y_scaler)
    Yva_scaled = scale_targets(Yva, y_scaler)
    
    print("--- 3. Train Encoder ---")
    enc = TinyDCENN(len(inputs), cfg["training"]["encoder"]["latent_channels"]).to(device)
    head = nn.Linear(cfg["training"]["encoder"]["latent_channels"], hz * len(targets)).to(device)
    optim = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=0.001)
    
    # Use smaller batch size to be safe
    train_dl = DataLoader(FinanceDataset(Xtr, Ytr_scaled), batch_size=256, shuffle=True)
    val_dl   = DataLoader(FinanceDataset(Xva, Yva_scaled), batch_size=256, shuffle=False)
    criterion = nn.L1Loss()
    
    for ep in range(cfg["training"]["encoder"]["epochs"]):
        enc.train()
        l_sum = 0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {ep+1}"):
            xb, yb = xb.to(device), yb.to(device)
            z = enc(xb)
            pred = head(z).reshape(yb.shape)
            loss = criterion(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            l_sum += loss.item()
        
        enc.eval()
        v_sum = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_pred = head(enc(xb)).reshape(yb.shape)
                v_sum += criterion(val_pred, yb).item()
        print(f"Loss Train: {l_sum/len(train_dl):.4f} | Val: {v_sum/len(val_dl):.4f}")

    del train_dl, val_dl, Xva, Yva, Ytr_scaled, Yva_scaled
    gc.collect()

    print("--- 4. Fit ELMs ---")
    enc.eval()
    
    def get_latents(X_in):
        Z_list = []
        for i in range(0, len(X_in), 512):
            with torch.no_grad():
                batch = torch.from_numpy(X_in[i:i+512]).float().to(device) # Force Float
                Z_list.append(enc(batch).cpu().numpy())
        return np.concatenate(Z_list)

    Ztr = get_latents(Xtr)
    Zte = get_latents(Xte)

    print("Preparing Recursive Price Targets...")
    close_idx = inputs.index("Close")
    
    # 1. Anchors
    Xtr_last = Xtr[:, -1, :, 0, 0]
    Xtr_last_raw = x_scaler.inverse_transform(Xtr_last)
    price_anchor_tr_raw = Xtr_last_raw[:, close_idx] 
    
    dummy_tr = np.zeros((len(price_anchor_tr_raw), len(targets)))
    dummy_tr[:, 0] = price_anchor_tr_raw
    price_anchor_tr_scaled = y_scaler.transform(dummy_tr)[:, 0] 
    
    # 2. Targets
    Ytr_scaled_fit = scale_targets(Ytr, y_scaler)
    Y_close_tr_scaled = Ytr_scaled_fit[:, :, 0]
    Y_vol_tr_scaled   = Ytr_scaled_fit[:, :, 1]
    
    # 3. Diff
    price_seq_tr = np.concatenate([price_anchor_tr_scaled[:, None], Y_close_tr_scaled], axis=1)
    diff_target_tr = np.diff(price_seq_tr, axis=1)

    print("Deleting Training Data...")
    del Xtr, Ytr, Xtr_last, Xtr_last_raw, price_seq_tr
    gc.collect()

    preds_scaled = np.zeros((len(Xte), hz, len(targets)))
    
    for i, name in enumerate(targets):
        print(f"  Fitting {name}...")
        elm = ELM(Ztr.shape[1], hz).to(device)
        
        if name == "Close":
            # --- FIX: FORCE FLOAT32 ---
            elm.fit(torch.from_numpy(Ztr).float().to(device), 
                    torch.from_numpy(diff_target_tr).float().to(device))
            
            diff_pred = elm.predict(torch.from_numpy(Zte).float().to(device)).cpu().numpy()
            
            # Reconstruct
            Xte_last = Xte[:, -1, :, 0, 0]
            Xte_last_raw = x_scaler.inverse_transform(Xte_last)
            price_anchor_te_raw = Xte_last_raw[:, close_idx]
            
            dummy_te = np.zeros((len(price_anchor_te_raw), len(targets)))
            dummy_te[:, 0] = price_anchor_te_raw
            price_anchor_te_scaled = y_scaler.transform(dummy_te)[:, 0]
            
            preds_scaled[:, :, i] = price_anchor_te_scaled[:, None] + np.cumsum(diff_pred, axis=1)
            
        else:
            # --- FIX: FORCE FLOAT32 ---
            elm.fit(torch.from_numpy(Ztr).float().to(device), 
                    torch.from_numpy(Y_vol_tr_scaled).float().to(device))
            preds_scaled[:, :, i] = elm.predict(torch.from_numpy(Zte).float().to(device)).cpu().numpy()

    print("--- 5. Saving ---")
    N, H, C = preds_scaled.shape
    flat = preds_scaled.reshape(-1, C)
    inverse = y_scaler.inverse_transform(flat).reshape(N, H, C)
    inverse = np.maximum(inverse, 0.0)
    
    cols = []
    for h in range(hz):
        for name in targets:
            cols.append(f"{name}+h{h+1}")
            
    out_data = []
    for h in range(hz):
        out_data.append(inverse[:, h, :])
    out_data = np.hstack(out_data)
    
    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_out = pd.DataFrame(out_data, index=te_idx, columns=cols)
    df_out.to_parquet(out_dir / "finance_forecast.parquet")
    print(f"Done. Saved to {out_dir}/finance_forecast.parquet")

if __name__ == "__main__":
    run("configs/finance_1m.yaml")