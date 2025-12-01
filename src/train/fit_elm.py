import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

from ..config import load_config
from ..dataio.window import cache_dcenn_windows
from ..models.dcenn import TinyDCENN

class RandomFeatureELM(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, act="tanh", ridge_lambda=1e-2, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.device = torch.device(device)
        self.dtype = torch.get_default_dtype()
        # Random projection weights (fixed)
        self.W = nn.Parameter(torch.randn(in_dim, hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5, requires_grad=False)
        self.act = act
        self.ridge = float(ridge_lambda)
        self.beta = None
    
    def _phi(self, XH):
        if self.act == "tanh": return torch.tanh(XH)
        if self.act == "relu": return torch.relu(XH)
        return XH
    
    def fit(self, X, Y):
        X = X.to(device=self.device, dtype=self.W.dtype)
        Y = Y.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)
        HtH = H.T @ H
        # Identity matrix for Ridge Regression
        lamI = self.ridge * torch.eye(int(HtH.size(0)), device=H.device, dtype=H.dtype)
        self.beta = torch.linalg.solve(HtH + lamI, H.T @ Y)
    
    def predict(self, X):
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        X = X.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)
        return H @ self.beta

def extract_latents(encoder, X_np, device="cpu", batch_size=256):
    """Runs the dCeNN encoder to get latent vectors z_t"""
    encoder.eval()
    device = torch.device(device)
    X = torch.from_numpy(X_np).to(device)
    N = X.shape[0]
    outs = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X[i : i + batch_size]
            z = encoder(xb)
            outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0)

def run(cfg_path: str):
    cfg = load_config(cfg_path)
    print("[fit_elm] Loading data...")
    
    # 1. Load Data
    Xtr, Ytr, _, _, _, _ = cache_dcenn_windows(cfg)
    
    # 2. Scale Targets
    # We scale targets to N(0,1) so the ELM gradients are balanced across variables
    N_tr, H, C = Ytr.shape
    Ytr_flat = Ytr.reshape(-1, C)
    target_scaler = StandardScaler()
    Ytr_scaled = target_scaler.fit_transform(Ytr_flat).reshape(N_tr, H, C).astype(np.float32)
    print(f"[fit_elm] Target Mean: {target_scaler.mean_}")

    # 3. Extract Latents from Trained Encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = Xtr.shape[2]
    enc = TinyDCENN(in_channels=in_channels, hidden_channels=cfg["training"]["encoder"]["latent_channels"]).to(device)
    
    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    
    print("[fit_elm] Extracting latents...")
    Ztr = extract_latents(enc, Xtr, device=device)
    
    # Standardize Latents (Z) to help the ELM
    mu = Ztr.mean(axis=0, keepdims=True)
    sigma = Ztr.std(axis=0, keepdims=True) + 1e-6
    Ztr_n = (Ztr - mu) / sigma

    # 4. Prepare Inputs with Skip Connections (Lags)
    print("[fit_elm] Building lag features...")
    
    def get_lags(X_tensor, col_idx):
        # Squeeze dimensions to get [N]
        lag1  = X_tensor[:, -1, col_idx, 0, 0]
        lag24 = X_tensor[:, -24, col_idx, 0, 0]
        return np.stack([lag1, lag24], axis=1)

    # Indices from window.py: 9=Wind, 10=Solar, 11=Load, 12=Price
    X_wind_tr  = np.concatenate([Ztr_n, get_lags(Xtr, 9)], axis=1)
    X_solar_tr = np.concatenate([Ztr_n, get_lags(Xtr, 10)], axis=1)
    X_load_tr  = np.concatenate([Ztr_n, get_lags(Xtr, 11)], axis=1)
    
    # Price Input: Latents + Price Lags
    # Note: The "Anchor" Price(t) is effectively lag1
    X_price_tr = np.concatenate([Ztr_n, get_lags(Xtr, 12)], axis=1)

    # 5. Fit Ensembles
    elm_cfg = cfg["training"]["elm"]
    models_wind, models_solar, models_load, models_price = [], [], [], []
    
    def fit_ens(X, Y, model_list, name, custom_lambda=None):
        print(f"Fitting {name} head (Input Dim: {X.shape[1]})...")
        X_t = torch.from_numpy(X.astype(np.float32))
        Y_t = torch.from_numpy(Y.astype(np.float32))
        
        # Use custom lambda if provided (e.g. for Wind peaks), else config default
        lam = custom_lambda if custom_lambda is not None else elm_cfg["ridge_lambda"]
        
        for k in range(elm_cfg.get("ensemble", 1)):
            elm = RandomFeatureELM(X.shape[1], H, hidden=elm_cfg["hidden"], ridge_lambda=lam, seed=k)
            elm.fit(X_t, Y_t)
            model_list.append(elm)

    # --- FITTING LOGIC ---

    # A. WIND: Direct Prediction
    # Using LOW lambda (1e-5) to allow high peaks (reduce bias)
    fit_ens(X_wind_tr,  Ytr_scaled[:,:,0], models_wind,  "Wind", custom_lambda=1e-5)
    
    # B. SOLAR: Direct Prediction
    fit_ens(X_solar_tr, Ytr_scaled[:,:,1], models_solar, "Solar", custom_lambda=1e-2)
    
    # C. LOAD: Direct Prediction
    fit_ens(X_load_tr,  Ytr_scaled[:,:,2], models_load,  "Load", custom_lambda=1e-2)
    
    # D. PRICE: Recursive (Chain) Prediction
    # Goal: Predict the HOURLY CHANGE (Step), not absolute value
    
    # 1. Get Anchor Price(t) from input features (Index 256 is Lag1)
    price_t = X_price_tr[:, 256] # [N]
    
    # 2. Create Sequence: [Price(t), Price(t+1), ... Price(t+12)]
    # We assume index 3 is Price in Ytr_scaled
    price_seq = np.concatenate([price_t[:, None], Ytr_scaled[:,:,3]], axis=1) # [N, H+1]
    
    # 3. Calculate Steps: Diff between t and t+1, t+1 and t+2...
    price_step_diffs = np.diff(price_seq, axis=1) # [N, H]
    
    # 4. Fit on Steps
    fit_ens(X_price_tr, price_step_diffs, models_price, "Price_Steps", custom_lambda=1e-2)

    # 6. Save everything
    out_dir = Path(cfg["paths"]["checkpoints_dir"])
    save_obj = {
        "mu": mu, "sigma": sigma, "target_scaler": target_scaler,
        "elm_wind": models_wind, "elm_solar": models_solar, 
        "elm_load": models_load, "elm_price": models_price
    }
    joblib.dump(save_obj, out_dir / "elms.joblib")
    print(f"[fit_elm] Done. Saved to {out_dir / 'elms.joblib'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); run(args.config)