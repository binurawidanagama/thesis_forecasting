# src/train/fit_elm.py

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import joblib

from ..config import load_config
from ..dataio.preprocess import build_master
from ..dataio.window import make_windows
from ..models.dcenn import TinyDCENN


class RandomFeatureELM(nn.Module):
    """
    Random features + ridge closed-form.
    Fit:     H = phi(X W + b), beta = (H^T H + λI)^{-1} H^T Y
    Predict: Yhat = H beta
    """

    def __init__(self, in_dim, out_dim, hidden=1024, act="tanh", ridge_lambda=1e-2, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.device = torch.device(device)
        self.dtype = torch.get_default_dtype()

        # weight / bias live on the chosen device
        self.W = nn.Parameter(
            torch.randn(in_dim, hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5,
            requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.randn(hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5,
            requires_grad=False,
        )
        self.act = act
        self.ridge = float(ridge_lambda)
        self.beta = None
        self.out_dim = out_dim

    def _phi(self, XH):
        if self.act == "tanh":
            return torch.tanh(XH)
        if self.act == "relu":
            return torch.relu(XH)
        return XH

    def fit(self, X, Y):
        """
        X: [N, in_dim]  (torch)
        Y: [N, out_dim] (torch)
        """
        X = X.to(device=self.device, dtype=self.W.dtype)
        Y = Y.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)  # [N, hidden]

        HtH = H.T @ H
        n = int(HtH.size(0))
        lamI = self.ridge * torch.eye(n, device=H.device, dtype=H.dtype)
        self.beta = torch.linalg.solve(HtH + lamI, H.T @ Y)  # [hidden, out_dim]

    def predict(self, X):
        """
        X: [N, in_dim] torch or numpy
        returns: [N, out_dim] torch
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)
        return H @ self.beta


def extract_latents(encoder, X_np, device="cpu", batch_size=256):
    """
    Run the dCeNN encoder over windows to obtain latent vectors.

    X_np: [N, T, F, 1, 1] numpy
    returns: Z: [N, latent_dim] numpy
    """
    encoder.eval()
    device = torch.device(device)
    X = torch.from_numpy(X_np).to(device)
    N = X.shape[0]
    outs = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X[i : i + batch_size]  # [B,T,F,1,1]
            z = encoder(xb)             # [B, latent_dim]
            outs.append(z.cpu().numpy())

    Z = np.concatenate(outs, axis=0)
    return Z


def run(cfg_path: str):
    cfg = load_config(cfg_path)

    # ---- 1. Load data and build windows (same features/targets as encoder) ----
    feature_cols = [
        # Calendar
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        # Holiday
        "holiday",
        # Capacities
        "cap_wind_mw", "cap_solar_mw",
        # Past targets
        "wind_mw", "solar_mw", "load_mw", "price_eur_mwh",
        # Weather + proxies
        "temperature_2m_C",
        "relative_humidity_2m_pct",
        "wind_speed_100m (m/s)",
        "surface_pressure_hPa",
        "precipitation_mm",
        "shortwave_radiation_Wm2",
        "air_density_kgm3",
        "pv_proxy",
        "wind_power_proxy",
    ]
    target_cols = ["cf_wind", "cf_solar", "load_mw", "price_eur_mwh"]

    ctx = cfg["features"]["context_hours"]
    hz = cfg["features"]["horizon_hours"]

    train_df, val_df, test_df = build_master(cfg)

    Xtr, Ytr, idx_tr = make_windows(train_df, feature_cols, target_cols, ctx, hz)
    Xva, Yva, idx_va = make_windows(val_df,   feature_cols, target_cols, ctx, hz)
    Xte, Yte, idx_te = make_windows(test_df,  feature_cols, target_cols, ctx, hz)

    # ---- 2. Load trained encoder and compute latents ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = Xtr.shape[2]
    enc = TinyDCENN(
        in_channels=in_channels,
        hidden_channels=cfg["training"]["encoder"]["latent_channels"],
    ).to(device)

    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])

    Ztr = extract_latents(enc, Xtr, device=device)
    Zva = extract_latents(enc, Xva, device=device)
    Zte = extract_latents(enc, Xte, device=device)

    # ---- 3. Standardize latents (μ,σ from train) ----
    mu = Ztr.mean(axis=0, keepdims=True)
    sigma = Ztr.std(axis=0, keepdims=True) + 1e-6

    Ztr_n = (Ztr - mu) / sigma
    Zva_n = (Zva - mu) / sigma
    Zte_n = (Zte - mu) / sigma  # unused here, but stored for completeness

    # ---- 4. Prepare targets ----
    # Y*: [N, H, 4] -> indices: 0: cf_wind, 1: cf_solar, 2: load_mw, 3: price
    y_cf_wind_tr  = Ytr[:, :, 0]
    y_cf_solar_tr = Ytr[:, :, 1]
    y_load_tr     = Ytr[:, :, 2]
    y_price_tr    = Ytr[:, :, 3]

    # ---- 5. Build extra price features (last price, price-24h) ----
    # for train set
    p_train = train_df["price_eur_mwh"]
    p_train_shift24 = p_train.shift(24)

    last_price_tr = p_train.reindex(idx_tr).values  # y_t
    price_24h_tr  = p_train_shift24.reindex(idx_tr)
    price_24h_tr  = price_24h_tr.fillna(p_train.reindex(idx_tr)).values  # fallback to last price if missing

    extra_price_tr = np.stack([last_price_tr, price_24h_tr], axis=1)  # [N_tr, 2]

    # same for val (even if we don't strictly need it for fitting, useful for diagnostics)
    p_val = val_df["price_eur_mwh"]
    p_val_shift24 = p_val.shift(24)

    last_price_va = p_val.reindex(idx_va).values
    price_24h_va  = p_val_shift24.reindex(idx_va)
    price_24h_va  = price_24h_va.fillna(p_val.reindex(idx_va)).values

    extra_price_va = np.stack([last_price_va, price_24h_va], axis=1)  # [N_va, 2]

    # ---- 6. Instantiate and train ELMs (with ensemble) ----
    elm_cfg = cfg["training"]["elm"]
    hidden = elm_cfg["hidden"]
    ridge = elm_cfg["ridge_lambda"]
    ensemble_size = elm_cfg.get("ensemble", 1)

    # Common train inputs
    X_wind_tr  = Ztr_n
    X_solar_tr = Ztr_n
    X_load_tr  = Ztr_n
    X_price_tr = np.concatenate([Ztr_n, extra_price_tr], axis=1)  # latents + 2 price lags

    in_dim_wind   = X_wind_tr.shape[1]
    in_dim_solar  = X_solar_tr.shape[1]
    in_dim_load   = X_load_tr.shape[1]
    in_dim_price  = X_price_tr.shape[1]
    out_dim = hz  # same for all heads: H horizons

    models_wind  = []
    models_solar = []
    models_load  = []
    models_price = []

    # convert Y to torch upfront
    Y_cf_wind_tr_t  = torch.from_numpy(y_cf_wind_tr.astype(np.float32))
    Y_cf_solar_tr_t = torch.from_numpy(y_cf_solar_tr.astype(np.float32))
    Y_load_tr_t     = torch.from_numpy(y_load_tr.astype(np.float32))
    Y_price_tr_t    = torch.from_numpy(y_price_tr.astype(np.float32))

    X_wind_tr_t  = torch.from_numpy(X_wind_tr.astype(np.float32))
    X_solar_tr_t = torch.from_numpy(X_solar_tr.astype(np.float32))
    X_load_tr_t  = torch.from_numpy(X_load_tr.astype(np.float32))
    X_price_tr_t = torch.from_numpy(X_price_tr.astype(np.float32))

    for k in range(ensemble_size):
        seed = k

        # Wind CF
        elm_w = RandomFeatureELM(
            in_dim=in_dim_wind,
            out_dim=out_dim,
            hidden=hidden,
            act="tanh",
            ridge_lambda=ridge,
            seed=seed,
            device="cpu",
        )
        elm_w.fit(X_wind_tr_t, Y_cf_wind_tr_t)
        models_wind.append(elm_w)

        # Solar CF
        elm_s = RandomFeatureELM(
            in_dim=in_dim_solar,
            out_dim=out_dim,
            hidden=hidden,
            act="tanh",
            ridge_lambda=ridge,
            seed=seed + 100,
            device="cpu",
        )
        elm_s.fit(X_solar_tr_t, Y_cf_solar_tr_t)
        models_solar.append(elm_s)

        # Load
        elm_l = RandomFeatureELM(
            in_dim=in_dim_load,
            out_dim=out_dim,
            hidden=hidden,
            act="tanh",
            ridge_lambda=ridge,
            seed=seed + 200,
            device="cpu",
        )
        elm_l.fit(X_load_tr_t, Y_load_tr_t)
        models_load.append(elm_l)

        # Price
        elm_p = RandomFeatureELM(
            in_dim=in_dim_price,
            out_dim=out_dim,
            hidden=hidden,
            act="tanh",
            ridge_lambda=ridge,
            seed=seed + 300,
            device="cpu",
        )
        elm_p.fit(X_price_tr_t, Y_price_tr_t)
        models_price.append(elm_p)

    # ---- 7. Save ELMs + latent normalization stats ----
    out_dir = Path(cfg["paths"]["checkpoints_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "elms.joblib"

    save_obj = {
        "mu": mu,
        "sigma": sigma,
        "elm_wind": models_wind,
        "elm_solar": models_solar,
        "elm_load": models_load,
        "elm_price": models_price,
    }

    joblib.dump(save_obj, save_path)
    print(f"[fit_elm] Saved ELMs and latent stats to {save_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
