"""
Run full ENERGY forecasting pipeline (dCeNN + ELM), save metrics like baselines.

Version: NO early stopping (as requested).
Includes:
- Latent normalization (Z StandardScaler) for ELM conditioning
- Adaptive ridge in ELM.fit() to avoid singular-matrix crashes
- StandardScaler by default (RobustScaler optional via YAML)
- Restores ELM init scale (default 0.5) to match older good runs more closely

THESIS-GRADE FIXES ADDED (reporting + artifacts + RAM sampling):
- Train vs Deploy parameter counts (Train_Params, Deploy_Params)
- Save deployment artifacts WITHOUT the linear head (so Size_MB is honest)
- Save ELM W,b,beta (deployment uses them; no “hidden” params)
- Peak RAM sampling inside training batches every N steps

Examples:
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 24  --horizon 12
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 24  --horizon 24
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 24  --horizon 72

python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 72  --horizon 12
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 72  --horizon 24
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 72  --horizon 72

python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 168 --horizon 12
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 168 --horizon 24
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 168 --horizon 72
"""

import os
import gc
import time
import json
import argparse
import random
from pathlib import Path

import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config
from src.dataio.preprocess import build_master
from src.dataio.window import make_windows
from src.models.dcenn import TinyDCENN


# -----------------------------
# Utils
# -----------------------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_process_metrics():
    """Return (rss_mb, cpu_time_s) for current python process."""
    p = psutil.Process(os.getpid())
    with p.oneshot():
        mem_mb = p.memory_info().rss / (1024 * 1024)
        cpu = p.cpu_times()
        cpu_time_s = float(cpu.user + cpu.system)
    return float(mem_mb), float(cpu_time_s)


class ResourceMonitor:
    def __init__(self):
        self.peak_ram_mb = get_process_metrics()[0]

    def update(self):
        self.peak_ram_mb = max(self.peak_ram_mb, get_process_metrics()[0])


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    smape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
    return mae, rmse, smape


def squeeze_X(X: np.ndarray) -> np.ndarray:
    """Return [N,T,F] from [N,T,F,1,1] or [N,T,F,1] or [N,T,F]."""
    if X.ndim == 5:
        return X[:, :, :, 0, 0]
    if X.ndim == 4:
        return X[:, :, :, 0]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unexpected X shape: {X.shape}")


def sum_file_sizes_mb(paths):
    total = 0
    for p in paths:
        if p.exists() and p.is_file():
            total += p.stat().st_size
    return float(total) / (1024 * 1024)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def serialize_scaler(scaler):
    """Store minimal scaler params needed for re-use."""
    if isinstance(scaler, StandardScaler):
        return {
            "type": "standard",
            "mean": scaler.mean_.astype(np.float32),
            "scale": scaler.scale_.astype(np.float32),
        }
    if isinstance(scaler, RobustScaler):
        return {
            "type": "robust",
            "center": scaler.center_.astype(np.float32),
            "scale": scaler.scale_.astype(np.float32),
        }
    return {"type": scaler.__class__.__name__}


def _get_datetime_index_or_column(df: pd.DataFrame, cfg: dict) -> pd.DatetimeIndex:
    """
    Robustly obtain datetime series:
    - prefer DatetimeIndex
    - else use cfg.columns.timestamp
    - else use 'timestamp' column
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index

    ts_col_cfg = cfg.get("columns", {}).get("timestamp", None)
    if ts_col_cfg and ts_col_cfg in df.columns:
        return pd.to_datetime(df[ts_col_cfg], utc=True, errors="coerce")

    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    raise KeyError(
        "Could not find a datetime index/column. "
        "Expected a DatetimeIndex or a timestamp column (cfg.columns.timestamp or 'timestamp')."
    )


def ensure_time_and_flag_features(df: pd.DataFrame, cfg: dict, required_cols: list) -> pd.DataFrame:
    """
    Fix #1: Ensure cyclical calendar features exist if referenced by YAML.
    If a referenced is_* flag is missing, fill with 0 (safe default).

    Creates (if missing and referenced):
      hour_sin, hour_cos (period 24)
      dow_sin,  dow_cos  (period 7)
      month_sin,month_cos(period 12)
      is_weekend (derived)
    """
    df = df.copy()
    dt = _get_datetime_index_or_column(df, cfg)

    hour = pd.Series(dt.hour, index=df.index)
    dow = pd.Series(dt.dayofweek, index=df.index)   # Mon=0..Sun=6
    month = pd.Series(dt.month, index=df.index)     # 1..12

    def make_cyc(name_sin, name_cos, values, period):
        ang = 2.0 * np.pi * (values.astype(np.float32) / float(period))
        if name_sin in required_cols and name_sin not in df.columns:
            df[name_sin] = np.sin(ang).astype(np.float32)
        if name_cos in required_cols and name_cos not in df.columns:
            df[name_cos] = np.cos(ang).astype(np.float32)

    make_cyc("hour_sin", "hour_cos", hour, 24)
    make_cyc("dow_sin", "dow_cos", dow, 7)
    month0 = (month - 1).astype(np.float32)  # shift 1..12 -> 0..11
    make_cyc("month_sin", "month_cos", month0, 12)

    if "is_weekend" in required_cols and "is_weekend" not in df.columns:
        df["is_weekend"] = (dow >= 5).astype(np.int8)

    for c in required_cols:
        if c.startswith("is_") and c not in df.columns:
            df[c] = np.int8(0)

    return df


def check_columns_exist(df: pd.DataFrame, cols: list, df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{df_name}] Missing required columns: {missing}\n"
            f"Available columns (sample): {list(df.columns)[:30]}"
        )


# -----------------------------
# Dataset + ELM
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]).float(), torch.from_numpy(self.Y[i]).float()


class ELM(nn.Module):
    """
    ELM with adaptive ridge solve:
    - start with ridge from cfg
    - if solve fails (singular / ill-conditioned), increase ridge x10 until solvable
    - final fallback: lstsq (very robust)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden=1024,
        ridge=1e-3,
        seed=42,
        device="cpu",
        weight_scale=0.5,
        bias_scale=0.5
    ):
        super().__init__()
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(in_dim, hidden, generator=g) * float(weight_scale)
        b = torch.randn(hidden, generator=g) * float(bias_scale)

        self.W = nn.Parameter(W.to(self.device), requires_grad=False)
        self.b = nn.Parameter(b.to(self.device), requires_grad=False)

        self.ridge = float(ridge)
        self.beta = None  # [hidden, out_dim]

    def fit(self, X, Y):
        H = torch.tanh(X @ self.W + self.b)  # [N, hidden]
        Ht = H.T
        A = Ht @ H
        I = torch.eye(A.shape[0], device=self.device)
        B = Ht @ Y

        ridge = max(self.ridge, 1e-8)

        for _ in range(8):
            try:
                self.beta = torch.linalg.lstsq(A + ridge * I, B).solution
                self.ridge = ridge
                return
            except RuntimeError:
                ridge *= 10.0

        A2 = A + ridge * I
        self.beta = torch.linalg.lstsq(A2, B).solution
        self.ridge = ridge

    def predict(self, X):
        H = torch.tanh(X @ self.W + self.b)
        return H @ self.beta


def extract_latents(enc, X_np, device, batch_size=256, res_mon=None):
    enc.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device)
            z = enc(xb).detach().cpu().numpy()
            outs.append(z)
            if res_mon:
                res_mon.update()
    return np.concatenate(outs, axis=0)


# -----------------------------
# CSV header parity (BASE + EXTRA)
# -----------------------------
BASE_HEADER = [
    "task", "lookback", "horizon",
    "MAE", "RMSE", "sMAPE",
    "BASE_MAE", "BASE_RMSE", "BASE_sMAPE",
    "Params",
    "Train_Wall_Sec", "Train_CPU_Sec", "Avg_CPU_Usage_Pct",
    "Peak_RAM_MB",
    "Infer_Wall_Sec", "Infer_CPU_Sec", "Infer_Avg_CPU_Pct",
    "Latency_ms_per_sample",
    "Size_MB"
]

EXTRA_HEADER = [
    "Train_Params",
    "Deploy_Params",
    "Train_Size_MB",
    "Deploy_Size_MB"
]

HEADER_V2 = BASE_HEADER + EXTRA_HEADER


def append_row_schema_safe(summary_csv: Path, row_dict: dict):
    """Auto-upgrade schema to HEADER_V2 and append."""
    df_row = pd.DataFrame([[row_dict.get(h, np.nan) for h in HEADER_V2]], columns=HEADER_V2)

    if summary_csv.exists():
        df_existing = pd.read_csv(summary_csv)
        for col in HEADER_V2:
            if col not in df_existing.columns:
                df_existing[col] = np.nan
        df_existing = df_existing[HEADER_V2]  # reorder + drop extras
        df_final = pd.concat([df_existing, df_row], ignore_index=True)
        df_final.to_csv(summary_csv, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)


def run(cfg_path: str, lookback=None, horizon=None, out_dir=None, summary_csv=None):
    cfg = load_config(cfg_path)

    # override LB/H from CLI
    if lookback is not None:
        cfg["features"]["context_hours"] = int(lookback)
    if horizon is not None:
        cfg["features"]["horizon_hours"] = int(horizon)

    ctx = int(cfg["features"]["context_hours"])
    hz = int(cfg["features"]["horizon_hours"])

    base_out = Path(cfg["paths"]["outputs_dir"])
    out_path = Path(out_dir) if out_dir else (base_out / f"LB{ctx}_H{hz}")
    out_path.mkdir(parents=True, exist_ok=True)

    if summary_csv is None:
        summary_csv = str(base_out / "summary_dcenn_energy_raw.csv")
    summary_csv = Path(summary_csv)

    seed = int(cfg.get("random_seed", 42))
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_mon = ResourceMonitor()

    # Fix #2 split: dynamic vs static
    dyn_inputs = list(cfg["features"]["input_features"])                 # encoder inputs (dynamic)
    static_inputs = list(cfg["features"].get("static_features", []))     # bypass encoder
    targets = list(cfg["features"]["target_features"])

    # For scaling and integrity checks
    x_cols = list(dict.fromkeys(dyn_inputs + static_inputs))

    latent_dim = int(cfg["training"]["encoder"]["latent_channels"])
    lr = float(cfg["training"]["encoder"].get("lr", 1e-3))
    batch_size = int(cfg["training"]["encoder"].get("batch_size", 128))
    epochs = int(cfg["training"]["encoder"].get("epochs", 10))
    ram_sample_every = int(cfg["training"]["encoder"].get("ram_sample_every", 10))

    elm_hidden = int(cfg["training"]["elm"].get("hidden", 1024))
    elm_ridge = float(cfg["training"]["elm"].get("ridge_lambda", 1e-3))
    elm_wscale = float(cfg["training"]["elm"].get("weight_scale", 0.5))

    scaler_name = cfg.get("scaling", {}).get("type", "standard").lower()

    print(f"\n[dCeNN ENERGY RAW] LB={ctx} H={hz} out={out_path}")
    print(f"device={device} seed={seed} latent={latent_dim} lr={lr} bs={batch_size} epochs={epochs}")
    print(f"Dynamic inputs (encoder): {len(dyn_inputs)} | Static bypass (ELM): {len(static_inputs)}")
    if static_inputs:
        print(f"Static features: {static_inputs}")
    print(f"ELM hidden={elm_hidden} ridge={elm_ridge} wscale={elm_wscale} | scaler={scaler_name}")
    print(f"RAM sampling every {ram_sample_every} train batches")

    # -----------------------------
    # 1) Load data
    # -----------------------------
    train_df, val_df, test_df = build_master(cfg)
    res_mon.update()

    # -----------------------------
    # 1b) Fix #1: Ensure time features/flags exist if referenced
    # -----------------------------
    required_for_x = list(dict.fromkeys(x_cols))
    train_df = ensure_time_and_flag_features(train_df, cfg, required_for_x)
    val_df = ensure_time_and_flag_features(val_df, cfg, required_for_x)
    test_df = ensure_time_and_flag_features(test_df, cfg, required_for_x)

    check_columns_exist(train_df, required_for_x + targets, "train_df")
    check_columns_exist(val_df, required_for_x + targets, "val_df")
    check_columns_exist(test_df, required_for_x + targets, "test_df")

    # -----------------------------
    # 2) Scaling (X + Y)
    # -----------------------------
    if scaler_name == "robust":
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()
    else:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

    x_scaler.fit(train_df[required_for_x])
    y_scaler.fit(train_df[targets])

    def scale_x(df):
        d = df.copy()
        d[required_for_x] = x_scaler.transform(df[required_for_x])
        return d

    def scale_y(df):
        d = df.copy()
        d[targets] = y_scaler.transform(df[targets])
        return d

    train_x, val_x, test_x = scale_x(train_df), scale_x(val_df), scale_x(test_df)
    train_y, val_y = scale_y(train_df), scale_y(val_df)

    # -----------------------------
    # 3) Windowing
    #    - encoder windows use ONLY dyn_inputs
    #    - static windows use ONLY static_inputs (aligned), then take last step
    # -----------------------------
    Xtr_dyn, _, _ = make_windows(train_x, dyn_inputs, targets, ctx, hz)
    _, Ytr, _ = make_windows(train_y, dyn_inputs, targets, ctx, hz)

    Xva_dyn, _, _ = make_windows(val_x, dyn_inputs, targets, ctx, hz)
    _, Yva, _ = make_windows(val_y, dyn_inputs, targets, ctx, hz)

    Xte_dyn, _, te_idx = make_windows(test_x, dyn_inputs, targets, ctx, hz)
    _, Yte_true, _ = make_windows(test_df, dyn_inputs, targets, ctx, hz)  # RAW truth

    # static bypass
    if len(static_inputs) > 0:
        Xtr_stat, _, _ = make_windows(train_x, static_inputs, targets, ctx, hz)
        Xva_stat, _, _ = make_windows(val_x, static_inputs, targets, ctx, hz)
        Xte_stat, _, _ = make_windows(test_x, static_inputs, targets, ctx, hz)

        S_tr = squeeze_X(Xtr_stat)[:, -1, :].astype(np.float32)
        S_va = squeeze_X(Xva_stat)[:, -1, :].astype(np.float32)
        S_te = squeeze_X(Xte_stat)[:, -1, :].astype(np.float32)

        if S_tr.shape[0] != Xtr_dyn.shape[0] or S_te.shape[0] != Xte_dyn.shape[0]:
            raise RuntimeError("Static/dynamic window counts do not match. Check make_windows() alignment.")
    else:
        S_tr = np.zeros((Xtr_dyn.shape[0], 0), dtype=np.float32)
        S_va = np.zeros((Xva_dyn.shape[0], 0), dtype=np.float32)
        S_te = np.zeros((Xte_dyn.shape[0], 0), dtype=np.float32)

    res_mon.update()

    # -----------------------------
    # 4) Baseline persistence (RAW)
    # -----------------------------
    Xte_raw, _, _ = make_windows(test_df, dyn_inputs, targets, ctx, hz)
    Xte_raw_3d = squeeze_X(Xte_raw)

    tgt_idx = [dyn_inputs.index(t) for t in targets if t in dyn_inputs]
    if len(tgt_idx) != len(targets):
        raise ValueError(
            "Some targets are not present in dynamic input_features; cannot compute persistence BASE fairly. "
            "Ensure targets are included in cfg.features.input_features."
        )

    last_vals = Xte_raw_3d[:, -1, :][:, tgt_idx]              # [N,C]
    base_pred = np.repeat(last_vals[:, None, :], hz, axis=1)  # [N,H,C]
    BASE_MAE, BASE_RMSE, BASE_sMAPE = calc_metrics(Yte_true, base_pred)

    # Save meta for ASP
    meta_cols = [c for c in ["cap_wind_mw", "cap_solar_mw", "cf_wind", "cf_solar"] if c in test_df.columns]
    meta_df = pd.DataFrame(index=te_idx)
    if meta_cols:
        tmp = test_df.reindex(te_idx)
        for c in meta_cols:
            meta_df[c] = tmp[c].values
    meta_df["timestamp"] = pd.to_datetime(te_idx)
    meta_df.to_parquet(out_path / "meta_energy.parquet")

    # -----------------------------
    # 5) Train encoder+linear head (timed)
    # -----------------------------
    train_t0 = time.time()
    cpu0 = get_process_metrics()[1]

    enc = TinyDCENN(len(dyn_inputs), latent_dim).to(device)
    head = nn.Linear(latent_dim, hz * len(targets)).to(device)

    optim = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.L1Loss()

    train_dl = DataLoader(WindowDataset(Xtr_dyn, Ytr), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(WindowDataset(Xva_dyn, Yva), batch_size=batch_size, shuffle=False)

    for ep in range(epochs):
        enc.train()
        head.train()
        tr_sum = 0.0

        for step, (xb, yb) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs} [Train]")):
            xb, yb = xb.to(device), yb.to(device)
            z = enc(xb)
            pred = head(z).reshape(yb.shape)
            loss = loss_fn(pred, yb)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            optim.step()
            tr_sum += float(loss.item())

            if (step % max(1, ram_sample_every)) == 0:
                res_mon.update()

        enc.eval()
        head.eval()
        va_sum = 0.0
        with torch.no_grad():
            for step, (xb, yb) in enumerate(val_dl):
                xb, yb = xb.to(device), yb.to(device)
                z = enc(xb)
                pred = head(z).reshape(yb.shape)
                va_sum += float(loss_fn(pred, yb).item())
                if (step % max(1, ram_sample_every)) == 0:
                    res_mon.update()

        tr_loss = tr_sum / max(1, len(train_dl))
        va_loss = va_sum / max(1, len(val_dl))
        print(f"  > Epoch {ep+1}: train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        res_mon.update()

    # -----------------------------
    # 6) Fit ELM heads (within TRAIN timing)
    #    Fix #2: ELM sees [z_scaled || static_last_step]
    # -----------------------------
    Ztr_lat = extract_latents(enc, Xtr_dyn, device, batch_size=256, res_mon=res_mon)

    z_scaler = StandardScaler()
    Ztr_lat = z_scaler.fit_transform(Ztr_lat).astype(np.float32)

    Xelm_tr_np = np.concatenate([Ztr_lat, S_tr], axis=1).astype(np.float32)
    Xelm_tr_t = torch.from_numpy(Xelm_tr_np).float().to(device)

    elms = []
    elm_betas = []
    elm_Ws = []
    elm_bs = []
    elm_ridges_used = []

    for i in range(len(targets)):
        elm_seed = seed + 1000 + i
        elm = ELM(
            in_dim=Xelm_tr_np.shape[1],
            out_dim=hz,
            hidden=elm_hidden,
            ridge=elm_ridge,
            seed=elm_seed,
            device=str(device),
            weight_scale=elm_wscale,
            bias_scale=elm_wscale
        ).to(device)

        y_i = torch.from_numpy(Ytr[:, :, i]).float().to(device)
        elm.fit(Xelm_tr_t, y_i)

        print(f"  ELM[{targets[i]}] used ridge={elm.ridge:g}")
        elms.append(elm)
        elm_ridges_used.append(float(elm.ridge))

        elm_betas.append(elm.beta.detach().cpu().numpy().astype(np.float32))
        elm_Ws.append(elm.W.detach().cpu().numpy().astype(np.float32))
        elm_bs.append(elm.b.detach().cpu().numpy().astype(np.float32))

        res_mon.update()

    train_wall = time.time() - train_t0
    train_cpu = get_process_metrics()[1] - cpu0
    avg_cpu_pct = 100.0 * (train_cpu / train_wall) if train_wall > 0 else 0.0

    # -----------------------------
    # 7) Inference timing (RAW predictions)
    # -----------------------------
    inf_t0 = time.time()
    icpu0 = get_process_metrics()[1]

    Zte_lat = extract_latents(enc, Xte_dyn, device, batch_size=256, res_mon=res_mon)
    Zte_lat = z_scaler.transform(Zte_lat).astype(np.float32)

    Xelm_te_np = np.concatenate([Zte_lat, S_te], axis=1).astype(np.float32)
    Xelm_te_t = torch.from_numpy(Xelm_te_np).float().to(device)

    preds_scaled = np.zeros((len(Xte_dyn), hz, len(targets)), dtype=np.float32)
    for i, elm in enumerate(elms):
        with torch.no_grad():
            p = elm.predict(Xelm_te_t).detach().cpu().numpy().astype(np.float32)
        preds_scaled[:, :, i] = p
        res_mon.update()

    N = preds_scaled.shape[0]
    flat = preds_scaled.reshape(-1, len(targets))
    preds_raw = y_scaler.inverse_transform(flat).reshape(N, hz, len(targets))

    infer_wall = time.time() - inf_t0
    infer_cpu = get_process_metrics()[1] - icpu0
    infer_avg_cpu_pct = 100.0 * (infer_cpu / infer_wall) if infer_wall > 0 else 0.0
    latency_ms = (infer_wall * 1000.0) / max(1, N)

    # -----------------------------
    # 8) Metrics
    # -----------------------------
    m = min(len(Yte_true), len(preds_raw), len(te_idx))
    Yte_true = Yte_true[:m]
    preds_raw = preds_raw[:m]
    te_idx = te_idx[:m]

    MAE, RMSE, sMAPE = calc_metrics(Yte_true, preds_raw)

    try:
        neg_pct = 100.0 * float((preds_raw < 0).mean())
        print(f"[Sanity] preds_raw min={preds_raw.min():.3f} max={preds_raw.max():.3f} | negatives={neg_pct:.2f}%")
    except Exception:
        pass

    # -----------------------------
    # 9) Save predictions + truth in "+h" format
    # -----------------------------
    cols = []
    for h in range(hz):
        for name in targets:
            cols.append(f"{name}+h{h+1}")

    df_pred = pd.DataFrame(np.hstack([preds_raw[:, h, :] for h in range(hz)]), index=te_idx, columns=cols)
    df_true = pd.DataFrame(np.hstack([Yte_true[:, h, :] for h in range(hz)]), index=te_idx, columns=cols)

    df_pred.to_parquet(out_path / "raw_energy.parquet")
    df_true.to_parquet(out_path / "truth_energy.parquet")

    # -----------------------------
    # 10) Params + Size_MB artifacts (TRAIN vs DEPLOY)
    # -----------------------------
    params_enc = count_trainable_params(enc)
    params_head = count_trainable_params(head)
    Train_Params = int(params_enc + params_head)

    elm_in_dim = int(Xelm_tr_np.shape[1])
    C = int(len(targets))
    Deploy_Params_ELM = int(C * (elm_in_dim * elm_hidden + elm_hidden + elm_hidden * hz))
    Deploy_Params = int(params_enc + Deploy_Params_ELM)

    print("\n[Param Accounting]")
    print(f"  Train_Params (encoder+head)             = {Train_Params:,}")
    print(f"  Deploy_Params (encoder + ELM W/b/beta)  = {Deploy_Params:,}")
    print(f"    - encoder params: {params_enc:,}")
    print(f"    - ELM params (W+b+beta): {Deploy_Params_ELM:,}")
    print(f"    - ELM input dim (latent+static): {elm_in_dim} = {latent_dim} + {len(static_inputs)}")

    train_art_path = out_path / "dcenn_energy_train.pt"
    torch.save(
        {
            "encoder_state": enc.state_dict(),
            "head_state": head.state_dict(),
            "meta": {
                "lookback": ctx,
                "horizon": hz,
                "dynamic_inputs": dyn_inputs,
                "static_inputs": static_inputs,
                "targets": targets,
                "latent_dim": latent_dim,
                "seed": seed,
                "scaler": scaler_name,
                "elm_hidden": elm_hidden,
                "elm_ridge_start": float(elm_ridge),
                "elm_wscale": float(elm_wscale),
                "Train_Params": Train_Params,
                "Deploy_Params": Deploy_Params,
            },
            "scalers": {
                "x_scaler": serialize_scaler(x_scaler),
                "y_scaler": serialize_scaler(y_scaler),
            },
        },
        train_art_path
    )

    deploy_art_path = out_path / "dcenn_energy_deploy.pt"
    torch.save(
        {
            "encoder_state": enc.state_dict(),
            "meta": {
                "lookback": ctx,
                "horizon": hz,
                "dynamic_inputs": dyn_inputs,
                "static_inputs": static_inputs,
                "targets": targets,
                "latent_dim": latent_dim,
                "seed": seed,
                "scaler": scaler_name,
                "elm_hidden": elm_hidden,
                "elm_wscale": float(elm_wscale),
                "Train_Params": Train_Params,
                "Deploy_Params": Deploy_Params,
            },
            "latent_scaler": {
                "type": "standard",
                "mean": z_scaler.mean_.astype(np.float32),
                "scale": z_scaler.scale_.astype(np.float32),
            },
            "scalers": {
                "x_scaler": serialize_scaler(x_scaler),
                "y_scaler": serialize_scaler(y_scaler),
            },
        },
        deploy_art_path
    )

    elm_deploy_path = out_path / "elm_energy_deploy.npz"
    np.savez_compressed(
        elm_deploy_path,
        W=np.stack(elm_Ws, axis=0),           # [C, in_dim, hidden]
        b=np.stack(elm_bs, axis=0),           # [C, hidden]
        beta=np.stack(elm_betas, axis=0),     # [C, hidden, hz]
        targets=np.array(targets),
        elm_in_dim=np.int32(elm_in_dim),
        elm_hidden=np.int32(elm_hidden),
        horizon=np.int32(hz),
        ridge_start=np.float32(elm_ridge),
        ridge_used=np.array(elm_ridges_used, dtype=np.float32),
        wscale=np.float32(elm_wscale),
        seeds=np.array([seed + 1000 + i for i in range(C)], dtype=np.int32),
        dynamic_inputs=np.array(dyn_inputs),
        static_inputs=np.array(static_inputs),
    )

    Train_Size_MB = sum_file_sizes_mb([train_art_path])
    Deploy_Size_MB = sum_file_sizes_mb([deploy_art_path, elm_deploy_path])

    # Backward compatible columns (match other scripts)
    Params = int(Deploy_Params)
    Size_MB = float(Deploy_Size_MB)

    Peak_RAM_MB = float(res_mon.peak_ram_mb)

    print("\n[Artifact Sizes]")
    print(f"  Train_Size_MB (encoder+head)                = {Train_Size_MB:.3f} MB")
    print(f"  Deploy_Size_MB (encoder+scalers + ELM npz)  = {Deploy_Size_MB:.3f} MB")

    (out_path / "base_metrics.json").write_text(json.dumps({
        "BASE_MAE": BASE_MAE,
        "BASE_RMSE": BASE_RMSE,
        "BASE_sMAPE": BASE_sMAPE
    }, indent=2))

    (out_path / "params_accounting.json").write_text(json.dumps({
        "Train_Params_encoder_plus_head": Train_Params,
        "Deploy_Params_encoder_plus_elm_W_b_beta": Deploy_Params,
        "encoder_trainable_params": params_enc,
        "elm_params_total": Deploy_Params_ELM,
        "elm_in_dim": elm_in_dim,
        "elm_hidden": elm_hidden,
        "horizon": hz,
        "num_targets": len(targets),
        "targets": targets,
        "dynamic_inputs": dyn_inputs,
        "static_inputs": static_inputs,
    }, indent=2))

    # -----------------------------
    # 11) Append summary row (schema-safe, HEADER_V2)
    # -----------------------------
    row = {
        "task": "ENERGY",
        "lookback": ctx,
        "horizon": hz,
        "MAE": MAE,
        "RMSE": RMSE,
        "sMAPE": sMAPE,
        "BASE_MAE": BASE_MAE,
        "BASE_RMSE": BASE_RMSE,
        "BASE_sMAPE": BASE_sMAPE,
        "Params": int(Params),
        "Train_Wall_Sec": float(train_wall),
        "Train_CPU_Sec": float(train_cpu),
        "Avg_CPU_Usage_Pct": float(avg_cpu_pct),
        "Peak_RAM_MB": float(Peak_RAM_MB),
        "Infer_Wall_Sec": float(infer_wall),
        "Infer_CPU_Sec": float(infer_cpu),
        "Infer_Avg_CPU_Pct": float(infer_avg_cpu_pct),
        "Latency_ms_per_sample": float(latency_ms),
        "Size_MB": float(Size_MB),

        # EXTRA (new parity cols)
        "Train_Params": int(Train_Params),
        "Deploy_Params": int(Deploy_Params),
        "Train_Size_MB": float(Train_Size_MB),
        "Deploy_Size_MB": float(Deploy_Size_MB),
    }

    append_row_schema_safe(summary_csv, row)

    print(f"\n[DONE] ENERGY LB={ctx} H={hz}")
    print(f"MAE={MAE:.4f} RMSE={RMSE:.4f} sMAPE={sMAPE:.2f}% | BASE_MAE={BASE_MAE:.4f}")
    print(f"Train_Params={Train_Params:,} | Deploy_Params={Deploy_Params:,}")
    print(f"Train_Size_MB={Train_Size_MB:.3f} | Deploy_Size_MB={Deploy_Size_MB:.3f}")
    print(f"Updated summary: {summary_csv}")
    print(f"Saved outputs: {out_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/energy_full.yaml")
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)
    args = ap.parse_args()

    run(
        args.config,
        lookback=args.lookback,
        horizon=args.horizon,
        out_dir=args.out_dir,
        summary_csv=args.summary_csv
    )
