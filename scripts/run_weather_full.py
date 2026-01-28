"""
Run full weather forecasting pipeline (dCeNN + ELM), save metrics like baselines.

UPDATES (matches the ENERGY fixes for thesis-grade consistency):
- Latent normalization (Z StandardScaler) for ELM conditioning
- Adaptive ridge in ELM.fit() to avoid singular-matrix crashes (robust)
- Peak RAM sampling inside training batches (rss_every_batches)
- TWO parameter counts:
    * Train_Params  = encoder + linear head (trained)
    * Deploy_Params = encoder + ELM (W,b,beta stored & used at inference)
  (Legacy "Params" column is kept as Deploy_Params for backward compatibility.)
- Save two artifacts:
    * dcenn_weather_train.pt  (encoder + head)
    * dcenn_weather_deploy.pt (encoder only)
  (Legacy "Size_MB" is kept as Deploy_Size_MB for backward compatibility.)
- Store ELM W,b,beta + z_scaler + x/y scaler stats into elm_betas.npz (real deployment payload)
- Summary CSV auto-upgrades to include new columns if missing

Examples:
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 24  --horizon 12
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 24  --horizon 24
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 24  --horizon 72

python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 72  --horizon 12
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 72  --horizon 24
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 72  --horizon 72

python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 168 --horizon 12
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 168 --horizon 24
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 168 --horizon 72
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
from sklearn.preprocessing import StandardScaler
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


def _scaler_loc_scale(scaler):
    """
    Return (loc, scale, kind) for StandardScaler/RobustScaler in a uniform way.
    loc is mean_ (Standard) or center_ (Robust).
    """
    kind = scaler.__class__.__name__
    if hasattr(scaler, "mean_"):
        loc = np.asarray(scaler.mean_, dtype=np.float32)
    elif hasattr(scaler, "center_"):
        loc = np.asarray(scaler.center_, dtype=np.float32)
    else:
        loc = None

    if hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float32)
    else:
        scale = None

    return loc, scale, kind


# -----------------------------
# Dataset + ELM
# -----------------------------
class WeatherDataset(Dataset):
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
                self.beta = torch.linalg.solve(A + ridge * I, B)
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
# CSV header identical style to baselines (kept) + extra thesis columns
# -----------------------------
BASE_HEADER = [
    "task","lookback","horizon",
    "MAE","RMSE","sMAPE",
    "BASE_MAE","BASE_RMSE","BASE_sMAPE",
    "Params",
    "Train_Wall_Sec","Train_CPU_Sec","Avg_CPU_Usage_Pct",
    "Peak_RAM_MB",
    "Infer_Wall_Sec","Infer_CPU_Sec","Infer_Avg_CPU_Pct",
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


def run(cfg_path: str, lookback=None, horizon=None, out_dir=None, summary_csv=None):
    cfg = load_config(cfg_path)

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
        summary_csv = str(base_out / "summary_dcenn_weather_raw.csv")
    summary_csv = Path(summary_csv)

    seed = int(cfg.get("random_seed", 42))
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_mon = ResourceMonitor()

    inputs = cfg["features"]["input_features"]
    targets = cfg["features"]["target_features"]

    latent_dim = int(cfg["training"]["encoder"]["latent_channels"])
    lr = float(cfg["training"]["encoder"].get("lr", 1e-3))
    batch_size = int(cfg["training"]["encoder"].get("batch_size", 128))
    epochs = int(cfg["training"]["encoder"].get("epochs", 10))

    rss_every = int(cfg["training"]["encoder"].get("rss_every_batches", 10))
    rss_every = max(1, rss_every)

    elm_hidden = int(cfg["training"]["elm"].get("hidden", 1024))
    elm_ridge = float(cfg["training"]["elm"].get("ridge_lambda", 1e-3))
    elm_wscale = float(cfg["training"]["elm"].get("weight_scale", 0.5))

    scaler_name = "standard"  # weather pipeline uses StandardScaler for fairness

    print(f"\n[dCeNN WEATHER RAW] LB={ctx} H={hz} out={out_path}")
    print(f"device={device} seed={seed} latent={latent_dim} lr={lr} bs={batch_size} epochs={epochs} rss_every={rss_every}")
    print(f"ELM hidden={elm_hidden} ridge_start={elm_ridge} wscale={elm_wscale} | latent_norm=ON adaptive_ridge=ON")

    # -----------------------------
    # 1) Load data
    # -----------------------------
    train_df, val_df, test_df = build_master(cfg)
    res_mon.update()

    # -----------------------------
    # 2) Scaling (X uses inputs scaler, Y uses targets scaler)
    # -----------------------------
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[inputs])
    y_scaler.fit(train_df[targets])

    def make_x_df(df):
        d = df.copy()
        d[inputs] = x_scaler.transform(df[inputs])
        return d

    def make_y_df(df):
        d = df.copy()
        d[targets] = y_scaler.transform(df[targets])
        return d

    train_x = make_x_df(train_df)
    val_x   = make_x_df(val_df)
    test_x  = make_x_df(test_df)

    train_y = make_y_df(train_df)
    val_y   = make_y_df(val_df)

    # -----------------------------
    # 3) Windowing
    # -----------------------------
    Xtr, _, _       = make_windows(train_x, inputs, targets, ctx, hz)
    _,  Ytr, _      = make_windows(train_y, inputs, targets, ctx, hz)

    Xva, _, _       = make_windows(val_x, inputs, targets, ctx, hz)
    _,  Yva, _      = make_windows(val_y, inputs, targets, ctx, hz)

    Xte, _, te_idx  = make_windows(test_x, inputs, targets, ctx, hz)
    _,  Yte_true, _ = make_windows(test_df, inputs, targets, ctx, hz)  # RAW truth for metrics

    res_mon.update()

    # -----------------------------
    # 4) Baseline persistence metrics (RAW)
    # -----------------------------
    Xte_raw, _, _ = make_windows(test_df, inputs, targets, ctx, hz)
    Xte_raw_3d = squeeze_X(Xte_raw)

    tgt_idx = [inputs.index(t) for t in targets if t in inputs]
    if len(tgt_idx) != len(targets):
        raise ValueError("Some targets are not present in inputs; cannot compute persistence BASE fairly.")

    last_vals = Xte_raw_3d[:, -1, :][:, tgt_idx]              # [N,C]
    base_pred = np.repeat(last_vals[:, None, :], hz, axis=1)  # [N,H,C]
    BASE_MAE, BASE_RMSE, BASE_sMAPE = calc_metrics(Yte_true, base_pred)

    # -----------------------------
    # 5) Train encoder + head (timed)
    # -----------------------------
    train_t0 = time.time()
    cpu0 = get_process_metrics()[1]

    enc = TinyDCENN(len(inputs), latent_dim).to(device)
    head = nn.Linear(latent_dim, hz * len(targets)).to(device)

    optim = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.L1Loss()

    train_dl = DataLoader(WeatherDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(WeatherDataset(Xva, Yva), batch_size=batch_size, shuffle=False)

    for ep in range(epochs):
        enc.train(); head.train()
        tr_sum = 0.0

        for b, (xb, yb) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs} [Train]")):
            xb, yb = xb.to(device), yb.to(device)
            z = enc(xb)
            pred = head(z).reshape(yb.shape)
            loss = loss_fn(pred, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_sum += float(loss.item())

            if (b % rss_every) == 0:
                res_mon.update()

        enc.eval(); head.eval()
        va_sum = 0.0
        with torch.no_grad():
            for b, (xb, yb) in enumerate(val_dl):
                xb, yb = xb.to(device), yb.to(device)
                z = enc(xb)
                pred = head(z).reshape(yb.shape)
                va_sum += float(loss_fn(pred, yb).item())
                if (b % rss_every) == 0:
                    res_mon.update()

        print(f"  > Epoch {ep+1}: train_loss={tr_sum/max(1,len(train_dl)):.4f}  val_loss={va_sum/max(1,len(val_dl)):.4f}")
        res_mon.update()

    # -----------------------------
    # 6) Fit ELM heads (within TRAIN timing)
    # -----------------------------
    Ztr = extract_latents(enc, Xtr, device, batch_size=256, res_mon=res_mon)

    z_scaler = StandardScaler()
    Ztr = z_scaler.fit_transform(Ztr).astype(np.float32)
    Ztr_t = torch.from_numpy(Ztr).float().to(device)

    elms = []
    elm_betas = []
    Ws = []
    bs = []
    ridges_used = []

    elm_in_dim = int(Ztr.shape[1])  # IMPORTANT: actual encoder output dim

    for i in range(len(targets)):
        elm_seed = seed + 1000 + i
        elm = ELM(
            in_dim=elm_in_dim,
            out_dim=hz,
            hidden=elm_hidden,
            ridge=elm_ridge,
            seed=elm_seed,
            device=str(device),
            weight_scale=elm_wscale,
            bias_scale=elm_wscale
        ).to(device)

        y_i = torch.from_numpy(Ytr[:, :, i]).float().to(device)
        elm.fit(Ztr_t, y_i)

        print(f"  ELM[{targets[i]}] used ridge={elm.ridge:g}")
        ridges_used.append(float(elm.ridge))

        elms.append(elm)
        elm_betas.append(elm.beta.detach().cpu().numpy().astype(np.float32))
        Ws.append(elm.W.detach().cpu().numpy().astype(np.float32))
        bs.append(elm.b.detach().cpu().numpy().astype(np.float32))
        res_mon.update()

    train_wall = time.time() - train_t0
    train_cpu = get_process_metrics()[1] - cpu0
    avg_cpu_pct = 100.0 * (train_cpu / train_wall) if train_wall > 0 else 0.0

    # -----------------------------
    # 7) Inference timing (latents + ELM predict + inverse transform)
    # -----------------------------
    inf_t0 = time.time()
    icpu0 = get_process_metrics()[1]

    Zte = extract_latents(enc, Xte, device, batch_size=256, res_mon=res_mon)
    Zte = z_scaler.transform(Zte).astype(np.float32)
    Zte_t = torch.from_numpy(Zte).float().to(device)

    preds_scaled = np.zeros((len(Xte), hz, len(targets)), dtype=np.float32)
    for i, elm in enumerate(elms):
        with torch.no_grad():
            p = elm.predict(Zte_t).detach().cpu().numpy().astype(np.float32)
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
    # 8) Metrics (RAW)
    # -----------------------------
    m = min(len(Yte_true), len(preds_raw), len(te_idx))
    Yte_true = Yte_true[:m]
    preds_raw = preds_raw[:m]
    te_idx = te_idx[:m]

    MAE, RMSE, sMAPE = calc_metrics(Yte_true, preds_raw)

    # -----------------------------
    # 9) Save predictions + truth
    # -----------------------------
    cols = []
    for h in range(hz):
        for name in targets:
            cols.append(f"{name}+h{h+1}")

    df_pred = pd.DataFrame(np.hstack([preds_raw[:, h, :] for h in range(hz)]), index=te_idx, columns=cols)
    df_true = pd.DataFrame(np.hstack([Yte_true[:, h, :] for h in range(hz)]), index=te_idx, columns=cols)

    df_pred.to_parquet(out_path / "raw_weather.parquet")
    df_true.to_parquet(out_path / "truth_weather.parquet")

    # -----------------------------
    # 10) Params + Size_MB artifacts (TRAIN vs DEPLOY)
    # -----------------------------
    params_enc = count_trainable_params(enc)
    params_head = count_trainable_params(head)

    C = int(len(targets))
    # ELM params per target = W (in*hidden) + b (hidden) + beta (hidden*hz)
    elm_params_per_target = int(elm_in_dim * elm_hidden + elm_hidden + elm_hidden * hz)
    Deploy_Params = int(params_enc + C * elm_params_per_target)
    Train_Params = int(params_enc + params_head)

    # TRAIN artifact: encoder + head
    train_model_path = out_path / "dcenn_weather_train.pt"
    torch.save(
        {
            "encoder_state": enc.state_dict(),
            "head_state": head.state_dict(),
            "meta": {
                "lookback": ctx,
                "horizon": hz,
                "inputs": inputs,
                "targets": targets,
                "latent_dim": latent_dim,
                "seed": seed,
                "elm_hidden": elm_hidden,
                "elm_ridge_start": float(elm_ridge),
                "elm_wscale": float(elm_wscale),
                "Train_Params": int(Train_Params),
                "Deploy_Params": int(Deploy_Params),
                "scaler": scaler_name
            }
        },
        train_model_path
    )

    # DEPLOY artifact: encoder only
    deploy_model_path = out_path / "dcenn_weather_deploy.pt"
    torch.save(
        {
            "encoder_state": enc.state_dict(),
            "meta": {
                "lookback": ctx,
                "horizon": hz,
                "inputs": inputs,
                "targets": targets,
                "latent_dim": latent_dim,
                "seed": seed,
                "elm_hidden": elm_hidden,
                "elm_ridge_start": float(elm_ridge),
                "elm_wscale": float(elm_wscale),
                "Train_Params": int(Train_Params),
                "Deploy_Params": int(Deploy_Params),
                "scaler": scaler_name
            }
        },
        deploy_model_path
    )

    # Deployment payload: ELM W,b,beta + scaler stats (keeps filename for backward compatibility)
    betas_path = out_path / "elm_betas.npz"
    x_loc, x_scale, x_kind = _scaler_loc_scale(x_scaler)
    y_loc, y_scale, y_kind = _scaler_loc_scale(y_scaler)

    np.savez_compressed(
        betas_path,
        betas=np.stack(elm_betas, axis=0),              # [C, hidden, hz]
        Ws=np.stack(Ws, axis=0),                        # [C, in_dim, hidden]
        bs=np.stack(bs, axis=0),                        # [C, hidden]
        elm_in_dim=np.int32(elm_in_dim),
        elm_hidden=np.int32(elm_hidden),
        horizon=np.int32(hz),
        elm_ridge_start=np.float32(elm_ridge),
        elm_ridges_used=np.asarray(ridges_used, dtype=np.float32),
        elm_wscale=np.float32(elm_wscale),
        seeds=np.array([seed + 1000 + i for i in range(C)], dtype=np.int32),
        z_mean=np.asarray(z_scaler.mean_, dtype=np.float32),
        z_scale=np.asarray(z_scaler.scale_, dtype=np.float32),
        x_loc=(x_loc if x_loc is not None else np.array([], dtype=np.float32)),
        x_scale=(x_scale if x_scale is not None else np.array([], dtype=np.float32)),
        x_kind=np.array([x_kind], dtype=np.str_),
        y_loc=(y_loc if y_loc is not None else np.array([], dtype=np.float32)),
        y_scale=(y_scale if y_scale is not None else np.array([], dtype=np.float32)),
        y_kind=np.array([y_kind], dtype=np.str_),
        scaler_name=np.array([scaler_name], dtype=np.str_),
        targets=np.array(targets, dtype=np.str_)
    )

    Train_Size_MB = sum_file_sizes_mb([train_model_path])
    Deploy_Size_MB = sum_file_sizes_mb([deploy_model_path, betas_path])

    # Backward compatible columns
    Params = int(Deploy_Params)           # legacy "Params" => DEPLOY params
    Size_MB = float(Deploy_Size_MB)       # legacy "Size_MB" => DEPLOY size

    Peak_RAM_MB = float(res_mon.peak_ram_mb)

    # base metrics for ASP script
    base_json = out_path / "base_metrics.json"
    base_json.write_text(json.dumps({
        "BASE_MAE": BASE_MAE,
        "BASE_RMSE": BASE_RMSE,
        "BASE_sMAPE": BASE_sMAPE
    }, indent=2))

    # nice thesis sidecar
    (out_path / "params_accounting.json").write_text(json.dumps({
        "Train_Params_encoder_plus_head": int(Train_Params),
        "Deploy_Params_encoder_plus_elm_W_b_beta": int(Deploy_Params),
        "encoder_trainable_params": int(params_enc),
        "head_trainable_params": int(params_head),
        "elm_in_dim": int(elm_in_dim),
        "elm_hidden": int(elm_hidden),
        "horizon": int(hz),
        "num_targets": int(len(targets)),
        "targets": targets,
    }, indent=2))

    # -----------------------------
    # 11) Append summary row (safe upgrade if CSV already exists)
    # -----------------------------
    row = {
        "task": "WEATHER",
        "lookback": ctx,
        "horizon": hz,
        "MAE": MAE,
        "RMSE": RMSE,
        "sMAPE": sMAPE,
        "BASE_MAE": BASE_MAE,
        "BASE_RMSE": BASE_RMSE,
        "BASE_sMAPE": BASE_sMAPE,
        "Params": Params,
        "Train_Wall_Sec": float(train_wall),
        "Train_CPU_Sec": float(train_cpu),
        "Avg_CPU_Usage_Pct": float(avg_cpu_pct),
        "Peak_RAM_MB": Peak_RAM_MB,
        "Infer_Wall_Sec": float(infer_wall),
        "Infer_CPU_Sec": float(infer_cpu),
        "Infer_Avg_CPU_Pct": float(infer_avg_cpu_pct),
        "Latency_ms_per_sample": float(latency_ms),
        "Size_MB": float(Size_MB),
        "Train_Params": int(Train_Params),
        "Deploy_Params": int(Deploy_Params),
        "Train_Size_MB": float(Train_Size_MB),
        "Deploy_Size_MB": float(Deploy_Size_MB),
    }

    df_row = pd.DataFrame([[row.get(h, np.nan) for h in HEADER_V2]], columns=HEADER_V2)

    if summary_csv.exists():
        df_existing = pd.read_csv(summary_csv)
        for col in HEADER_V2:
            if col not in df_existing.columns:
                df_existing[col] = np.nan
        df_existing = df_existing[HEADER_V2]
        df_final = pd.concat([df_existing, df_row], ignore_index=True)
        df_final.to_csv(summary_csv, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)

    print(f"\n[DONE] WEATHER LB={ctx} H={hz}")
    print(f"MAE={MAE:.4f} RMSE={RMSE:.4f} sMAPE={sMAPE:.2f}% | BASE_MAE={BASE_MAE:.4f}")
    print(f"Train_Params={Train_Params:,} | Deploy_Params={Deploy_Params:,}")
    print(f"Train_Size_MB={Train_Size_MB:.3f} | Deploy_Size_MB={Deploy_Size_MB:.3f}")
    print(f"Updated summary: {summary_csv}")
    print(f"Saved outputs: {out_path}")
    print(f"  - {train_model_path.name} (train)")
    print(f"  - {deploy_model_path.name} (deploy)")
    print(f"  - {betas_path.name} (deploy ELM+scalers)")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/weather_full.yaml")
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
