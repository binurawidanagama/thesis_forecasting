"""
Run full weather forecasting pipeline.
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
    """
    make_windows often returns X as [N,T,F,1,1] or [N,T,F,1] or [N,T,F]
    This returns [N,T,F]
    """
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
    def __init__(self, in_dim, out_dim, hidden=1024, ridge=1e-2, seed=42, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(in_dim, hidden, generator=g) * 0.5
        b = torch.randn(hidden, generator=g) * 0.5
        self.W = nn.Parameter(W.to(self.device), requires_grad=False)
        self.b = nn.Parameter(b.to(self.device), requires_grad=False)
        self.ridge = float(ridge)
        self.beta = None  # [hidden, out_dim]

    def fit(self, X, Y):
        # X: [N, D], Y: [N, H]
        Hm = torch.tanh(X @ self.W + self.b)
        I = torch.eye(Hm.shape[1], device=self.device)
        self.beta = torch.linalg.solve(Hm.T @ Hm + self.ridge * I, Hm.T @ Y)

    def predict(self, X):
        Hm = torch.tanh(X @ self.W + self.b)
        return Hm @ self.beta


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


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# -----------------------------
# Main
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

    # Where to append summary rows (same format as your baseline CSV)
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

    elm_hidden = int(cfg["training"]["elm"].get("hidden", 1024))
    elm_ridge = float(cfg["training"]["elm"].get("ridge_lambda", 1e-3))

    print(f"\n[dCeNN WEATHER RAW] LB={ctx} H={hz} out={out_path}")
    print(f"device={device} seed={seed} latent={latent_dim} elm_hidden={elm_hidden} ridge={elm_ridge}")

    # -----------------------------
    # 1) Load data
    # -----------------------------
    train_df, val_df, test_df = build_master(cfg)
    res_mon.update()

    # -----------------------------
    # 2) Scaling (IMPORTANT FIX)
    # X uses x_scaler on inputs
    # Y uses y_scaler on targets
    # (We call make_windows twice so overlap targets-in-inputs doesn't break scaling.)
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
    Xtr, _, _   = make_windows(train_x, inputs, targets, ctx, hz)
    _,  Ytr, _  = make_windows(train_y, inputs, targets, ctx, hz)

    Xva, _, _   = make_windows(val_x, inputs, targets, ctx, hz)
    _,  Yva, _  = make_windows(val_y, inputs, targets, ctx, hz)

    Xte, _, te_idx = make_windows(test_x, inputs, targets, ctx, hz)
    _,  Yte_true, _ = make_windows(test_df, inputs, targets, ctx, hz)  # raw truth for metrics

    res_mon.update()

    # -----------------------------
    # 4) Baseline (Persistence) metrics in RAW space
    # BASE pred = last observed value in context, repeated over horizon
    # -----------------------------
    Xte_raw, _, _ = make_windows(test_df, inputs, targets, ctx, hz)
    Xte_raw_3d = squeeze_X(Xte_raw)  # [N,T,F]
    tgt_idx = [inputs.index(t) for t in targets if t in inputs]
    if len(tgt_idx) != len(targets):
        raise ValueError("Some targets are not present in inputs, cannot compute persistence BASE fairly.")

    last_vals = Xte_raw_3d[:, -1, :][:, tgt_idx]  # [N,C]
    base_pred = np.repeat(last_vals[:, None, :], hz, axis=1)  # [N,H,C]
    BASE_MAE, BASE_RMSE, BASE_sMAPE = calc_metrics(Yte_true, base_pred)

    # -----------------------------
    # 5) Train encoder + head  (timed as Train_*)
    # -----------------------------
    train_t0 = time.time()
    cpu0 = get_process_metrics()[1]

    enc = TinyDCENN(len(inputs), latent_dim).to(device)
    head = nn.Linear(latent_dim, hz * len(targets)).to(device)

    optim = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.L1Loss()

    train_ds = WeatherDataset(Xtr, Ytr)
    val_ds   = WeatherDataset(Xva, Yva)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for ep in range(epochs):
        enc.train(); head.train()
        tr_sum = 0.0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            z = enc(xb)
            pred = head(z).reshape(yb.shape)
            loss = loss_fn(pred, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_sum += float(loss.item())

        enc.eval(); head.eval()
        va_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                z = enc(xb)
                pred = head(z).reshape(yb.shape)
                va_sum += float(loss_fn(pred, yb).item())

        print(f"  > Epoch {ep+1}: train_loss={tr_sum/max(1,len(train_dl)):.4f}  val_loss={va_sum/max(1,len(val_dl)):.4f}")
        res_mon.update()

    # Latents for ELM fit
    Ztr = extract_latents(enc, Xtr, device, batch_size=256, res_mon=res_mon)

    # Fit ELMs (still part of "training" cost)
    Ztr_t = torch.from_numpy(Ztr).float().to(device)

    elms = []
    elm_betas = []
    for i in range(len(targets)):
        elm_seed = seed + 1000 + i
        elm = ELM(in_dim=Ztr.shape[1], out_dim=hz, hidden=elm_hidden, ridge=elm_ridge, seed=elm_seed, device=str(device)).to(device)
        y_i = torch.from_numpy(Ytr[:, :, i]).float().to(device)  # y-scaled
        elm.fit(Ztr_t, y_i)
        elms.append(elm)
        elm_betas.append(elm.beta.detach().cpu().numpy().astype(np.float32))
        res_mon.update()

    train_wall = time.time() - train_t0
    train_cpu = get_process_metrics()[1] - cpu0
    avg_cpu_pct = 100.0 * (train_cpu / train_wall) if train_wall > 0 else 0.0

    # -----------------------------
    # 6) Inference timing (Latents + ELM predict + inverse transform)
    # -----------------------------
    inf_t0 = time.time()
    icpu0 = get_process_metrics()[1]

    Zte = extract_latents(enc, Xte, device, batch_size=256, res_mon=res_mon)
    Zte_t = torch.from_numpy(Zte).float().to(device)

    preds_scaled = np.zeros((len(Xte), hz, len(targets)), dtype=np.float32)
    for i, elm in enumerate(elms):
        with torch.no_grad():
            p = elm.predict(Zte_t).detach().cpu().numpy().astype(np.float32)
        preds_scaled[:, :, i] = p
        res_mon.update()

    # Inverse transform to RAW targets
    N = preds_scaled.shape[0]
    flat = preds_scaled.reshape(-1, len(targets))
    preds_raw = y_scaler.inverse_transform(flat).reshape(N, hz, len(targets))

    infer_wall = time.time() - inf_t0
    infer_cpu = get_process_metrics()[1] - icpu0
    infer_avg_cpu_pct = 100.0 * (infer_cpu / infer_wall) if infer_wall > 0 else 0.0
    latency_ms = (infer_wall * 1000.0) / max(1, N)

    res_mon.update()

    # -----------------------------
    # 7) Metrics (RAW)
    # -----------------------------
    # Align just in case
    m = min(len(Yte_true), len(preds_raw), len(te_idx))
    Yte_true = Yte_true[:m]
    preds_raw = preds_raw[:m]
    te_idx = te_idx[:m]

    MAE, RMSE, sMAPE = calc_metrics(Yte_true, preds_raw)

    # -----------------------------
    # 8) Save predictions + truth (same column style as you already use)
    # -----------------------------
    cols = []
    for h in range(hz):
        for name in targets:
            cols.append(f"{name}+h{h+1}")

    df_pred = pd.DataFrame(
        np.hstack([preds_raw[:, h, :] for h in range(hz)]),
        index=te_idx,
        columns=cols
    )
    df_true = pd.DataFrame(
        np.hstack([Yte_true[:, h, :] for h in range(hz)]),
        index=te_idx,
        columns=cols
    )

    df_pred.to_parquet(out_path / "raw_weather.parquet")
    df_true.to_parquet(out_path / "truth_weather.parquet")

    # -----------------------------
    # 9) Params + Size_MB
    # -----------------------------
    # Trainable params: encoder + head
    # Learned ELM params: beta only (hidden*hz per target)
    params_enc = count_trainable_params(enc)
    params_head = count_trainable_params(head)
    params_beta = int(elm_hidden * hz * len(targets))
    Params = int(params_enc + params_head + params_beta)

    # Save artifacts for Size_MB
    model_path = out_path / "dcenn_weather.pt"
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
            }
        },
        model_path
    )

    betas_path = out_path / "elm_betas.npz"
    np.savez_compressed(
        betas_path,
        betas=np.stack(elm_betas, axis=0),  # [C, hidden, hz]
        elm_hidden=elm_hidden,
        elm_ridge=elm_ridge,
        seeds=np.array([seed + 1000 + i for i in range(len(targets))], dtype=np.int32)
    )

    Size_MB = sum_file_sizes_mb([model_path, betas_path])

    Peak_RAM_MB = float(res_mon.peak_ram_mb)

    # Save base metrics so ASP can reuse (no re-windowing needed there)
    base_json = out_path / "base_metrics.json"
    base_json.write_text(json.dumps({
        "BASE_MAE": BASE_MAE,
        "BASE_RMSE": BASE_RMSE,
        "BASE_sMAPE": BASE_sMAPE
    }, indent=2))

    # -----------------------------
    # 10) Append summary row EXACTLY like your baseline CSV header
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
    }

    df_row = pd.DataFrame([[row[h] for h in BASE_HEADER]], columns=BASE_HEADER)
    df_row.to_csv(summary_csv, mode="a", header=not summary_csv.exists(), index=False)

    print(f"\n[DONE] WEATHER LB={ctx} H={hz}")
    print(f"MAE={MAE:.4f} RMSE={RMSE:.4f} sMAPE={sMAPE:.2f}% | BASE_MAE={BASE_MAE:.4f}")
    print(f"Appended summary: {summary_csv}")
    print(f"Saved outputs: {out_path}")

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
