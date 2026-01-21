"""
Run dCeNN ASP post-processing for weather forecasting.
python run_weather_asp.py --config configs/weather_full.yaml --lookback 24  --horizon 12
python run_weather_asp.py --config configs/weather_full.yaml --lookback 24  --horizon 24
python run_weather_asp.py --config configs/weather_full.yaml --lookback 24  --horizon 72

python run_weather_asp.py --config configs/weather_full.yaml --lookback 72  --horizon 12
python run_weather_asp.py --config configs/weather_full.yaml --lookback 72  --horizon 24
python run_weather_asp.py --config configs/weather_full.yaml --lookback 72  --horizon 72

python run_weather_asp.py --config configs/weather_full.yaml --lookback 168 --horizon 12
python run_weather_asp.py --config configs/weather_full.yaml --lookback 168 --horizon 24
python run_weather_asp.py --config configs/weather_full.yaml --lookback 168 --horizon 72
"""


import os
import re
import time
import json
import argparse
import subprocess
from pathlib import Path

import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config


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


def get_process_metrics(pid=None):
    p = psutil.Process(pid) if pid else psutil.Process(os.getpid())
    with p.oneshot():
        mem_mb = p.memory_info().rss / (1024 * 1024)
        cpu = p.cpu_times()
        cpu_time_s = float(cpu.user + cpu.system)
    return float(mem_mb), float(cpu_time_s)


class ResourceMonitor:
    def __init__(self):
        self.peak_ram_mb = get_process_metrics()[0]
        self.peak_child_mb = 0.0

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


def run_clingo_with_metrics(cmd, res_mon: ResourceMonitor):
    t0 = time.time()
    cpu0 = get_process_metrics()[1]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = psutil.Process(p.pid)

    peak_child = 0.0
    while True:
        if p.poll() is not None:
            break
        try:
            peak_child = max(peak_child, child.memory_info().rss / (1024 * 1024))
        except Exception:
            pass
        res_mon.update()
        time.sleep(0.01)

    out, err = p.communicate()
    wall = time.time() - t0

    # child CPU time
    child_cpu = 0.0
    try:
        ct = child.cpu_times()
        child_cpu = float(ct.user + ct.system)
    except Exception:
        child_cpu = 0.0

    # python CPU time used during this call (lightweight)
    py_cpu = get_process_metrics()[1] - cpu0

    res_mon.peak_child_mb = max(res_mon.peak_child_mb, float(peak_child))
    return out, wall, (py_cpu + child_cpu)


def run_asp(cfg_path: str, lookback=None, horizon=None, out_dir=None, summary_csv=None, raw_summary_csv=None):
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
        summary_csv = str(base_out / "summary_dcenn_weather_asp.csv")
    summary_csv = Path(summary_csv)

    if raw_summary_csv is None:
        raw_summary_csv = str(base_out / "summary_dcenn_weather_raw.csv")
    raw_summary_csv = Path(raw_summary_csv)

    pred_path = out_path / "raw_weather.parquet"
    truth_path = out_path / "truth_weather.parquet"
    base_metrics_path = out_path / "base_metrics.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path} (run run_weather_full.py first)")
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing: {truth_path} (run run_weather_full.py first)")
    if not base_metrics_path.exists():
        raise FileNotFoundError(f"Missing: {base_metrics_path} (created by run_weather_full.py)")

    preds = pd.read_parquet(pred_path)
    truth = pd.read_parquet(truth_path)

    base_metrics = json.loads(base_metrics_path.read_text())
    BASE_MAE = float(base_metrics["BASE_MAE"])
    BASE_RMSE = float(base_metrics["BASE_RMSE"])
    BASE_sMAPE = float(base_metrics["BASE_sMAPE"])

    # Pull Params/Train stats/Size from RAW summary row for same LB/H (fair end-to-end reporting)
    Params = 0
    Train_Wall_Sec = 0.0
    Train_CPU_Sec = 0.0
    Avg_CPU_Usage_Pct = 0.0
    Size_MB = 0.0
    Raw_Infer_Wall = 0.0
    Raw_Infer_CPU = 0.0
    Peak_RAM_prev = 0.0

    if raw_summary_csv.exists():
        df_raw = pd.read_csv(raw_summary_csv)
        df_match = df_raw[(df_raw["task"] == "WEATHER") & (df_raw["lookback"] == ctx) & (df_raw["horizon"] == hz)]
        if len(df_match) > 0:
            last = df_match.iloc[-1]
            Params = int(last["Params"])
            Train_Wall_Sec = float(last["Train_Wall_Sec"])
            Train_CPU_Sec = float(last["Train_CPU_Sec"])
            Avg_CPU_Usage_Pct = float(last["Avg_CPU_Usage_Pct"])
            Size_MB = float(last["Size_MB"])
            Raw_Infer_Wall = float(last["Infer_Wall_Sec"])
            Raw_Infer_CPU = float(last["Infer_CPU_Sec"])
            Peak_RAM_prev = float(last["Peak_RAM_MB"])

    # ASP mappings
    t_map = {
        "temperature_2m_C": "temp",
        "relative_humidity_2m_pct": "hum",
        "wind_speed_100m (m/s)": "wind",
        "surface_pressure_hPa": "press",
        "shortwave_radiation_Wm2": "rad",
        "precipitation_mm": "precip",
    }
    targets = cfg["features"]["target_features"]

    night_hours = set(cfg["asp"]["pv_night_hours"])
    facts_path = out_path / "weather_facts.lp"

    BATCH = 100
    SCALE = 100

    res_mon = ResourceMonitor()

    cleaned = preds.copy()
    if "timestamp" not in cleaned.columns:
        cleaned["timestamp"] = cleaned.index

    pattern = re.compile(r"repair\(\s*([a-z_]+)\s*,\s*([a-z]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

    asp_wall_total = 0.0
    asp_cpu_total = 0.0

    print(f"\n[dCeNN WEATHER ASP] LB={ctx} H={hz} out={out_path}")
    print("Applying ASP...")

    for start in tqdm(range(0, len(cleaned), BATCH), desc="ASP Batches"):
        end = min(start + BATCH, len(cleaned))
        chunk = cleaned.iloc[start:end]

        # 1) facts
        with open(facts_path, "w") as f:
            for h in range(1, hz + 1):
                f.write(f"horizon({h}).\n")

            for i in range(len(chunk)):
                s_glob = start + i
                f.write(f"sample({s_glob}).\n")
                row = chunk.iloc[i]
                for h in range(1, hz + 1):
                    for t_col, asp_name in t_map.items():
                        val = row.get(f"{t_col}+h{h}", 0.0)
                        f.write(f"pred({asp_name},{s_glob},{h},{int(val*SCALE)}).\n")

                    ts = row["timestamp"]
                    hour_h = (ts.hour + h) % 24
                    if hour_h in night_hours:
                        f.write(f"night({s_glob},{h}).\n")

        # 2) clingo
        cmd = ["clingo", "src/asp/weather_physics.lp", str(facts_path), "--opt-mode=opt", "--quiet=1", "--time-limit=5"]
        out, wall, cpu = run_clingo_with_metrics(cmd, res_mon=res_mon)
        asp_wall_total += wall
        asp_cpu_total += cpu

        # 3) parse repairs
        for line in out.splitlines():
            matches = pattern.findall(line)
            for kind, tgt, s, h in matches:
                s, h = int(s), int(h)

                col = ""
                for k, v in t_map.items():
                    if v == tgt:
                        col = f"{k}+h{h}"
                        break
                if not col or col not in cleaned.columns:
                    continue

                idx_label = cleaned.index[s]
                curr = cleaned.at[idx_label, col]

                if kind == "bound":
                    if tgt == "hum":
                        cleaned.at[idx_label, col] = max(0, min(100, curr))
                    elif tgt == "press":
                        cleaned.at[idx_label, col] = max(850, min(1100, curr))
                    elif tgt == "temp":
                        cleaned.at[idx_label, col] = max(-30, min(45, curr))
                    else:
                        cleaned.at[idx_label, col] = max(0, curr)
                elif kind == "night" and tgt == "rad":
                    cleaned.at[idx_label, col] = 0.0
                elif kind == "noise_gate" and tgt == "precip":
                    cleaned.at[idx_label, col] = 0.0
                elif kind == "boost_hum" and tgt == "hum":
                    cleaned.at[idx_label, col] = max(curr, 75.0)

        res_mon.update()

    # Safety pass for night radiation
    start_hours = cleaned["timestamp"].dt.hour.values
    hours_2d = (start_hours[:, None] + np.arange(1, hz + 1)[None, :]) % 24
    is_night = np.isin(hours_2d, list(night_hours))

    rad_col_base = "shortwave_radiation_Wm2"
    for h_idx in range(hz):
        h = h_idx + 1
        col = f"{rad_col_base}+h{h}"
        if col in cleaned.columns:
            cleaned.loc[is_night[:, h_idx], col] = 0.0

    if "timestamp" in cleaned.columns:
        cleaned = cleaned.drop(columns=["timestamp"])

    out_clean = out_path / "clean_weather.parquet"
    cleaned.to_parquet(out_clean)

    try:
        if facts_path.exists():
            facts_path.unlink()
    except Exception:
        pass

    # Metrics: reshape to [N,H,C]
    cols = []
    for h in range(hz):
        for name in targets:
            cols.append(f"{name}+h{h+1}")

    truth_al = truth[cols].to_numpy(dtype=np.float64)
    pred_al = cleaned[cols].to_numpy(dtype=np.float64)

    m = min(len(truth_al), len(pred_al))
    truth_al = truth_al[:m].reshape(m, hz, len(targets))
    pred_al  = pred_al[:m].reshape(m, hz, len(targets))

    MAE, RMSE, sMAPE = calc_metrics(truth_al, pred_al)

    # Inference for ASP variant = RAW inference + ASP solver time
    Infer_Wall_Sec = float(Raw_Infer_Wall + asp_wall_total)
    Infer_CPU_Sec  = float(Raw_Infer_CPU + asp_cpu_total)
    Infer_Avg_CPU_Pct = 100.0 * (Infer_CPU_Sec / Infer_Wall_Sec) if Infer_Wall_Sec > 0 else 0.0
    Latency_ms = (Infer_Wall_Sec * 1000.0) / max(1, m)

    # Peak RAM: include python + child clingo best-effort
    Peak_RAM_MB = float(max(Peak_RAM_prev, res_mon.peak_ram_mb, res_mon.peak_child_mb))

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
        "Params": int(Params),
        "Train_Wall_Sec": float(Train_Wall_Sec),
        "Train_CPU_Sec": float(Train_CPU_Sec),
        "Avg_CPU_Usage_Pct": float(Avg_CPU_Usage_Pct),
        "Peak_RAM_MB": Peak_RAM_MB,
        "Infer_Wall_Sec": float(Infer_Wall_Sec),
        "Infer_CPU_Sec": float(Infer_CPU_Sec),
        "Infer_Avg_CPU_Pct": float(Infer_Avg_CPU_Pct),
        "Latency_ms_per_sample": float(Latency_ms),
        "Size_MB": float(Size_MB),
    }

    df_row = pd.DataFrame([[row[h] for h in BASE_HEADER]], columns=BASE_HEADER)
    df_row.to_csv(summary_csv, mode="a", header=not summary_csv.exists(), index=False)

    print(f"\n[DONE-ASP] WEATHER LB={ctx} H={hz}")
    print(f"MAE={MAE:.4f} RMSE={RMSE:.4f} sMAPE={sMAPE:.2f}% | (BASE_MAE={BASE_MAE:.4f})")
    print(f"Appended summary: {summary_csv}")
    print(f"Saved: {out_clean}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/weather_full.yaml")
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)
    ap.add_argument("--raw_summary_csv", type=str, default=None)
    args = ap.parse_args()

    run_asp(
        args.config,
        lookback=args.lookback,
        horizon=args.horizon,
        out_dir=args.out_dir,
        summary_csv=args.summary_csv,
        raw_summary_csv=args.raw_summary_csv
    )
