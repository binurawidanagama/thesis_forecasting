"""
Run ASP post-processing for ENERGY forecasting.

Examples (run run_energy_full.py first for the same LB/H):
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 24  --horizon 12
...
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

    child_cpu = 0.0
    try:
        ct = child.cpu_times()
        child_cpu = float(ct.user + ct.system)
    except Exception:
        child_cpu = 0.0

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
        summary_csv = str(base_out / "summary_dcenn_energy_asp.csv")
    summary_csv = Path(summary_csv)

    if raw_summary_csv is None:
        raw_summary_csv = str(base_out / "summary_dcenn_energy_raw.csv")
    raw_summary_csv = Path(raw_summary_csv)

    pred_path = out_path / "raw_energy.parquet"
    truth_path = out_path / "truth_energy.parquet"
    meta_path = out_path / "meta_energy.parquet"
    base_metrics_path = out_path / "base_metrics.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path} (run run_energy_full.py first)")
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing: {truth_path} (run run_energy_full.py first)")
    if not base_metrics_path.exists():
        raise FileNotFoundError(f"Missing: {base_metrics_path} (created by run_energy_full.py)")

    preds = pd.read_parquet(pred_path)
    truth = pd.read_parquet(truth_path)
    meta = pd.read_parquet(meta_path) if meta_path.exists() else None

    base_metrics = json.loads(base_metrics_path.read_text())
    BASE_MAE = float(base_metrics["BASE_MAE"])
    BASE_RMSE = float(base_metrics["BASE_RMSE"])
    BASE_sMAPE = float(base_metrics["BASE_sMAPE"])

    # Pull Params/Train stats/Size + raw inference from RAW summary for same LB/H
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
        df_match = df_raw[(df_raw["task"] == "ENERGY") & (df_raw["lookback"] == ctx) & (df_raw["horizon"] == hz)]
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

    asp_cfg = cfg.get("asp", {})
    night_hours = set(asp_cfg.get("solar_night_hours", []))
    program = asp_cfg.get("program", "src/asp/energy_physics.lp")
    target_map = asp_cfg.get("target_map", {"wind_mw":"wind","solar_mw":"solar","load_mw":"load"})

    # targets in config order
    targets_cfg = cfg["features"]["target_features"]
    asp_targets = [target_map[t] for t in targets_cfg]

    facts_path = out_path / "energy_facts.lp"
    BATCH = 150
    SCALE = 100

    res_mon = ResourceMonitor()

    cleaned = preds.copy()
    # need timestamps for night rules
    if "timestamp" not in cleaned.columns:
        cleaned["timestamp"] = cleaned.index

    # Parse repairs emitted by clingo:
    # repair(kind, target, sample, horizon)
    pattern = re.compile(r"repair\(\s*([a-z_]+)\s*,\s*([a-z_]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

    asp_wall_total = 0.0
    asp_cpu_total = 0.0

    print(f"\n[dCeNN ENERGY ASP] LB={ctx} H={hz} out={out_path}")
    print(f"Using ASP program: {program}")

    for start in tqdm(range(0, len(cleaned), BATCH), desc="ASP Batches"):
        end = min(start + BATCH, len(cleaned))
        chunk = cleaned.iloc[start:end]

        # 1) write facts
        with open(facts_path, "w") as f:
            for h in range(1, hz + 1):
                f.write(f"horizon({h}).\n")

            for i in range(len(chunk)):
                s_glob = start + i
                f.write(f"sample({s_glob}).\n")
                row = chunk.iloc[i]

                ts = row["timestamp"]
                for h in range(1, hz + 1):
                    # preds
                    for cfg_name, asp_name in target_map.items():
                        val = row.get(f"{cfg_name}+h{h}", 0.0)
                        f.write(f"pred({asp_name},{s_glob},{h},{int(val*SCALE)}).\n")

                    # night
                    hour_h = (ts.hour + h) % 24
                    if hour_h in night_hours:
                        f.write(f"night({s_glob},{h}).\n")

                    # capacity caps if available (optional)
                    if meta is not None and "cap_wind_mw" in meta.columns:
                        capw = float(meta.iloc[s_glob].get("cap_wind_mw", 0.0))
                        f.write(f"cap(wind,{s_glob},{h},{int(capw*SCALE)}).\n")
                    if meta is not None and "cap_solar_mw" in meta.columns:
                        caps = float(meta.iloc[s_glob].get("cap_solar_mw", 0.0))
                        f.write(f"cap(solar,{s_glob},{h},{int(caps*SCALE)}).\n")

        # 2) clingo
        cmd = ["clingo", program, str(facts_path), "--opt-mode=opt", "--quiet=1", "--time-limit=5"]
        out, wall, cpu = run_clingo_with_metrics(cmd, res_mon=res_mon)
        asp_wall_total += wall
        asp_cpu_total += cpu

        # 3) apply repairs
        for line in out.splitlines():
            matches = pattern.findall(line)
            for kind, tgt, s, h in matches:
                s, h = int(s), int(h)

                # map ASP target name back to config column base name
                cfg_col_base = None
                for k, v in target_map.items():
                    if v == tgt:
                        cfg_col_base = k
                        break
                if cfg_col_base is None:
                    continue

                col = f"{cfg_col_base}+h{h}"
                if col not in cleaned.columns:
                    continue

                idx_label = cleaned.index[s]
                curr = float(cleaned.at[idx_label, col])

                if kind == "bound":
                    cleaned.at[idx_label, col] = max(0.0, curr)

                elif kind == "cap":
                    # if cap exists, clamp down
                    cap_val = None
                    if meta is not None:
                        if tgt == "wind" and "cap_wind_mw" in meta.columns:
                            cap_val = float(meta.iloc[s].get("cap_wind_mw", curr))
                        if tgt == "solar" and "cap_solar_mw" in meta.columns:
                            cap_val = float(meta.iloc[s].get("cap_solar_mw", curr))
                    if cap_val is not None:
                        cleaned.at[idx_label, col] = min(curr, cap_val)

                elif kind == "night" and tgt == "solar":
                    cleaned.at[idx_label, col] = 0.0

        res_mon.update()

    # cleanup facts
    try:
        if facts_path.exists():
            facts_path.unlink()
    except Exception:
        pass

    # drop timestamp helper column
    if "timestamp" in cleaned.columns:
        cleaned = cleaned.drop(columns=["timestamp"])

    out_clean = out_path / "clean_energy.parquet"
    cleaned.to_parquet(out_clean)

    # metrics: reshape to [N,H,C] using cfg order
    cols = []
    for h in range(hz):
        for name in targets_cfg:
            cols.append(f"{name}+h{h+1}")

    truth_al = truth[cols].to_numpy(dtype=np.float64)
    pred_al = cleaned[cols].to_numpy(dtype=np.float64)

    m = min(len(truth_al), len(pred_al))
    truth_al = truth_al[:m].reshape(m, hz, len(targets_cfg))
    pred_al  = pred_al[:m].reshape(m, hz, len(targets_cfg))

    MAE, RMSE, sMAPE = calc_metrics(truth_al, pred_al)

    # inference for ASP variant = RAW inference + ASP solver time
    Infer_Wall_Sec = float(Raw_Infer_Wall + asp_wall_total)
    Infer_CPU_Sec  = float(Raw_Infer_CPU + asp_cpu_total)
    Infer_Avg_CPU_Pct = 100.0 * (Infer_CPU_Sec / Infer_Wall_Sec) if Infer_Wall_Sec > 0 else 0.0
    Latency_ms = (Infer_Wall_Sec * 1000.0) / max(1, m)

    Peak_RAM_MB = float(max(Peak_RAM_prev, res_mon.peak_ram_mb, res_mon.peak_child_mb))

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

    print(f"\n[DONE-ASP] ENERGY LB={ctx} H={hz}")
    print(f"MAE={MAE:.4f} RMSE={RMSE:.4f} sMAPE={sMAPE:.2f}% | (BASE_MAE={BASE_MAE:.4f})")
    print(f"Appended summary: {summary_csv}")
    print(f"Saved: {out_clean}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/energy_full.yaml")
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
