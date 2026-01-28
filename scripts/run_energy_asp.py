"""
Run ASP post-processing for ENERGY forecasting.

Examples (run run_energy_full.py first for the same LB/H):
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 24 --horizon 12
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 24 --horizon 24
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 24 --horizon 72

python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 72 --horizon 12
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 72 --horizon 24
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 72 --horizon 72

python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 168 --horizon 12
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 168 --horizon 24
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 168 --horizon 72
...
"""

import os
import re
import time
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config


# -----------------------------
# Base + extended summary schema
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
    "Size_MB",
]

EXTRA_HEADER = [
    "Train_Params","Deploy_Params",
    "Train_Size_MB","Deploy_Size_MB",
]

ASP_HEADER = [
    "RAW_MAE","RAW_RMSE","RAW_sMAPE",
    "Repairs_Total",
    "Cells_Changed",
    "Repair_Cell_Rate_Pct",
    "Mean_Abs_Adjustment",
    "Max_Adjustment",
    "Repairs_ByKind_JSON",
    "Repairs_ByTarget_JSON",
    "Repairs_After_Total",   # only filled if --check_after
]

HEADER_V2 = BASE_HEADER + EXTRA_HEADER + ASP_HEADER


# -----------------------------
# Process + metrics helpers
# -----------------------------
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

    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        child = psutil.Process(p.pid)
    except Exception as e:
        print(f"[ERROR] Failed to start Clingo: {e}")
        return "", 0.0, 0.0

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

    if err and "error" in err.lower():
        print(f"[CLINGO ERROR] {err.strip()}")

    return out, wall, (py_cpu + child_cpu)


def _safe_int_scaled(x, scale: int):
    """Convert possibly-nan to int(scale*x) safely."""
    try:
        if x is None:
            return 0
        xf = float(x)
        if not np.isfinite(xf):
            return 0
        return int(xf * scale)
    except Exception:
        return 0


# -----------------------------
# Main ASP runner
# -----------------------------
def run_asp(
    cfg_path: str,
    lookback=None,
    horizon=None,
    out_dir=None,
    summary_csv=None,
    raw_summary_csv=None,
    check_after: bool = False,
):
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

    # Pull Params/Train stats/Size + raw inference from RAW summary
    Params = 0
    Train_Wall_Sec = 0.0
    Train_CPU_Sec = 0.0
    Avg_CPU_Usage_Pct = 0.0
    Size_MB = 0.0
    Raw_Infer_Wall = 0.0
    Raw_Infer_CPU = 0.0
    Peak_RAM_prev = 0.0

    Train_Params = np.nan
    Deploy_Params = np.nan
    Train_Size_MB = np.nan
    Deploy_Size_MB = np.nan

    if raw_summary_csv.exists():
        df_raw = pd.read_csv(raw_summary_csv)
        df_match = df_raw[(df_raw.get("task") == "ENERGY") & (df_raw.get("lookback") == ctx) & (df_raw.get("horizon") == hz)]
        if len(df_match) > 0:
            last = df_match.iloc[-1]
            Params = int(last.get("Params", 0))
            Train_Wall_Sec = float(last.get("Train_Wall_Sec", 0.0))
            Train_CPU_Sec = float(last.get("Train_CPU_Sec", 0.0))
            Avg_CPU_Usage_Pct = float(last.get("Avg_CPU_Usage_Pct", 0.0))
            Size_MB = float(last.get("Size_MB", 0.0))
            Raw_Infer_Wall = float(last.get("Infer_Wall_Sec", 0.0))
            Raw_Infer_CPU = float(last.get("Infer_CPU_Sec", 0.0))
            Peak_RAM_prev = float(last.get("Peak_RAM_MB", 0.0))

            # v2 passthrough if you added it in run_energy_full.py
            if "Train_Params" in df_raw.columns:
                Train_Params = float(last.get("Train_Params", np.nan))
            if "Deploy_Params" in df_raw.columns:
                Deploy_Params = float(last.get("Deploy_Params", np.nan))
            if "Train_Size_MB" in df_raw.columns:
                Train_Size_MB = float(last.get("Train_Size_MB", np.nan))
            if "Deploy_Size_MB" in df_raw.columns:
                Deploy_Size_MB = float(last.get("Deploy_Size_MB", np.nan))

            # If v2 exists, treat legacy as deploy for consistency
            if np.isfinite(Deploy_Params):
                Params = int(Deploy_Params)
            if np.isfinite(Deploy_Size_MB):
                Size_MB = float(Deploy_Size_MB)

    asp_cfg = cfg.get("asp", {})
    program = asp_cfg.get("program", "src/asp/core_asp.lp")  # default to your optimization program

    target_map = asp_cfg.get("target_map", {"wind_mw": "wind", "solar_mw": "solar", "load_mw": "load"})
    asp_to_cfg = {v: k for k, v in target_map.items()}

    night_hours = set(asp_cfg.get("solar_night_hours", [22, 23, 0, 1, 2, 3]))

    # Seasonal CF Limits (Matches core_asp.lp)
    SEASONAL_LIMITS = {
        1: 40, 2: 55, 3: 80, 4: 95, 5: 100, 6: 100,
        7: 100, 8: 95, 9: 85, 10: 65, 11: 45, 12: 35
    }

    targets_cfg = cfg["features"]["target_features"]
    facts_path = out_path / "energy_facts.lp"

    BATCH = int(asp_cfg.get("batch_size", 10))
    SCALE = int(asp_cfg.get("scale", 100))

    res_mon = ResourceMonitor()

    cleaned = preds.copy()

    # Ensure timestamp column exists (if it's in the index, move to column)
    if "timestamp" not in cleaned.columns:
        cleaned["timestamp"] = cleaned.index
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")

    # Align meta to cleaned order by index (safer than assuming .iloc matches)
    meta_aligned = meta.reindex(cleaned.index) if meta is not None else None

    # Patterns:
    # 1) repair(kind, target, s, h)
    pattern_repair = re.compile(r"repair\(\s*([a-z_][a-z0-9_]*)\s*,\s*([a-z_][a-z0-9_]*)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
    # 2) set(target, s, h, value)  (optional, if you ever add it to ASP)
    pattern_set = re.compile(r"(?:set|fixed|newpred|repair_value)\(\s*([a-z_][a-z0-9_]*)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+)\s*\)")

    # Evidence counters
    repairs_total = 0
    repairs_by_kind = defaultdict(int)
    repairs_by_target = defaultdict(int)
    changed_cells = set()  # (s, h, tgt)
    deltas = []            # abs adjustments

    asp_wall_total = 0.0
    asp_cpu_total = 0.0

    # Pre-metrics (raw -> before ASP)
    cols = [f"{name}+h{h+1}" for h in range(hz) for name in targets_cfg]
    truth_np = truth[cols].to_numpy(dtype=np.float64)
    pred_np  = preds[cols].to_numpy(dtype=np.float64)
    m0 = min(len(truth_np), len(pred_np))
    truth_np = truth_np[:m0].reshape(m0, hz, len(targets_cfg))
    pred_np  = pred_np[:m0].reshape(m0, hz, len(targets_cfg))
    RAW_MAE, RAW_RMSE, RAW_sMAPE = calc_metrics(truth_np, pred_np)

    print(f"\n[dCeNN ENERGY ASP] LB={ctx} H={hz} out={out_path}")
    print(f"ASP program: {program}")
    print(f"Batch Size: {BATCH} | SCALE={SCALE} | check_after={check_after}")

    for start in tqdm(range(0, len(cleaned), BATCH), desc="ASP Batches"):
        end = min(start + BATCH, len(cleaned))
        chunk = cleaned.iloc[start:end]

        # ---------------------------------------------------------
        # 1) WRITE FACTS  (correct: pred/night for EVERY horizon step)
        # ---------------------------------------------------------
        with open(facts_path, "w") as f:
            for h in range(1, hz + 1):
                f.write(f"horizon({h}).\n")

            for i in range(len(chunk)):
                s_glob = start + i
                row = chunk.iloc[i]
                ts = row["timestamp"]

                f.write(f"sample({s_glob}).\n")
                if isinstance(ts, pd.Timestamp) and pd.notna(ts):
                    f.write(f"month({s_glob},{ts.month}).\n")
                    f.write(f"hour0({s_glob},{ts.hour}).\n")
                else:
                    f.write(f"month({s_glob},1).\n")
                    f.write(f"hour0({s_glob},0).\n")

                for h in range(1, hz + 1):
                    for cfg_name, asp_name in target_map.items():
                        v = row.get(f"{cfg_name}+h{h}", 0.0)
                        f.write(f"pred({asp_name},{s_glob},{h},{_safe_int_scaled(v, SCALE)}).\n")

                    # Night fact (solar) for this horizon
                    if isinstance(ts, pd.Timestamp) and pd.notna(ts):
                        hour_h = (ts.hour + h) % 24
                        if hour_h in night_hours:
                            f.write(f"night({s_glob},{h}).\n")

                # Capacity facts (arity 3), once per sample
                if meta_aligned is not None and s_glob < len(meta_aligned):
                    mrow = meta_aligned.iloc[s_glob]
                    if "cap_wind_mw" in meta_aligned.columns:
                        capw = mrow.get("cap_wind_mw", 0.0)
                        f.write(f"cap(wind,{s_glob},{_safe_int_scaled(capw, SCALE)}).\n")
                    if "cap_solar_mw" in meta_aligned.columns:
                        caps = mrow.get("cap_solar_mw", 0.0)
                        f.write(f"cap(solar,{s_glob},{_safe_int_scaled(caps, SCALE)}).\n")

        # ---------------------------------------------------------
        # 2) RUN CLINGO
        # ---------------------------------------------------------
        cmd = ["clingo", program, str(facts_path), "--opt-mode=opt", "--quiet=1", "--time-limit=5"]
        out, wall, cpu = run_clingo_with_metrics(cmd, res_mon=res_mon)
        asp_wall_total += wall
        asp_cpu_total += cpu

        # ---------------------------------------------------------
        # 3) APPLY REPAIRS + LOG EVIDENCE
        # ---------------------------------------------------------
        for line in out.splitlines():
            # 3a) set(...) style (if present)
            for tgt, s, h, v in pattern_set.findall(line):
                s, h, v = int(s), int(h), int(v)

                repairs_total += 1
                repairs_by_kind["set"] += 1
                repairs_by_target[tgt] += 1

                if s < 0 or s >= len(cleaned):
                    continue

                cfg_col_base = asp_to_cfg.get(tgt, None)
                if cfg_col_base is None:
                    continue

                col = f"{cfg_col_base}+h{h}"
                if col not in cleaned.columns:
                    continue

                idx_label = cleaned.index[s]
                curr = float(cleaned.at[idx_label, col])
                newv = float(v) / float(SCALE)

                if np.isfinite(curr) and np.isfinite(newv) and newv != curr:
                    cleaned.at[idx_label, col] = newv
                    changed_cells.add((s, h, tgt))
                    deltas.append(abs(newv - curr))

            # 3b) repair(kind, tgt, s, h)
            for kind, tgt, s, h in pattern_repair.findall(line):
                s, h = int(s), int(h)

                repairs_total += 1
                repairs_by_kind[kind] += 1
                repairs_by_target[tgt] += 1

                if s < 0 or s >= len(cleaned):
                    continue

                cfg_col_base = asp_to_cfg.get(tgt, None)
                if cfg_col_base is None:
                    continue

                col = f"{cfg_col_base}+h{h}"
                if col not in cleaned.columns:
                    continue

                idx_label = cleaned.index[s]
                curr = float(cleaned.at[idx_label, col])
                newv = curr

                if kind == "bound_low":
                    newv = max(0.0, curr)

                elif kind == "bound_cap":
                    cap_val = 99999.0
                    if meta_aligned is not None and s < len(meta_aligned):
                        if tgt == "wind" and "cap_wind_mw" in meta_aligned.columns:
                            cap_val = float(meta_aligned.iloc[s].get("cap_wind_mw", 99999.0))
                        elif tgt == "solar" and "cap_solar_mw" in meta_aligned.columns:
                            cap_val = float(meta_aligned.iloc[s].get("cap_solar_mw", 99999.0))
                    newv = min(curr, cap_val)

                elif kind == "pv_night":
                    newv = 0.0

                elif kind == "seasonal_cap":
                    if meta_aligned is not None and "cap_solar_mw" in meta_aligned.columns and s < len(meta_aligned):
                        cap_val = float(meta_aligned.iloc[s].get("cap_solar_mw", 0.0))
                        ts2 = cleaned.iloc[s]["timestamp"]
                        m_idx = int(ts2.month) if isinstance(ts2, pd.Timestamp) and pd.notna(ts2) else 1
                        limit_pct = SEASONAL_LIMITS.get(m_idx, 100)
                        allowed = (cap_val * limit_pct) / 100.0
                        newv = min(curr, allowed)

                if np.isfinite(curr) and np.isfinite(newv) and newv != curr:
                    cleaned.at[idx_label, col] = newv
                    changed_cells.add((s, h, tgt))
                    deltas.append(abs(newv - curr))

        res_mon.update()

        # Optional: rerun clingo on repaired chunk to estimate remaining repairs
        if check_after:
            chunk2 = cleaned.iloc[start:end]
            with open(facts_path, "w") as f:
                for h in range(1, hz + 1):
                    f.write(f"horizon({h}).\n")
                for i in range(len(chunk2)):
                    s_glob = start + i
                    row = chunk2.iloc[i]
                    ts = row["timestamp"]
                    f.write(f"sample({s_glob}).\n")
                    if isinstance(ts, pd.Timestamp) and pd.notna(ts):
                        f.write(f"month({s_glob},{ts.month}).\n")
                        f.write(f"hour0({s_glob},{ts.hour}).\n")
                    else:
                        f.write(f"month({s_glob},1).\n")
                        f.write(f"hour0({s_glob},0).\n")
                    for h in range(1, hz + 1):
                        for cfg_name, asp_name in target_map.items():
                            v = row.get(f"{cfg_name}+h{h}", 0.0)
                            f.write(f"pred({asp_name},{s_glob},{h},{_safe_int_scaled(v, SCALE)}).\n")
                        if isinstance(ts, pd.Timestamp) and pd.notna(ts):
                            hour_h = (ts.hour + h) % 24
                            if hour_h in night_hours:
                                f.write(f"night({s_glob},{h}).\n")
                    if meta_aligned is not None and s_glob < len(meta_aligned):
                        mrow = meta_aligned.iloc[s_glob]
                        if "cap_wind_mw" in meta_aligned.columns:
                            capw = mrow.get("cap_wind_mw", 0.0)
                            f.write(f"cap(wind,{s_glob},{_safe_int_scaled(capw, SCALE)}).\n")
                        if "cap_solar_mw" in meta_aligned.columns:
                            caps = mrow.get("cap_solar_mw", 0.0)
                            f.write(f"cap(solar,{s_glob},{_safe_int_scaled(caps, SCALE)}).\n")

            out2, w2, c2 = run_clingo_with_metrics(cmd, res_mon=res_mon)
            asp_wall_total += w2
            asp_cpu_total += c2

            _after_repairs = 0
            for line2 in out2.splitlines():
                _after_repairs += len(pattern_set.findall(line2))
                _after_repairs += len(pattern_repair.findall(line2))

            try:
                repairs_after_total += _after_repairs
            except NameError:
                repairs_after_total = _after_repairs

    # Cleanup
    try:
        if facts_path.exists():
            facts_path.unlink()
    except Exception:
        pass

    if "timestamp" in cleaned.columns:
        cleaned = cleaned.drop(columns=["timestamp"])

    out_clean = out_path / "clean_energy.parquet"
    cleaned.to_parquet(out_clean)

    # ---------------------------------------------------------
    # FINAL METRICS
    # ---------------------------------------------------------
    truth_al = truth[cols].to_numpy(dtype=np.float64)
    pred_al  = cleaned[cols].to_numpy(dtype=np.float64)

    m = min(len(truth_al), len(pred_al))
    truth_al = truth_al[:m].reshape(m, hz, len(targets_cfg))
    pred_al  = pred_al[:m].reshape(m, hz, len(targets_cfg))

    MAE, RMSE, sMAPE = calc_metrics(truth_al, pred_al)

    # Additive Costs
    Infer_Wall_Sec = float(Raw_Infer_Wall + asp_wall_total)
    Infer_CPU_Sec  = float(Raw_Infer_CPU + asp_cpu_total)
    Infer_Avg_CPU_Pct = 100.0 * (Infer_CPU_Sec / Infer_Wall_Sec) if Infer_Wall_Sec > 0 else 0.0
    Latency_ms = (Infer_Wall_Sec * 1000.0) / max(1, m)
    Peak_RAM_MB = float(max(Peak_RAM_prev, res_mon.peak_ram_mb, res_mon.peak_child_mb))

    total_cells = max(1, m * hz * len(targets_cfg))
    cells_changed = int(len(changed_cells))
    mean_abs_adj = float(np.mean(deltas)) if len(deltas) else 0.0
    max_adj = float(np.max(deltas)) if len(deltas) else 0.0
    repair_cell_rate = 100.0 * (cells_changed / total_cells)

    if check_after:
        repairs_after = float(locals().get("repairs_after_total", 0))
    else:
        repairs_after = np.nan

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
        "Peak_RAM_MB": float(Peak_RAM_MB),
        "Infer_Wall_Sec": float(Infer_Wall_Sec),
        "Infer_CPU_Sec": float(Infer_CPU_Sec),
        "Infer_Avg_CPU_Pct": float(Infer_Avg_CPU_Pct),
        "Latency_ms_per_sample": float(Latency_ms),
        "Size_MB": float(Size_MB),

        # v2 passthrough if present
        "Train_Params": Train_Params,
        "Deploy_Params": Deploy_Params,
        "Train_Size_MB": Train_Size_MB,
        "Deploy_Size_MB": Deploy_Size_MB,

        # ASP evidence
        "RAW_MAE": float(RAW_MAE),
        "RAW_RMSE": float(RAW_RMSE),
        "RAW_sMAPE": float(RAW_sMAPE),
        "Repairs_Total": int(repairs_total),
        "Cells_Changed": int(cells_changed),
        "Repair_Cell_Rate_Pct": float(repair_cell_rate),
        "Mean_Abs_Adjustment": float(mean_abs_adj),
        "Max_Adjustment": float(max_adj),
        "Repairs_ByKind_JSON": json.dumps(dict(repairs_by_kind), sort_keys=True),
        "Repairs_ByTarget_JSON": json.dumps(dict(repairs_by_target), sort_keys=True),
        "Repairs_After_Total": float(repairs_after),
    }

    # ---------------------------------------------------------
    # SAVE SUMMARY (DEDUPLICATED + HEADER UPGRADE)
    # ---------------------------------------------------------
    df_new = pd.DataFrame([[row.get(h, np.nan) for h in HEADER_V2]], columns=HEADER_V2)

    if summary_csv.exists():
        df_existing = pd.read_csv(summary_csv)
        for col in HEADER_V2:
            if col not in df_existing.columns:
                df_existing[col] = np.nan

        mask = (
            (df_existing["task"] == "ENERGY") &
            (df_existing["lookback"] == ctx) &
            (df_existing["horizon"] == hz)
        )
        df_existing = df_existing[~mask]

        df_final = pd.concat([df_existing[HEADER_V2], df_new], ignore_index=True)
        df_final = df_final.sort_values(["task", "lookback", "horizon"])
        df_final.to_csv(summary_csv, index=False)
    else:
        df_new.to_csv(summary_csv, index=False)

    print(f"\n[DONE-ASP] ENERGY LB={ctx} H={hz}")
    print(f"RAW:  MAE={RAW_MAE:.4f} RMSE={RAW_RMSE:.4f} sMAPE={RAW_sMAPE:.2f}%")
    print(f"ASP:  MAE={MAE:.4f} RMSE={RMSE:.4f} sMAPE={sMAPE:.2f}%")
    print(f"Repairs_Total={repairs_total} | Cells_Changed={cells_changed} ({repair_cell_rate:.4f}% of all cells)")
    print(f"Mean|Δ|={mean_abs_adj:.6g}  Max|Δ|={max_adj:.6g}")
    if check_after:
        print(f"Repairs_After_Total={repairs_after:.0f}  (lower is better)")
    print(f"Updated summary: {summary_csv}")
    print(f"Saved: {out_clean}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/energy_full.yaml")
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)
    ap.add_argument("--raw_summary_csv", type=str, default=None)
    ap.add_argument("--check_after", action="store_true", help="rerun clingo on repaired chunks to estimate remaining repairs (slower)")
    args = ap.parse_args()

    run_asp(
        args.config,
        lookback=args.lookback,
        horizon=args.horizon,
        out_dir=args.out_dir,
        summary_csv=args.summary_csv,
        raw_summary_csv=args.raw_summary_csv,
        check_after=bool(args.check_after),
    )
