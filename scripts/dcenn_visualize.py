"""
Thesis Visualizer (Baselines + dCeNN/ELM + ASP)

Updates:
- Defaults to plotting ONLY the final horizon step (e.g., h12 for H12).
- Removes 'hX' subfolders when only plotting one step.

Usage:
  python scripts/dcenn_visualize.py --compare_all --start_date 2022-01-10 --length 168
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Fixed grid
# -----------------------------
LOOKBACKS = [24, 72, 168]
HORIZONS  = [12, 24, 72]


# -----------------------------
# Targets
# -----------------------------
PREFERRED_TARGETS: Dict[str, List[str]] = {
    "energy": [
        "load_mw", "wind_mw", "solar_mw",
    ],
    "weather": [
        "temperature_2m_C", "shortwave_radiation_Wm2",
        "relative_humidity_2m_pct", "precipitation_mm",
        "wind_speed_100m", "surface_pressure_hPa",
    ],
}


# -----------------------------
# Helpers
# -----------------------------
def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "timestamp" in d.columns:
        ts = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(d.index, utc=True, errors="coerce")
    
    d["timestamp"] = ts
    d.index = ts
    d.index.name = None 
    d = d.dropna(subset=["timestamp"])
    d = d.sort_index()
    return d


def time_slice(df: pd.DataFrame, start_date: Optional[str], length_hours: int) -> pd.DataFrame:
    d = ensure_timestamp(df)
    if len(d) == 0: return d

    length_hours = int(length_hours)
    if start_date:
        start_ts = pd.Timestamp(start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        end_ts = start_ts + pd.Timedelta(hours=length_hours)

        sub = d.loc[(d.index >= start_ts) & (d.index < end_ts)]
        if len(sub) == 0:
            return d.iloc[-min(length_hours, len(d)):]
        return sub

    n = len(d)
    if n <= length_hours: return d
    start_idx = min(max(n // 2, 0), max(n - length_hours, 0))
    return d.iloc[start_idx:start_idx + length_hours]


def align_on_common_index(*dfs: Optional[pd.DataFrame]) -> List[Optional[pd.DataFrame]]:
    idx = None
    for d in dfs:
        if d is not None and len(d) > 0:
            idx = d.index if idx is None else idx.intersection(d.index)
    
    if idx is None: return list(dfs)

    out = []
    for d in dfs:
        out.append(d.reindex(idx) if d is not None else None)
    return out


def available_feature_basenames(df: pd.DataFrame) -> List[str]:
    feats = []
    for c in df.columns:
        if "+h" in c:
            feats.append(c.split("+h")[0])
    return sorted(set(feats))


def choose_features(task: str, df_raw: pd.DataFrame) -> List[str]:
    avail = set(available_feature_basenames(df_raw))
    pref = PREFERRED_TARGETS.get(task, [])
    picked = [f for f in pref if f in avail]
    if picked: return picked
    return sorted(avail)


def dcenn_col(feat: str, h_step: int) -> str:
    return f"{feat}+h{int(h_step)}"


def _find_col_substring(cols: List[str], needle: str) -> Optional[str]:
    if needle in cols: return needle
    for c in cols:
        if needle in c: return c
    return None


def find_baseline_true_pred_cols(df: pd.DataFrame, feat: str, h_step: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    if h_step is not None:
        t = _find_col_substring(cols, f"True_{feat}+h{h_step}")
        p = _find_col_substring(cols, f"Pred_{feat}+h{h_step}")
        if t and p: return t, p
    t = _find_col_substring(cols, f"True_{feat}")
    p = _find_col_substring(cols, f"Pred_{feat}")
    return t, p


# -----------------------------
# Plotting Functions
# -----------------------------
def plot_dcenn_raw_vs_truth(df_raw, df_true, feat, h_step, title, save_path):
    c = dcenn_col(feat, h_step)
    if c not in df_raw.columns or c not in df_true.columns: return

    plt.figure(figsize=(12, 5))
    plt.plot(df_true["timestamp"], df_true[c], label="Actual", linewidth=2.0, alpha=0.55, color="black")
    plt.plot(df_raw["timestamp"],  df_raw[c],  label="dCeNN (Raw)", linewidth=1.6, alpha=0.95)
    plt.title(title, fontsize=13)
    plt.ylabel(feat, fontsize=11)
    plt.xlabel("Time (UTC)", fontsize=10)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_dcenn_raw_clean_truth(df_raw, df_true, df_clean, feat, h_step, title, save_path):
    c = dcenn_col(feat, h_step)
    if c not in df_raw.columns or c not in df_true.columns: return
    has_clean = (df_clean is not None) and (c in df_clean.columns)

    plt.figure(figsize=(12, 5))
    plt.plot(df_true["timestamp"], df_true[c], label="Actual", linewidth=2.0, alpha=0.45, color="black")
    plt.plot(df_raw["timestamp"],  df_raw[c],  label="Before ASP (Raw)", linewidth=1.2, linestyle="--", alpha=0.9)
    if has_clean:
        plt.plot(df_clean["timestamp"], df_clean[c], label="After ASP (Clean)", linewidth=2.0, alpha=0.95)

    plt.title(title, fontsize=13)
    plt.ylabel(feat, fontsize=11)
    plt.xlabel("Time (UTC)", fontsize=10)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_all_models_vs_truth(df_true_dcenn, df_raw_dcenn, df_clean_dcenn, df_cnn, df_lstm, feat, h_step, title, save_path):
    c = dcenn_col(feat, h_step)
    if c not in df_true_dcenn.columns or c not in df_raw_dcenn.columns: return

    d_true = df_true_dcenn[["timestamp", c]].rename(columns={c: "truth"})
    d_raw  = df_raw_dcenn[["timestamp", c]].rename(columns={c: "dcenn_raw"})

    d_clean = None
    if df_clean_dcenn is not None and c in df_clean_dcenn.columns:
        d_clean = df_clean_dcenn[["timestamp", c]].rename(columns={c: "dcenn_asp"})

    d_cnn = None
    if df_cnn is not None and len(df_cnn) > 0:
        _, pcol = find_baseline_true_pred_cols(df_cnn, feat, h_step)
        if pcol is None:
            pcol = _find_col_substring(list(df_cnn.columns), f"{feat}+h{h_step}") or _find_col_substring(list(df_cnn.columns), feat)
        if pcol is not None:
            d_cnn = ensure_timestamp(df_cnn)[["timestamp", pcol]].rename(columns={pcol: "cnn"})

    d_lstm = None
    if df_lstm is not None and len(df_lstm) > 0:
        _, pcol = find_baseline_true_pred_cols(df_lstm, feat, h_step)
        if pcol is None:
            pcol = _find_col_substring(list(df_lstm.columns), f"{feat}+h{h_step}") or _find_col_substring(list(df_lstm.columns), feat)
        if pcol is not None:
            d_lstm = ensure_timestamp(df_lstm)[["timestamp", pcol]].rename(columns={pcol: "lstm"})

    base = ensure_timestamp(d_true)
    raw  = ensure_timestamp(d_raw)
    clean = ensure_timestamp(d_clean) if d_clean is not None else None
    cnn  = ensure_timestamp(d_cnn) if d_cnn is not None else None
    lstm = ensure_timestamp(d_lstm) if d_lstm is not None else None

    base, raw, clean, cnn, lstm = align_on_common_index(base, raw, clean, cnn, lstm)
    if base is None or raw is None or len(base) == 0: return

    plt.figure(figsize=(12, 5))
    plt.plot(base["timestamp"], base["truth"], label="Actual (Truth)", linewidth=2.2, alpha=0.55, color="black")
    plt.plot(raw["timestamp"],  raw["dcenn_raw"], label="dCeNN Raw", linewidth=1.4, linestyle="--", alpha=0.9)
    if clean is not None and "dcenn_asp" in clean.columns:
        plt.plot(clean["timestamp"], clean["dcenn_asp"], label="dCeNN + ASP", linewidth=2.0, alpha=0.95)
    if lstm is not None and "lstm" in lstm.columns:
        plt.plot(lstm["timestamp"], lstm["lstm"], label="LSTM", linewidth=1.4, alpha=0.9)
    if cnn is not None and "cnn" in cnn.columns:
        plt.plot(cnn["timestamp"], cnn["cnn"], label="CNN", linewidth=1.4, alpha=0.9)

    plt.title(title, fontsize=13)
    plt.ylabel(feat, fontsize=11)
    plt.xlabel("Time (UTC)", fontsize=10)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# Loaders
# -----------------------------
def load_dcenn_triplet(root: Path, task: str, lb: int, hz: int):
    folder = root / f"LB{lb}_H{hz}"
    raw_p   = folder / f"raw_{task}.parquet"
    truth_p = folder / f"truth_{task}.parquet"
    clean_p = folder / f"clean_{task}.parquet"
    if not raw_p.exists() or not truth_p.exists(): return None, None, None
    return pd.read_parquet(raw_p), pd.read_parquet(truth_p), (pd.read_parquet(clean_p) if clean_p.exists() else None)

def load_baseline_preds(baseline_root: Path, lb: int, hz: int, task: str):
    folder_name = f"LB{lb}_H{hz}_{task.upper()}"
    p = baseline_root / folder_name / "preds.parquet"
    if not p.exists(): return None
    try: return pd.read_parquet(p)
    except: return None


# -----------------------------
# Configuration Logic (Updated)
# -----------------------------
def get_h_steps(h_steps_arg: str, hz: int) -> List[int]:
    """
    If 'max', returns [hz] (The "Main Horizon").
    If 'auto', returns [1, mid, hz].
    """
    if h_steps_arg == "max":
        return [hz]
    if h_steps_arg == "auto":
        mid = max(1, hz // 2)
        return sorted(list(set([1, mid, hz])))
    
    parts = [p.strip() for p in h_steps_arg.split(",") if p.strip()]
    steps = []
    for p in parts:
        try: steps.append(int(p))
        except: pass
    return sorted(list(set([s for s in steps if 1 <= s <= hz]))) or [hz]


# -----------------------------
# Runners
# -----------------------------
def run_dcenn_plots(out_dir, energy_root, weather_root, start_date, length, h_steps_arg):
    for lb in LOOKBACKS:
        for hz in HORIZONS:
            h_steps = get_h_steps(h_steps_arg, hz)
            # Flatten folder structure if only plotting one step
            flatten_folder = (len(h_steps) == 1)

            print(f"\n[dCeNN PLOTS] LB={lb} H={hz} -> Steps: {h_steps}")

            for task in ["energy", "weather"]:
                root = energy_root if task == "energy" else weather_root
                df_raw, df_true, df_clean = load_dcenn_triplet(root, task, lb, hz)
                if df_raw is None: continue

                sub_raw  = time_slice(df_raw,  start_date, length)
                sub_true = time_slice(df_true, start_date, length)
                sub_clean = time_slice(df_clean, start_date, length) if df_clean is not None else None
                sub_raw, sub_true, sub_clean = align_on_common_index(sub_raw, sub_true, sub_clean)
                if sub_raw is None or len(sub_raw) == 0: continue

                features = choose_features(task, sub_raw)
                
                # Base dirs
                base_raw = out_dir / "DCENN_RAW" / f"LB{lb}_H{hz}" / task
                base_asp = out_dir / "DCENN_ASP" / f"LB{lb}_H{hz}" / task
                
                for hs in h_steps:
                    # If flatten, save directly to task folder. Else create h{s} folder.
                    d_raw = base_raw if flatten_folder else base_raw / f"h{hs}"
                    d_asp = base_asp if flatten_folder else base_asp / f"h{hs}"
                    d_raw.mkdir(parents=True, exist_ok=True)
                    d_asp.mkdir(parents=True, exist_ok=True)

                    for feat in features:
                        plot_dcenn_raw_vs_truth(sub_raw, sub_true, feat, hs,
                            f"dCeNN Raw vs Truth | {task.upper()} | {feat}",
                            d_raw / f"{task}_{feat}_RAW_h{hs}.png")
                        
                        plot_dcenn_raw_clean_truth(sub_raw, sub_true, sub_clean, feat, hs,
                            f"dCeNN + ASP | {task.upper()} | {feat}",
                            d_asp / f"{task}_{feat}_ASP_h{hs}.png")


def run_compare_all(out_dir, energy_root, weather_root, cnn_root, lstm_root, start_date, length, h_steps_arg):
    for lb in LOOKBACKS:
        for hz in HORIZONS:
            h_steps = get_h_steps(h_steps_arg, hz)
            flatten_folder = (len(h_steps) == 1)
            
            print(f"\n[COMPARE ALL] LB={lb} H={hz} -> Steps: {h_steps}")

            for task in ["energy", "weather"]:
                df_cnn  = load_baseline_preds(cnn_root, lb, hz, task)
                df_lstm = load_baseline_preds(lstm_root, lb, hz, task)
                
                root = energy_root if task == "energy" else weather_root
                df_raw, df_true, df_clean = load_dcenn_triplet(root, task, lb, hz)
                if df_raw is None: continue

                sub_raw  = time_slice(df_raw,  start_date, length)
                sub_true = time_slice(df_true, start_date, length)
                sub_clean = time_slice(df_clean, start_date, length) if df_clean is not None else None
                sub_raw, sub_true, sub_clean = align_on_common_index(sub_raw, sub_true, sub_clean)
                if sub_true is None or len(sub_true) == 0: continue

                features = choose_features(task, sub_raw)
                
                sub_cnn = time_slice(df_cnn, start_date, length) if df_cnn is not None else None
                sub_lstm = time_slice(df_lstm, start_date, length) if df_lstm is not None else None

                base_dest = out_dir / "COMPARE_ALL" / f"LB{lb}_H{hz}" / task
                
                for hs in h_steps:
                    dest = base_dest if flatten_folder else base_dest / f"h{hs}"
                    dest.mkdir(parents=True, exist_ok=True)

                    for feat in features:
                        plot_all_models_vs_truth(sub_true, sub_raw, sub_clean, sub_cnn, sub_lstm, feat, hs,
                            f"Truth vs CNN/LSTM vs dCeNN | {task.upper()} | {feat}",
                            dest / f"{task}_{feat}_ALL_h{hs}.png")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="thesis_plots")
    ap.add_argument("--dcenn_energy_root", type=str, default="outputs_energy_full")
    ap.add_argument("--dcenn_weather_root", type=str, default="outputs_weather_full")
    ap.add_argument("--cnn_root", type=str, default="artifacts_cnn_baseline")
    ap.add_argument("--lstm_root", type=str, default="artifacts_lstm_baseline")
    ap.add_argument("--start_date", type=str, default=None, help="2022-01-10")
    ap.add_argument("--length", type=int, default=168)
    
    # Updated default to "max" to just get the main horizon
    ap.add_argument("--h_steps", type=str, default="max", help='"max" (default), "auto", or "1,12"')
    
    ap.add_argument("--dcenn_only", action="store_true")
    ap.add_argument("--compare_all", action="store_true")

    args = ap.parse_args()
    
    if not args.dcenn_only and not args.compare_all:
        args.dcenn_only = True

    if args.dcenn_only:
        run_dcenn_plots(Path(args.out_dir), Path(args.dcenn_energy_root), Path(args.dcenn_weather_root), 
                        args.start_date, args.length, args.h_steps)
    
    if args.compare_all:
        run_compare_all(Path(args.out_dir), Path(args.dcenn_energy_root), Path(args.dcenn_weather_root), 
                        Path(args.cnn_root), Path(args.lstm_root), 
                        args.start_date, args.length, args.h_steps)
    
    print(f"\n[DONE] Plots saved to {args.out_dir}")

if __name__ == "__main__":
    main()