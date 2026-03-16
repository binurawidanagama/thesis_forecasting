#!/usr/bin/env python3
"""
Evaluate forecasting models directly from saved parquet outputs.

Supports:
  - CNN baseline parquet outputs
  - LSTM baseline parquet outputs
  - dCeNN+ELM raw parquet outputs
  - dCeNN+ELM+ASP clean parquet outputs

What this script computes:
  1) overall metrics per model/run
  2) per-target metrics
  3) per-horizon-step metrics
  4) per-target-per-horizon-step metrics
  5) rankings (best/worst by RMSE / MAE / sMAPE)

Important note about persistence / RMSE_ratio:
  - Raw parquets contain predictions and truth.
  - They DO NOT contain the persistence baseline forecasts.
  - Therefore, per-target or per-horizon RMSE_ratio cannot be reconstructed from parquet alone.
  - If your summary CSVs exist, this script will also collect overall BASE_* metrics and compute:
        RMSE_ratio = RMSE / BASE_RMSE
        MAE_ratio  = MAE  / BASE_MAE
        RMSE_gain  = 1 - RMSE_ratio

Usage:
  python scripts/evaluate_all_from_parquets.py
  python scripts/evaluate_all_from_parquets.py --out_dir parquet_eval
  python scripts/evaluate_all_from_parquets.py \
      --cnn_root artifacts_cnn_baseline \
      --lstm_root artifacts_lstm_baseline \
      --dcenn_energy_root outputs_energy_full \
      --dcenn_weather_root outputs_weather_full
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12
BASELINE_FILE_RE = re.compile(r"LB(?P<lookback>\d+)_H(?P<horizon>\d+)_(?P<task>[A-Za-z0-9_\-]+)$")
STEP_COL_RE = re.compile(r"^(?P<name>.+)\+h(?P<step>\d+)$")


@dataclass(frozen=True)
class RunKey:
    model: str
    variant: str
    task: str
    lookback: int
    horizon: int


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    v = mse(y_true, y_pred)
    return float(math.sqrt(v)) if np.isfinite(v) else float("nan")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_pred - y_true))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= EPS:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def nrmse_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    scale = float(np.mean(np.abs(y_true)))
    if scale <= EPS:
        return float("nan")
    return float(rmse(y_true, y_pred) / scale)


def _clean_pair(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def compute_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt, yp = _clean_pair(y_true, y_pred)
    return {
        "n": int(yt.size),
        "MAE": mae(yt, yp),
        "MSE": mse(yt, yp),
        "RMSE": rmse(yt, yp),
        "sMAPE": smape(yt, yp),
        "MAPE": mape(yt, yp),
        "Bias": bias(yt, yp),
        "R2": r2(yt, yp),
        "NRMSE_mean": nrmse_mean(yt, yp),
    }


# -----------------------------------------------------------------------------
# Column discovery / normalization
# -----------------------------------------------------------------------------
def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    else:
        out["timestamp"] = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def extract_step_columns(df: pd.DataFrame, prefix: str = "") -> Dict[Tuple[str, int], str]:
    """
    Returns mapping: (target_name, step) -> column_name.

    Examples:
      baseline columns: Pred_load_mw+h12, True_load_mw+h12
      dcenn columns:    load_mw+h12
    """
    mapping: Dict[Tuple[str, int], str] = {}
    cols = list(df.columns)

    for col in cols:
        if col == "timestamp":
            continue

        work = col
        if prefix:
            if not work.startswith(prefix):
                continue
            work = work[len(prefix):]

        m = STEP_COL_RE.match(work)
        if m:
            target = m.group("name")
            step = int(m.group("step"))
            mapping[(target, step)] = col

    # Fallback for backward-compatible h1 aliases such as Pred_load_mw / True_load_mw
    if prefix:
        for col in cols:
            if col == "timestamp":
                continue
            if not col.startswith(prefix):
                continue
            work = col[len(prefix):]
            if STEP_COL_RE.match(work):
                continue
            mapping.setdefault((work, 1), col)
    else:
        for col in cols:
            if col == "timestamp":
                continue
            if STEP_COL_RE.match(col):
                continue
            # Avoid grabbing prefixed baseline columns when parsing dcenn files.
            if col.startswith("Pred_") or col.startswith("True_"):
                continue
            mapping.setdefault((col, 1), col)

    return mapping


# -----------------------------------------------------------------------------
# Reading baseline parquet outputs (CNN/LSTM)
# -----------------------------------------------------------------------------
def parse_baseline_run_from_folder(folder: Path) -> Optional[Tuple[int, int, str]]:
    m = BASELINE_FILE_RE.match(folder.name)
    if not m:
        return None
    return int(m.group("lookback")), int(m.group("horizon")), str(m.group("task")).upper()


def read_baseline_parquet(pred_path: Path, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = pred_path.parent
    parsed = parse_baseline_run_from_folder(folder)
    if parsed is None:
        raise ValueError(f"Could not parse baseline run folder name: {folder}")

    lookback, horizon, task = parsed
    df = ensure_timestamp(pd.read_parquet(pred_path))

    pred_cols = extract_step_columns(df, prefix="Pred_")
    true_cols = extract_step_columns(df, prefix="True_")
    keys = sorted(set(pred_cols).intersection(true_cols), key=lambda x: (x[1], x[0]))

    if not keys:
        raise ValueError(f"No matching True_/Pred_ horizon columns found in {pred_path}")

    key = RunKey(model=model_name, variant="baseline", task=task, lookback=lookback, horizon=horizon)
    return build_all_metric_tables_from_pairs(df, key, keys, true_cols, pred_cols)


# -----------------------------------------------------------------------------
# Reading dCeNN parquet outputs
# -----------------------------------------------------------------------------
def read_dcenn_parquet_pair(
    truth_path: Path,
    pred_path: Path,
    model_name: str,
    variant: str,
    task: str,
    lookback: int,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth_df = ensure_timestamp(pd.read_parquet(truth_path))
    pred_df = ensure_timestamp(pd.read_parquet(pred_path))

    truth_cols = extract_step_columns(truth_df, prefix="")
    pred_cols = extract_step_columns(pred_df, prefix="")
    keys = sorted(set(pred_cols).intersection(truth_cols), key=lambda x: (x[1], x[0]))

    if not keys:
        raise ValueError(f"No matching truth/pred horizon columns found: truth={truth_path}, pred={pred_path}")

    key = RunKey(model=model_name, variant=variant, task=task.upper(), lookback=lookback, horizon=horizon)
    return build_all_metric_tables_from_pairs(truth_df, key, keys, truth_cols, pred_cols, pred_df=pred_df)


# -----------------------------------------------------------------------------
# Shared evaluation core
# -----------------------------------------------------------------------------
def build_all_metric_tables_from_pairs(
    truth_df: pd.DataFrame,
    run_key: RunKey,
    keys: List[Tuple[str, int]],
    truth_col_map: Dict[Tuple[str, int], str],
    pred_col_map: Dict[Tuple[str, int], str],
    pred_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns four DataFrames:
      overall_df, by_target_df, by_horizon_df, by_target_horizon_df
    """
    if pred_df is None:
        pred_df = truth_df

    pair_rows: List[Dict[str, object]] = []
    overall_true: List[np.ndarray] = []
    overall_pred: List[np.ndarray] = []

    for target, step in keys:
        tcol = truth_col_map.get((target, step))
        pcol = pred_col_map.get((target, step))
        if tcol is None or pcol is None:
            continue

        a = truth_df[["timestamp", tcol]].rename(columns={tcol: "y_true"})
        b = pred_df[["timestamp", pcol]].rename(columns={pcol: "y_pred"})
        merged = a.merge(b, on="timestamp", how="inner")
        yt, yp = _clean_pair(merged["y_true"].to_numpy(), merged["y_pred"].to_numpy())
        if yt.size == 0:
            continue

        overall_true.append(yt)
        overall_pred.append(yp)

        row = {
            "model": run_key.model,
            "variant": run_key.variant,
            "task": run_key.task,
            "lookback": int(run_key.lookback),
            "horizon": int(run_key.horizon),
            "target": str(target),
            "horizon_step": int(step),
        }
        row.update(compute_metric_dict(yt, yp))
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    overall_df = pd.DataFrame([
        {
            "model": run_key.model,
            "variant": run_key.variant,
            "task": run_key.task,
            "lookback": int(run_key.lookback),
            "horizon": int(run_key.horizon),
            **compute_metric_dict(np.concatenate(overall_true), np.concatenate(overall_pred)),
        }
    ])

    by_target_rows: List[Dict[str, object]] = []
    for target, g in pair_df.groupby("target", sort=True):
        yt_list: List[np.ndarray] = []
        yp_list: List[np.ndarray] = []
        for _, row in g.iterrows():
            step = int(row["horizon_step"])
            tcol = truth_col_map[(target, step)]
            pcol = pred_col_map[(target, step)]
            a = truth_df[["timestamp", tcol]].rename(columns={tcol: "y_true"})
            b = pred_df[["timestamp", pcol]].rename(columns={pcol: "y_pred"})
            merged = a.merge(b, on="timestamp", how="inner")
            yt, yp = _clean_pair(merged["y_true"].to_numpy(), merged["y_pred"].to_numpy())
            if yt.size:
                yt_list.append(yt)
                yp_list.append(yp)
        if yt_list:
            row_out = {
                "model": run_key.model,
                "variant": run_key.variant,
                "task": run_key.task,
                "lookback": int(run_key.lookback),
                "horizon": int(run_key.horizon),
                "target": str(target),
            }
            row_out.update(compute_metric_dict(np.concatenate(yt_list), np.concatenate(yp_list)))
            by_target_rows.append(row_out)

    by_horizon_rows: List[Dict[str, object]] = []
    for step, g in pair_df.groupby("horizon_step", sort=True):
        yt_list = []
        yp_list = []
        for _, row in g.iterrows():
            target = str(row["target"])
            tcol = truth_col_map[(target, int(step))]
            pcol = pred_col_map[(target, int(step))]
            a = truth_df[["timestamp", tcol]].rename(columns={tcol: "y_true"})
            b = pred_df[["timestamp", pcol]].rename(columns={pcol: "y_pred"})
            merged = a.merge(b, on="timestamp", how="inner")
            yt, yp = _clean_pair(merged["y_true"].to_numpy(), merged["y_pred"].to_numpy())
            if yt.size:
                yt_list.append(yt)
                yp_list.append(yp)
        if yt_list:
            row_out = {
                "model": run_key.model,
                "variant": run_key.variant,
                "task": run_key.task,
                "lookback": int(run_key.lookback),
                "horizon": int(run_key.horizon),
                "horizon_step": int(step),
            }
            row_out.update(compute_metric_dict(np.concatenate(yt_list), np.concatenate(yp_list)))
            by_horizon_rows.append(row_out)

    by_target_df = pd.DataFrame(by_target_rows)
    by_horizon_df = pd.DataFrame(by_horizon_rows)
    return overall_df, by_target_df, by_horizon_df, pair_df


# -----------------------------------------------------------------------------
# Collect all runs
# -----------------------------------------------------------------------------
def iter_baseline_pred_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("LB*_H*_*/preds.parquet"))


def iter_dcenn_runs(root: Path, task: str) -> Iterable[Tuple[int, int, Path, Path, Optional[Path]]]:
    if not root.exists():
        return []
    rows = []
    for run_dir in sorted(root.glob("LB*_H*")):
        m = re.match(r"LB(?P<lb>\d+)_H(?P<hz>\d+)$", run_dir.name)
        if not m:
            continue
        lb = int(m.group("lb"))
        hz = int(m.group("hz"))
        truth = run_dir / f"truth_{task}.parquet"
        raw = run_dir / f"raw_{task}.parquet"
        clean = run_dir / f"clean_{task}.parquet"
        if truth.exists() and raw.exists():
            rows.append((lb, hz, truth, raw, clean if clean.exists() else None))
    return rows


def evaluate_all(
    cnn_root: Path,
    lstm_root: Path,
    dcenn_energy_root: Path,
    dcenn_weather_root: Path,
) -> Dict[str, pd.DataFrame]:
    overall_parts: List[pd.DataFrame] = []
    target_parts: List[pd.DataFrame] = []
    horizon_parts: List[pd.DataFrame] = []
    pair_parts: List[pd.DataFrame] = []

    # Baselines
    for pred_path in iter_baseline_pred_files(cnn_root):
        o, t, h, p = read_baseline_parquet(pred_path, model_name="CNN")
        if not o.empty:
            overall_parts.append(o)
            target_parts.append(t)
            horizon_parts.append(h)
            pair_parts.append(p)

    for pred_path in iter_baseline_pred_files(lstm_root):
        o, t, h, p = read_baseline_parquet(pred_path, model_name="LSTM")
        if not o.empty:
            overall_parts.append(o)
            target_parts.append(t)
            horizon_parts.append(h)
            pair_parts.append(p)

    # dCeNN energy
    for lb, hz, truth, raw, clean in iter_dcenn_runs(dcenn_energy_root, task="energy"):
        o, t, h, p = read_dcenn_parquet_pair(
            truth_path=truth,
            pred_path=raw,
            model_name="dCeNN+ELM",
            variant="raw",
            task="ENERGY",
            lookback=lb,
            horizon=hz,
        )
        if not o.empty:
            overall_parts.append(o)
            target_parts.append(t)
            horizon_parts.append(h)
            pair_parts.append(p)

        if clean is not None:
            o, t, h, p = read_dcenn_parquet_pair(
                truth_path=truth,
                pred_path=clean,
                model_name="dCeNN+ELM+ASP",
                variant="asp",
                task="ENERGY",
                lookback=lb,
                horizon=hz,
            )
            if not o.empty:
                overall_parts.append(o)
                target_parts.append(t)
                horizon_parts.append(h)
                pair_parts.append(p)

    # dCeNN weather
    for lb, hz, truth, raw, clean in iter_dcenn_runs(dcenn_weather_root, task="weather"):
        o, t, h, p = read_dcenn_parquet_pair(
            truth_path=truth,
            pred_path=raw,
            model_name="dCeNN+ELM",
            variant="raw",
            task="WEATHER",
            lookback=lb,
            horizon=hz,
        )
        if not o.empty:
            overall_parts.append(o)
            target_parts.append(t)
            horizon_parts.append(h)
            pair_parts.append(p)

        if clean is not None:
            o, t, h, p = read_dcenn_parquet_pair(
                truth_path=truth,
                pred_path=clean,
                model_name="dCeNN+ELM+ASP",
                variant="asp",
                task="WEATHER",
                lookback=lb,
                horizon=hz,
            )
            if not o.empty:
                overall_parts.append(o)
                target_parts.append(t)
                horizon_parts.append(h)
                pair_parts.append(p)

    return {
        "overall": _concat_or_empty(overall_parts),
        "by_target": _concat_or_empty(target_parts),
        "by_horizon": _concat_or_empty(horizon_parts),
        "by_target_horizon": _concat_or_empty(pair_parts),
    }


# -----------------------------------------------------------------------------
# Summary CSVs (for BASE_* and ratios)
# -----------------------------------------------------------------------------
def normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Train_Avg_CPU_Pct": "Avg_CPU_Usage_Pct",
        "Train_Peak_RSS_MB": "Peak_RAM_MB",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if "task" in df.columns:
        df["task"] = df["task"].astype(str).str.upper()
    for c in ["lookback", "horizon", "MAE", "RMSE", "sMAPE", "BASE_MAE", "BASE_RMSE", "BASE_sMAPE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_summary_csvs(
    cnn_root: Path,
    lstm_root: Path,
    dcenn_energy_root: Path,
    dcenn_weather_root: Path,
) -> pd.DataFrame:
    paths: List[Path] = []
    paths.extend(sorted(cnn_root.glob("summary_lb*.csv")) if cnn_root.exists() else [])
    paths.extend(sorted(lstm_root.glob("summary_lb*.csv")) if lstm_root.exists() else [])
    paths.extend(sorted(dcenn_energy_root.glob("summary_dcenn_*.csv")) if dcenn_energy_root.exists() else [])
    paths.extend(sorted(dcenn_weather_root.glob("summary_dcenn_*.csv")) if dcenn_weather_root.exists() else [])

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = normalize_summary_df(pd.read_csv(p))
        model, variant = tag_summary_path(p)
        df["model"] = model
        df["variant"] = variant
        df["source_file"] = str(p)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["RMSE_ratio"] = out["RMSE"] / (out["BASE_RMSE"] + EPS)
    out["MAE_ratio"] = out["MAE"] / (out["BASE_MAE"] + EPS)
    out["RMSE_gain"] = 1.0 - out["RMSE_ratio"]
    out = out.sort_values("source_file").drop_duplicates(
        subset=["model", "variant", "task", "lookback", "horizon"], keep="last"
    )
    return out


def tag_summary_path(path: Path) -> Tuple[str, str]:
    p = str(path).lower()
    name = path.name.lower()
    if "artifacts_cnn_baseline" in p:
        return "CNN", "baseline"
    if "artifacts_lstm_baseline" in p:
        return "LSTM", "baseline"
    if name.startswith("summary_dcenn_"):
        if "asp" in name:
            return "dCeNN+ELM+ASP", "asp"
        return "dCeNN+ELM", "raw"
    return "UNKNOWN", "unknown"


# -----------------------------------------------------------------------------
# Rankings / helpers
# -----------------------------------------------------------------------------
def add_rankings(df: pd.DataFrame, group_cols: List[str], metric: str, ascending: bool = True) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out[f"rank_{metric}"] = out.groupby(group_cols)[metric].rank(method="dense", ascending=ascending)
    return out


def build_best_worst_tables(df: pd.DataFrame, group_cols: List[str], metric: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    sort_df = df.sort_values(group_cols + [metric, "model", "variant"]).reset_index(drop=True)
    best = sort_df.groupby(group_cols, as_index=False).first()
    worst = sort_df.groupby(group_cols, as_index=False).last()
    return best, worst


def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all forecasting models from saved parquet outputs.")
    ap.add_argument("--cnn_root", type=str, default="artifacts_cnn_baseline")
    ap.add_argument("--lstm_root", type=str, default="artifacts_lstm_baseline")
    ap.add_argument("--dcenn_energy_root", type=str, default="outputs_energy_full")
    ap.add_argument("--dcenn_weather_root", type=str, default="outputs_weather_full")
    ap.add_argument("--out_dir", type=str, default="parquet_eval")
    args = ap.parse_args()

    cnn_root = Path(args.cnn_root)
    lstm_root = Path(args.lstm_root)
    dcenn_energy_root = Path(args.dcenn_energy_root)
    dcenn_weather_root = Path(args.dcenn_weather_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = evaluate_all(
        cnn_root=cnn_root,
        lstm_root=lstm_root,
        dcenn_energy_root=dcenn_energy_root,
        dcenn_weather_root=dcenn_weather_root,
    )

    overall = tables["overall"].copy()
    by_target = tables["by_target"].copy()
    by_horizon = tables["by_horizon"].copy()
    by_target_horizon = tables["by_target_horizon"].copy()

    if not overall.empty:
        overall = overall.sort_values(["task", "lookback", "horizon", "RMSE", "model", "variant"])
    if not by_target.empty:
        by_target = by_target.sort_values(["task", "lookback", "horizon", "target", "RMSE", "model", "variant"])
    if not by_horizon.empty:
        by_horizon = by_horizon.sort_values(["task", "lookback", "horizon", "horizon_step", "RMSE", "model", "variant"])
    if not by_target_horizon.empty:
        by_target_horizon = by_target_horizon.sort_values([
            "task", "lookback", "horizon", "target", "horizon_step", "RMSE", "model", "variant"
        ])

    # Rankings
    overall = add_rankings(overall, ["task", "lookback", "horizon"], "RMSE", ascending=True)
    overall = add_rankings(overall, ["task", "lookback", "horizon"], "MAE", ascending=True)
    overall = add_rankings(overall, ["task", "lookback", "horizon"], "sMAPE", ascending=True)

    by_target = add_rankings(by_target, ["task", "lookback", "horizon", "target"], "RMSE", ascending=True)
    by_target = add_rankings(by_target, ["task", "lookback", "horizon", "target"], "MAE", ascending=True)
    by_target = add_rankings(by_target, ["task", "lookback", "horizon", "target"], "sMAPE", ascending=True)

    by_horizon = add_rankings(by_horizon, ["task", "lookback", "horizon", "horizon_step"], "RMSE", ascending=True)
    by_horizon = add_rankings(by_horizon, ["task", "lookback", "horizon", "horizon_step"], "MAE", ascending=True)
    by_horizon = add_rankings(by_horizon, ["task", "lookback", "horizon", "horizon_step"], "sMAPE", ascending=True)

    by_target_horizon = add_rankings(
        by_target_horizon,
        ["task", "lookback", "horizon", "target", "horizon_step"],
        "RMSE",
        ascending=True,
    )
    by_target_horizon = add_rankings(
        by_target_horizon,
        ["task", "lookback", "horizon", "target", "horizon_step"],
        "MAE",
        ascending=True,
    )
    by_target_horizon = add_rankings(
        by_target_horizon,
        ["task", "lookback", "horizon", "target", "horizon_step"],
        "sMAPE",
        ascending=True,
    )

    # Write parquet-derived metrics
    overall.to_csv(out_dir / "metrics_overall_from_parquets.csv", index=False)
    by_target.to_csv(out_dir / "metrics_by_target_from_parquets.csv", index=False)
    by_horizon.to_csv(out_dir / "metrics_by_horizon_step_from_parquets.csv", index=False)
    by_target_horizon.to_csv(out_dir / "metrics_by_target_horizon_from_parquets.csv", index=False)

    best_overall, worst_overall = build_best_worst_tables(overall, ["task", "lookback", "horizon"], "RMSE")
    best_target, worst_target = build_best_worst_tables(by_target, ["task", "lookback", "horizon", "target"], "RMSE")
    best_step, worst_step = build_best_worst_tables(by_horizon, ["task", "lookback", "horizon", "horizon_step"], "RMSE")
    best_pair, worst_pair = build_best_worst_tables(
        by_target_horizon, ["task", "lookback", "horizon", "target", "horizon_step"], "RMSE"
    )

    best_overall.to_csv(out_dir / "best_models_overall_by_rmse.csv", index=False)
    worst_overall.to_csv(out_dir / "worst_models_overall_by_rmse.csv", index=False)
    best_target.to_csv(out_dir / "best_models_by_target_rmse.csv", index=False)
    worst_target.to_csv(out_dir / "worst_models_by_target_rmse.csv", index=False)
    best_step.to_csv(out_dir / "best_models_by_horizon_step_rmse.csv", index=False)
    worst_step.to_csv(out_dir / "worst_models_by_horizon_step_rmse.csv", index=False)
    best_pair.to_csv(out_dir / "best_models_by_target_horizon_rmse.csv", index=False)
    worst_pair.to_csv(out_dir / "worst_models_by_target_horizon_rmse.csv", index=False)

    # Write summary-derived ratios if available
    summary_df = read_summary_csvs(
        cnn_root=cnn_root,
        lstm_root=lstm_root,
        dcenn_energy_root=dcenn_energy_root,
        dcenn_weather_root=dcenn_weather_root,
    )
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["task", "lookback", "horizon", "RMSE_ratio", "model", "variant"])
        summary_df.to_csv(out_dir / "summary_overall_with_persistence_ratios.csv", index=False)

        best_ratio, worst_ratio = build_best_worst_tables(summary_df, ["task", "lookback", "horizon"], "RMSE_ratio")
        best_ratio.to_csv(out_dir / "best_models_overall_by_rmse_ratio.csv", index=False)
        worst_ratio.to_csv(out_dir / "worst_models_overall_by_rmse_ratio.csv", index=False)

    # Console summary
    print("\n[OK] Saved parquet-derived evaluation tables to:", out_dir)
    print("  - metrics_overall_from_parquets.csv")
    print("  - metrics_by_target_from_parquets.csv")
    print("  - metrics_by_horizon_step_from_parquets.csv")
    print("  - metrics_by_target_horizon_from_parquets.csv")
    print("  - best/worst model ranking CSVs")
    if not summary_df.empty:
        print("  - summary_overall_with_persistence_ratios.csv")
        print("  - best_models_overall_by_rmse_ratio.csv")
    else:
        print("[NOTE] No summary CSVs found, so BASE_* / RMSE_ratio tables were not produced.")
        print("[NOTE] That is normal if only parquets exist. Raw metrics vs truth are still fully computed.")

    if not overall.empty:
        print("\n[Quick view] Best overall by RMSE per task/lookback/horizon:")
        cols = ["task", "lookback", "horizon", "model", "variant", "RMSE", "MAE", "sMAPE"]
        print(best_overall[cols].to_string(index=False))


if __name__ == "__main__":
    main()
