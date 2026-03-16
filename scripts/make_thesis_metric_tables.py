#!/usr/bin/env python3
"""
Create thesis-ready table figures and exportable tables from the CSV outputs
produced by evaluate_all_from_parquets.py.

Reads from an evaluation directory (default: parquet_eval):
  - metrics_overall_from_parquets.csv
  - metrics_by_target_from_parquets.csv
  - metrics_by_horizon_step_from_parquets.csv
  - metrics_by_target_horizon_from_parquets.csv
  - summary_overall_with_persistence_ratios.csv   (optional)
  - best/worst ranking CSVs                        (optional)

Creates clear table artifacts that can be inserted directly into a thesis:
  1) Overall per-run model comparison tables (rows = models)
  2) Fixed-horizon lookback comparison tables (rows = lookbacks, cols = models)
  3) Fixed-lookback horizon comparison tables (rows = horizons, cols = models)
  4) Per-target model comparison tables
  5) Per-step model comparison tables
  6) Persistence-ratio tables (if available)
  7) Best-model summary tables

Exports each table in three forms:
  - image table (.png / .pdf / other matplotlib-supported formats)
  - CSV (.csv)
  - LaTeX (.tex)

Usage:
  python scripts/make_thesis_metric_tables.py
  python scripts/make_thesis_metric_tables.py --eval_dir parquet_eval --out_dir thesis_tables
  python scripts/make_thesis_metric_tables.py --formats png pdf --dpi 300
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Global styling
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
    }
)

LOWER_IS_BETTER = {"RMSE", "MAE", "sMAPE", "MSE", "MAPE", "NRMSE_mean", "RMSE_ratio", "MAE_ratio"}
HIGHER_IS_BETTER = {"R2", "RMSE_gain"}

PRIMARY_METRICS = ["RMSE", "MAE", "sMAPE"]
SECONDARY_METRICS = ["R2", "Bias", "NRMSE_mean"]
RATIO_METRICS = ["RMSE_ratio", "MAE_ratio", "RMSE_gain"]

MODEL_ORDER = ["CNN", "LSTM", "dCeNN+ELM", "dCeNN+ELM+ASP"]
MODEL_COLORS = {
    "CNN": "#4E79A7",
    "LSTM": "#F28E2B",
    "dCeNN+ELM": "#59A14F",
    "dCeNN+ELM+ASP": "#E15759",
    "UNKNOWN": "#9C9C9C",
}

TARGET_NAME_MAP = {
    "load_mw": "Load",
    "temperature_2m_c": "Temperature",
    "mean_global_radiation": "Global radiation",
    "precipitation_mm": "Precipitation",
    "mean_wind_speed": "Wind speed",
}

COLUMN_LABELS = {
    "model_label": "Model",
    "lookback": "Lookback",
    "horizon": "Forecast horizon",
    "horizon_step": "Forecast step",
    "target_label": "Target variable",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "sMAPE": "sMAPE (%)",
    "R2": "R²",
    "Bias": "Bias",
    "NRMSE_mean": "NRMSE",
    "RMSE_ratio": "RMSE / persistence",
    "MAE_ratio": "MAE / persistence",
    "RMSE_gain": "RMSE gain",
}

METRIC_EXPLANATIONS = {
    "RMSE": "Lower is better. Large mistakes are penalized more strongly than in MAE.",
    "MAE": "Lower is better. This is the average absolute difference between prediction and truth.",
    "sMAPE": "Lower is better. This is a percentage-style error that is easier to compare across scales.",
    "R2": "Higher is better. Values closer to 1 mean the predictions follow the true pattern more closely.",
    "Bias": "Closer to 0 is better. Negative means under-prediction on average; positive means over-prediction.",
    "NRMSE_mean": "Lower is better. RMSE scaled by the average size of the true values.",
    "RMSE_ratio": "Lower is better. Values below 1.0 mean the model beats the persistence baseline.",
    "MAE_ratio": "Lower is better. Values below 1.0 mean the model beats the persistence baseline.",
    "RMSE_gain": "Higher is better. Positive values mean improvement over persistence.",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "table"


def wrap_text(text: str, width: int = 110) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def pretty_target(name: str) -> str:
    key = str(name).strip().lower()
    return TARGET_NAME_MAP.get(key, str(name).replace("_", " ").replace("/", " ").title())


def model_display(row: pd.Series) -> str:
    model = str(row.get("model", "UNKNOWN"))
    if model in MODEL_ORDER:
        return model
    return model


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def order_models(labels: Iterable[str]) -> List[str]:
    labels = list(dict.fromkeys([str(x) for x in labels]))
    known = [m for m in MODEL_ORDER if m in labels]
    unknown = sorted([m for m in labels if m not in known])
    return known + unknown


def task_note(task: str, target_count: Optional[int] = None) -> List[str]:
    notes: List[str] = []
    if str(task).upper() == "WEATHER":
        notes.append(
            "Weather combines multiple physical variables. Aggregate RMSE or MAE should therefore be read mainly as a model-comparison tool, not as a single physical-unit error."
        )
    if target_count is not None and target_count > 1:
        notes.append(
            f"This table covers {target_count} target variables. Per-target tables are the clearest place to explain which variable is easy or hard to predict."
        )
    return notes


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "task" in out.columns:
        out["task"] = out["task"].astype(str).str.upper()
    if "target" in out.columns:
        out["target_label"] = out["target"].map(pretty_target)
    out["model_label"] = out.apply(model_display, axis=1)
    for c in [
        "lookback", "horizon", "horizon_step", "RMSE", "MAE", "sMAPE", "R2", "Bias", "NRMSE_mean",
        "RMSE_ratio", "MAE_ratio", "RMSE_gain"
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [COLUMN_LABELS.get(c, c) for c in out.columns]
    return out


def format_value(val: object, decimals: int = 3) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if math.isfinite(float(val)):
            return f"{float(val):.{decimals}f}"
        return ""
    return str(val)


def df_to_latex(df: pd.DataFrame, caption: str) -> str:
    try:
        return df.to_latex(index=False, escape=True, caption=caption)
    except Exception:
        return df.to_string(index=False)


def save_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def compute_best_mask(df: pd.DataFrame, columns: List[str]) -> Dict[Tuple[int, int], bool]:
    mask: Dict[Tuple[int, int], bool] = {}
    for col_idx, col in enumerate(df.columns):
        if col not in columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        finite = s[np.isfinite(s)]
        if finite.empty:
            continue
        if col in LOWER_IS_BETTER:
            best = float(finite.min())
            rows = np.where(np.isclose(s.to_numpy(dtype=float), best, equal_nan=False))[0]
        elif col in HIGHER_IS_BETTER:
            best = float(finite.max())
            rows = np.where(np.isclose(s.to_numpy(dtype=float), best, equal_nan=False))[0]
        else:
            continue
        for r in rows:
            mask[(int(r), int(col_idx))] = True
    return mask


def render_table_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    notes: List[str],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    highlight_metric_cols: Optional[List[str]] = None,
) -> None:
    ensure_dir(out_base.parent)

    disp = pretty_columns(df)
    raw_cols = list(df.columns)
    highlight_metric_cols = highlight_metric_cols or []
    best_mask = compute_best_mask(df, highlight_metric_cols)

    n_rows, n_cols = disp.shape
    col_width = max(1.1, min(2.8, 14.0 / max(n_cols, 1)))
    fig_w = min(18.0, max(8.5, n_cols * col_width + 1.0))
    fig_h = min(18.0, max(2.8, 1.7 + 0.42 * (n_rows + 2)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")

    cell_text = [[format_value(v, decimals=3) for v in row] for row in disp.values]
    table = ax.table(
        cellText=cell_text,
        colLabels=list(disp.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    # Header styling
    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_text_props(weight="bold", color="black")
        cell.set_facecolor("#D9E2F3")
        cell.set_edgecolor("#6B6B6B")

    # Body styling + metric highlighting
    for r in range(1, n_rows + 1):
        for c in range(n_cols):
            cell = table[(r, c)]
            cell.set_edgecolor("#B5B5B5")
            cell.set_facecolor("#FFFFFF" if r % 2 == 1 else "#F8F8F8")
            if raw_cols[c] == "model_label":
                model_name = str(df.iloc[r - 1, c])
                cell.set_facecolor(MODEL_COLORS.get(model_name, MODEL_COLORS["UNKNOWN"]))
                cell.set_text_props(color="white", weight="bold")
            if (r - 1, c) in best_mask:
                cell.set_facecolor("#D9F2D9")
                cell.set_text_props(weight="bold")

    title_text = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(title_text, fontsize=13, fontweight="bold", y=0.98)

    if notes:
        note_text = "\n".join([wrap_text(f"• {x}", 140) for x in notes if x])
        fig.text(0.01, 0.015, note_text, ha="left", va="bottom", fontsize=9, color="#404040")
        plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    for ext in formats:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def export_table_bundle(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    notes: List[str],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    highlight_metric_cols: Optional[List[str]] = None,
) -> None:
    ensure_dir(out_base.parent)
    df.to_csv(out_base.with_suffix(".csv"), index=False)
    save_text(out_base.with_suffix(".tex"), df_to_latex(pretty_columns(df), caption=title))
    render_table_figure(df, title, subtitle, notes, out_base, formats, dpi, highlight_metric_cols)


# -----------------------------------------------------------------------------
# Table generators
# -----------------------------------------------------------------------------
def grouped_runs(df: pd.DataFrame) -> List[Tuple[str, int, int]]:
    if df.empty:
        return []
    keys = df[["task", "lookback", "horizon"]].drop_duplicates().sort_values(["task", "lookback", "horizon"])
    return [(str(r.task), int(r.lookback), int(r.horizon)) for r in keys.itertuples(index=False)]


def subset_run(df: pd.DataFrame, task: str, lookback: int, horizon: int) -> pd.DataFrame:
    mask = (
        (df["task"].astype(str) == str(task))
        & (df["lookback"].astype(int) == int(lookback))
        & (df["horizon"].astype(int) == int(horizon))
    )
    return df.loc[mask].copy()


def create_overall_tables(overall: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    count = 0
    for task, lookback, horizon in grouped_runs(overall):
        sub = subset_run(overall, task, lookback, horizon)
        if sub.empty:
            continue
        order = order_models(sub["model_label"].tolist())
        sub["model_label"] = pd.Categorical(sub["model_label"], categories=order, ordered=True)
        sub = sub.sort_values("model_label")
        keep = ["model_label", "RMSE", "MAE", "sMAPE", "R2", "Bias", "NRMSE_mean"]
        present = [c for c in keep if c in sub.columns]
        table_df = sub[present].reset_index(drop=True)
        target_count = None
        notes = [
            "This table compares all models for one fixed experimental setting.",
            "For RMSE, MAE, sMAPE, and NRMSE, smaller values are better. For R², larger values are better. For Bias, values closer to zero are better.",
            *task_note(task, target_count),
        ]
        export_table_bundle(
            table_df,
            title=f"Overall model comparison for {task}",
            subtitle=f"Lookback = {lookback} steps, Forecast horizon = {horizon} steps",
            notes=notes,
            out_base=out_dir / "overall_tables" / safe_name(f"overall_{task}_LB{lookback}_H{horizon}"),
            formats=formats,
            dpi=dpi,
            highlight_metric_cols=[c for c in ["RMSE", "MAE", "sMAPE", "R2", "NRMSE_mean"] if c in table_df.columns],
        )
        count += 1
    return count


def create_fixed_horizon_tables(overall: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    if overall.empty:
        return 0
    count = 0
    for task in sorted(overall["task"].dropna().astype(str).unique()):
        task_df = overall.loc[overall["task"] == task].copy()
        for horizon in sorted(task_df["horizon"].dropna().astype(int).unique()):
            sub = task_df.loc[task_df["horizon"].astype(int) == horizon].copy()
            for metric in PRIMARY_METRICS + ["R2"]:
                if metric not in sub.columns:
                    continue
                pivot = sub.pivot_table(index="lookback", columns="model_label", values=metric, aggfunc="first")
                if pivot.empty:
                    continue
                pivot = pivot.reindex(columns=order_models(pivot.columns), fill_value=np.nan).reset_index()
                notes = [
                    f"This table keeps the forecast horizon fixed at {horizon} steps and shows how model error changes when the lookback window changes.",
                    METRIC_EXPLANATIONS.get(metric, ""),
                    *task_note(task),
                ]
                export_table_bundle(
                    pivot,
                    title=f"{task}: {metric} across lookback settings",
                    subtitle=f"Fixed forecast horizon = {horizon} steps",
                    notes=notes,
                    out_base=out_dir / "lookback_tables" / safe_name(f"lookback_{task}_H{horizon}_{metric}"),
                    formats=formats,
                    dpi=dpi,
                    highlight_metric_cols=[c for c in pivot.columns if c != "lookback"],
                )
                count += 1
    return count


def create_fixed_lookback_tables(overall: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    if overall.empty:
        return 0
    count = 0
    for task in sorted(overall["task"].dropna().astype(str).unique()):
        task_df = overall.loc[overall["task"] == task].copy()
        for lookback in sorted(task_df["lookback"].dropna().astype(int).unique()):
            sub = task_df.loc[task_df["lookback"].astype(int) == lookback].copy()
            for metric in PRIMARY_METRICS + ["R2"]:
                if metric not in sub.columns:
                    continue
                pivot = sub.pivot_table(index="horizon", columns="model_label", values=metric, aggfunc="first")
                if pivot.empty:
                    continue
                pivot = pivot.reindex(columns=order_models(pivot.columns), fill_value=np.nan).reset_index()
                notes = [
                    f"This table keeps the input lookback fixed at {lookback} steps and shows how model error changes as the forecast horizon becomes longer.",
                    METRIC_EXPLANATIONS.get(metric, ""),
                    *task_note(task),
                ]
                export_table_bundle(
                    pivot,
                    title=f"{task}: {metric} across forecast horizons",
                    subtitle=f"Fixed lookback = {lookback} steps",
                    notes=notes,
                    out_base=out_dir / "horizon_tables" / safe_name(f"horizon_{task}_LB{lookback}_{metric}"),
                    formats=formats,
                    dpi=dpi,
                    highlight_metric_cols=[c for c in pivot.columns if c != "horizon"],
                )
                count += 1
    return count


def create_per_target_tables(by_target: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    if by_target.empty:
        return 0
    count = 0
    for task, lookback, horizon in grouped_runs(by_target):
        sub = subset_run(by_target, task, lookback, horizon)
        if sub.empty:
            continue
        for target_label in sorted(sub["target_label"].dropna().astype(str).unique()):
            tsub = sub.loc[sub["target_label"].astype(str) == target_label].copy()
            order = order_models(tsub["model_label"].tolist())
            tsub["model_label"] = pd.Categorical(tsub["model_label"], categories=order, ordered=True)
            tsub = tsub.sort_values("model_label")
            keep = ["model_label", "RMSE", "MAE", "sMAPE", "R2", "Bias", "NRMSE_mean"]
            present = [c for c in keep if c in tsub.columns]
            table_df = tsub[present].reset_index(drop=True)
            notes = [
                f"This is the clearest table for explaining performance on the {target_label} variable.",
                "Use it when discussing which variable is easy or difficult to forecast.",
            ]
            export_table_bundle(
                table_df,
                title=f"Per-target model comparison for {target_label}",
                subtitle=f"Task = {task}, Lookback = {lookback} steps, Forecast horizon = {horizon} steps",
                notes=notes,
                out_base=out_dir / "target_tables" / safe_name(f"target_{task}_{target_label}_LB{lookback}_H{horizon}"),
                formats=formats,
                dpi=dpi,
                highlight_metric_cols=[c for c in ["RMSE", "MAE", "sMAPE", "R2", "NRMSE_mean"] if c in table_df.columns],
            )
            count += 1
    return count


def create_per_step_tables(by_horizon: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    if by_horizon.empty:
        return 0
    count = 0
    for task, lookback, horizon in grouped_runs(by_horizon):
        sub = subset_run(by_horizon, task, lookback, horizon)
        if sub.empty:
            continue
        for metric in PRIMARY_METRICS + ["R2"]:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(index="horizon_step", columns="model_label", values=metric, aggfunc="first")
            if pivot.empty:
                continue
            pivot = pivot.reindex(columns=order_models(pivot.columns), fill_value=np.nan).reset_index()
            notes = [
                "This table shows how prediction error changes from the first forecast step to the last forecast step.",
                "It is especially useful for explaining whether a model degrades smoothly or sharply as it predicts further into the future.",
                METRIC_EXPLANATIONS.get(metric, ""),
                *task_note(task),
            ]
            export_table_bundle(
                pivot,
                title=f"{task}: {metric} at each forecast step",
                subtitle=f"Lookback = {lookback} steps, Forecast horizon = {horizon} steps",
                notes=notes,
                out_base=out_dir / "step_tables" / safe_name(f"step_{task}_LB{lookback}_H{horizon}_{metric}"),
                formats=formats,
                dpi=dpi,
                highlight_metric_cols=[c for c in pivot.columns if c != "horizon_step"],
            )
            count += 1
    return count


def create_ratio_tables(summary_df: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    if summary_df.empty:
        return 0
    count = 0
    for task, lookback, horizon in grouped_runs(summary_df):
        sub = subset_run(summary_df, task, lookback, horizon)
        if sub.empty:
            continue
        order = order_models(sub["model_label"].tolist())
        sub["model_label"] = pd.Categorical(sub["model_label"], categories=order, ordered=True)
        sub = sub.sort_values("model_label")
        keep = ["model_label", "RMSE_ratio", "MAE_ratio", "RMSE_gain", "BASE_RMSE", "BASE_MAE"]
        present = [c for c in keep if c in sub.columns]
        table_df = sub[present].reset_index(drop=True)
        notes = [
            "These ratios compare each model with the persistence baseline. Values below 1.0 for RMSE/MAE ratios mean the model beats persistence.",
            "Persistence is a simple forecast that repeats the last observed value forward.",
            *task_note(task),
        ]
        export_table_bundle(
            table_df,
            title=f"Performance relative to the persistence baseline for {task}",
            subtitle=f"Lookback = {lookback} steps, Forecast horizon = {horizon} steps",
            notes=notes,
            out_base=out_dir / "ratio_tables" / safe_name(f"ratio_{task}_LB{lookback}_H{horizon}"),
            formats=formats,
            dpi=dpi,
            highlight_metric_cols=[c for c in ["RMSE_ratio", "MAE_ratio", "RMSE_gain"] if c in table_df.columns],
        )
        count += 1
    return count


def create_best_tables(eval_dir: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    count = 0
    files = [
        ("best_models_overall_by_rmse.csv", "Best overall model by RMSE", ["task", "lookback", "horizon", "model", "variant", "RMSE", "MAE", "sMAPE"]),
        ("worst_models_overall_by_rmse.csv", "Worst overall model by RMSE", ["task", "lookback", "horizon", "model", "variant", "RMSE", "MAE", "sMAPE"]),
        ("best_models_by_target_rmse.csv", "Best model by target RMSE", ["task", "lookback", "horizon", "target", "model", "variant", "RMSE", "MAE", "sMAPE"]),
        ("best_models_by_horizon_step_rmse.csv", "Best model by step RMSE", ["task", "lookback", "horizon", "horizon_step", "model", "variant", "RMSE", "MAE", "sMAPE"]),
        ("best_models_overall_by_rmse_ratio.csv", "Best overall model by RMSE ratio", ["task", "lookback", "horizon", "model", "variant", "RMSE_ratio", "RMSE_gain"]),
    ]
    for filename, title, cols in files:
        path = eval_dir / filename
        df = load_csv(path)
        if df.empty:
            continue
        df = prepare_df(df)
        if "target_label" in df.columns and "target" in cols:
            df["target"] = df["target_label"]
        if "model" in cols and "model_label" in df.columns:
            df["model"] = df["model_label"]
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        table_df = df[present].copy().reset_index(drop=True)
        notes = [
            "These compact summary tables are useful in the discussion chapter when you need a quick statement about which model performed best or worst under each setting."
        ]
        export_table_bundle(
            table_df,
            title=title,
            subtitle="Compact summary table",
            notes=notes,
            out_base=out_dir / "ranking_tables" / safe_name(path.stem),
            formats=formats,
            dpi=dpi,
            highlight_metric_cols=[c for c in ["RMSE", "MAE", "sMAPE", "RMSE_ratio", "RMSE_gain"] if c in table_df.columns],
        )
        count += 1
    return count


def create_master_summary(overall: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> int:
    if overall.empty:
        return 0
    best = overall.sort_values(["task", "lookback", "horizon", "RMSE", "model_label"]).groupby(["task", "lookback", "horizon"], as_index=False).first()
    cols = ["task", "lookback", "horizon", "model_label", "RMSE", "MAE", "sMAPE"]
    if not summary_df.empty:
        ratio_keep = ["task", "lookback", "horizon", "model_label", "RMSE_ratio", "RMSE_gain"]
        s2 = summary_df[ratio_keep].copy()
        best = best.merge(s2, on=["task", "lookback", "horizon", "model_label"], how="left")
        cols += ["RMSE_ratio", "RMSE_gain"]
    table_df = best[[c for c in cols if c in best.columns]].reset_index(drop=True)
    notes = [
        "This is the single most useful summary table for the results chapter.",
        "Each row gives the best overall model for one experimental setting based on RMSE.",
    ]
    export_table_bundle(
        table_df,
        title="Best overall model for each experimental setting",
        subtitle="Recommended compact thesis summary table",
        notes=notes,
        out_base=out_dir / "summary_tables" / "best_model_master_summary",
        formats=formats,
        dpi=dpi,
        highlight_metric_cols=[c for c in ["RMSE", "MAE", "sMAPE", "RMSE_ratio", "RMSE_gain"] if c in table_df.columns],
    )
    return 1


def write_readme(out_dir: Path, created: Dict[str, int]) -> None:
    lines = [
        "THESIS TABLE OUTPUTS",
        "",
        "These tables are intended for direct use in the thesis text, appendices, or presentation slides.",
        "Each table is exported as image + CSV + LaTeX.",
        "",
        "Suggested use in the thesis:",
        "- overall_tables: one fixed experiment setting (best for main results section)",
        "- lookback_tables: explain whether longer input windows help or hurt", 
        "- horizon_tables: explain how performance changes as forecasting becomes harder",
        "- target_tables: explain which target variable is easiest or hardest to predict",
        "- step_tables: explain short-range vs long-range forecast degradation",
        "- ratio_tables: explain whether the model really beats a persistence baseline",
        "- ranking_tables: concise summary tables for discussion or appendix",
        "- summary_tables: strongest single summary table for the thesis body",
        "",
        "Tables created:",
    ]
    for key, value in created.items():
        lines.append(f"- {key}: {value}")
    save_text(out_dir / "TABLE_README.txt", "\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate thesis-ready table figures from evaluation CSVs.")
    ap.add_argument("--eval_dir", type=str, default="parquet_eval")
    ap.add_argument("--out_dir", type=str, default="thesis_tables")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    overall = prepare_df(load_csv(eval_dir / "metrics_overall_from_parquets.csv"))
    by_target = prepare_df(load_csv(eval_dir / "metrics_by_target_from_parquets.csv"))
    by_horizon = prepare_df(load_csv(eval_dir / "metrics_by_horizon_step_from_parquets.csv"))
    by_target_horizon = prepare_df(load_csv(eval_dir / "metrics_by_target_horizon_from_parquets.csv"))
    summary_df = prepare_df(load_csv(eval_dir / "summary_overall_with_persistence_ratios.csv"))

    created: Dict[str, int] = {}
    created["overall_tables"] = create_overall_tables(overall, out_dir, args.formats, args.dpi)
    created["lookback_tables"] = create_fixed_horizon_tables(overall, out_dir, args.formats, args.dpi)
    created["horizon_tables"] = create_fixed_lookback_tables(overall, out_dir, args.formats, args.dpi)
    created["target_tables"] = create_per_target_tables(by_target, out_dir, args.formats, args.dpi)
    created["step_tables"] = create_per_step_tables(by_horizon, out_dir, args.formats, args.dpi)
    created["ratio_tables"] = create_ratio_tables(summary_df, out_dir, args.formats, args.dpi)
    created["ranking_tables"] = create_best_tables(eval_dir, out_dir, args.formats, args.dpi)
    created["summary_tables"] = create_master_summary(overall, summary_df, out_dir, args.formats, args.dpi)

    # small convenience export from by_target_horizon for appendix use
    if not by_target_horizon.empty:
        appendix = by_target_horizon[[
            c for c in ["task", "lookback", "horizon", "target_label", "horizon_step", "model_label", "RMSE", "MAE", "sMAPE", "R2"]
            if c in by_target_horizon.columns
        ]].copy()
        appendix.to_csv(out_dir / "appendix_target_horizon_metrics.csv", index=False)

    write_readme(out_dir, created)

    print("\n[OK] Thesis table artifacts saved to:", out_dir)
    for key, value in created.items():
        print(f"  - {key}: {value}")
    if not by_target_horizon.empty:
        print("  - appendix_target_horizon_metrics.csv")
    print("  - TABLE_README.txt")


if __name__ == "__main__":
    main()
