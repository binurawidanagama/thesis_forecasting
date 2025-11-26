# scripts/plot_results.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.dataio.preprocess import build_master
from src.dataio.window import make_windows


FIG_DIR = Path("outputs") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helpers to get full dataframe and test windows
# ---------------------------------------------------------

def load_full_df(cfg):
    """Return full df (train+val+test concatenated)."""
    train_df, val_df, test_df = build_master(cfg)
    full = pd.concat([train_df, val_df, test_df]).sort_index()
    return full, train_df, val_df, test_df


def get_test_truth(cfg):
    """
    Build test windows and return:
      test_df, Yte, te_index, caps_w, caps_s, hz

    Yte shape: [N, H, 4] -> [cf_wind, cf_solar, load_mw, price_eur_mwh]
    """
    _, _, test_df = build_master(cfg)
    feature_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "holiday",
        "cap_wind_mw", "cap_solar_mw",
        "wind_mw", "solar_mw", "load_mw", "price_eur_mwh",
    ]
    target_cols = ["cf_wind", "cf_solar", "load_mw", "price_eur_mwh"]
    ctx = cfg["features"]["context_hours"]
    hz = cfg["features"]["horizon_hours"]

    Xte, Yte, te_index = make_windows(test_df, feature_cols, target_cols, ctx, hz)
    N = Yte.shape[0]
    caps_w = test_df["cap_wind_mw"].values[ctx: ctx + N]
    caps_s = test_df["cap_solar_mw"].values[ctx: ctx + N]

    return test_df, Yte, te_index, caps_w, caps_s, hz


# ---------------------------------------------------------
# 1) Alternative to heatmaps: simple charts
# ---------------------------------------------------------

def plot_hourly_profile(cfg_path, column="load_mw", out_path=None):
    """
    Average value by hour-of-day: simple line chart.
    Example: load_mw, price_eur_mwh, wind_mw, solar_mw, cf_solar, etc.
    """
    cfg = load_config(cfg_path)
    full, _, _, _ = load_full_df(cfg)
    profile = full.groupby(full.index.hour)[column].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(profile.index, profile.values, marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(column)
    ax.set_title(f"Average {column} by hour-of-day")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = FIG_DIR / f"hourly_profile_{column}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_monthly_profile(cfg_path, column="load_mw", out_path=None):
    """
    Average value by month: simple bar chart.
    """
    cfg = load_config(cfg_path)
    full, _, _, _ = load_full_df(cfg)
    profile = full.groupby(full.index.month)[column].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(profile.index, profile.values)
    ax.set_xlabel("Month")
    ax.set_xticks(range(1, 13))
    ax.set_ylabel(column)
    ax.set_title(f"Average {column} by month")
    fig.tight_layout()

    if out_path is None:
        out_path = FIG_DIR / f"monthly_profile_{column}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------
# 2) Ground truth vs prediction (single model)
# ---------------------------------------------------------

def get_truth_and_pred(cfg_path, pred_path, target="price", horizon=1):
    """
    Return time index, y_true, y_pred, label for a given target and horizon.

    target: 'wind', 'solar', 'load', or 'price'
    horizon: 1..H (H = cfg['features']['horizon_hours'])
    """
    cfg = load_config(cfg_path)
    test_df, Yte, te_index, caps_w, caps_s, hz = get_test_truth(cfg)
    assert 1 <= horizon <= hz
    idx_h = horizon - 1

    # Ground truth from Yte
    cf_w = Yte[:, idx_h, 0]
    cf_s = Yte[:, idx_h, 1]
    load_true = Yte[:, idx_h, 2]
    price_true = Yte[:, idx_h, 3]

    wind_true = cf_w * caps_w
    solar_true = cf_s * caps_s

    # Predictions from parquet
    pred_df = pd.read_parquet(pred_path).loc[te_index]
    if target == "wind":
        y_true = wind_true
        y_pred = pred_df[f"wind_mw+h{horizon}"].values
        label = "Wind (MW)"
    elif target == "solar":
        y_true = solar_true
        y_pred = pred_df[f"solar_mw+h{horizon}"].values
        label = "Solar (MW)"
    elif target == "load":
        y_true = load_true
        y_pred = pred_df[f"load_mw+h{horizon}"].values
        label = "Load (MW)"
    else:
        y_true = price_true
        y_pred = pred_df[f"price+h{horizon}"].values
        label = "Price (€/MWh)"

    return te_index, y_true, y_pred, label


def plot_gt_vs_pred_single(cfg_path, pred_path,
                           target="price", horizon=1,
                           start=None, end=None,
                           model_name="dCeNN-ELM",
                           out_path=None):
    """
    Simple line plot: ground truth vs predictions for one model.
    """
    t_index, y_true, y_pred, label = get_truth_and_pred(cfg_path, pred_path, target, horizon)

    # Convert to Series for easy slicing by date
    s_true = pd.Series(y_true, index=t_index)
    s_pred = pd.Series(y_pred, index=t_index)

    if start is not None:
        s_true = s_true.loc[start:]
        s_pred = s_pred.loc[start:]
    if end is not None:
        s_true = s_true.loc[:end]
        s_pred = s_pred.loc[:end]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s_true.index, s_true.values, label="Ground truth", linewidth=2)
    ax.plot(s_pred.index, s_pred.values, label=model_name, linestyle="--")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} – Ground truth vs {model_name}, horizon h={horizon}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        safe_target = target.replace("/", "_")
        start_str = (start or "full").replace(":", "-") if isinstance(start, str) else "full"
        out_path = FIG_DIR / f"gt_vs_pred_{safe_target}_h{horizon}_{start_str}.png"

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------
# 3) Ground truth vs prediction (multiple models on one plot)
# ---------------------------------------------------------

def plot_gt_vs_pred_multi(cfg_path, pred_paths, model_names,
                          target="price", horizon=1,
                          start=None, end=None,
                          out_path=None):
    """
    Ground truth + multiple models on one plot.
    Example: dCeNN-ELM vs CNN vs LSTM.

    pred_paths: list of parquet files
    model_names: list of labels, same length as pred_paths
    """
    assert len(pred_paths) == len(model_names)

    cfg = load_config(cfg_path)
    test_df, Yte, te_index, caps_w, caps_s, hz = get_test_truth(cfg)
    assert 1 <= horizon <= hz
    idx_h = horizon - 1

    # Ground truth
    cf_w = Yte[:, idx_h, 0]
    cf_s = Yte[:, idx_h, 1]
    load_true = Yte[:, idx_h, 2]
    price_true = Yte[:, idx_h, 3]

    wind_true = cf_w * caps_w
    solar_true = cf_s * caps_s

    # Choose truth & label by target
    if target == "wind":
        s_true = pd.Series(wind_true, index=te_index)
        col_name = lambda h: f"wind_mw+h{h}"
        label = "Wind (MW)"
    elif target == "solar":
        s_true = pd.Series(solar_true, index=te_index)
        col_name = lambda h: f"solar_mw+h{h}"
        label = "Solar (MW)"
    elif target == "load":
        s_true = pd.Series(load_true, index=te_index)
        col_name = lambda h: f"load_mw+h{h}"
        label = "Load (MW)"
    else:
        s_true = pd.Series(price_true, index=te_index)
        col_name = lambda h: f"price+h{h}"
        label = "Price (€/MWh)"

    # Slice time range
    if start is not None:
        s_true = s_true.loc[start:]
    if end is not None:
        s_true = s_true.loc[:end]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s_true.index, s_true.values, label="Ground truth", linewidth=2)

    # Add each model
    for p, name in zip(pred_paths, model_names):
        pred_df = pd.read_parquet(p).loc[te_index]
        s_pred = pd.Series(pred_df[col_name(horizon)].values, index=te_index)
        if start is not None:
            s_pred = s_pred.loc[start:]
        if end is not None:
            s_pred = s_pred.loc[:end]
        ax.plot(s_pred.index, s_pred.values, label=name, linestyle="--")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} – Ground truth vs models, h={horizon}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        safe_target = target.replace("/", "_")
        start_str = (start or "full").replace(":", "-") if isinstance(start, str) else "full"
        out_path = FIG_DIR / f"gt_vs_pred_multi_{safe_target}_h{horizon}_{start_str}.png"

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------
# CLI usage
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thesis plotting helpers")
    sub = parser.add_subparsers(dest="cmd")

    # hourly profile
    p_hour = sub.add_parser("hourly", help="Average by hour-of-day")
    p_hour.add_argument("--config", required=True)
    p_hour.add_argument("--column", default="load_mw")

    # monthly profile
    p_month = sub.add_parser("monthly", help="Average by month")
    p_month.add_argument("--config", required=True)
    p_month.add_argument("--column", default="load_mw")

    # gt vs pred (single model)
    p_gt1 = sub.add_parser("gt_single", help="Ground truth vs prediction (single model)")
    p_gt1.add_argument("--config", required=True)
    p_gt1.add_argument("--pred", required=True)
    p_gt1.add_argument("--target", default="price",
                       choices=["wind", "solar", "load", "price"])
    p_gt1.add_argument("--horizon", type=int, default=1)
    p_gt1.add_argument("--start", default=None)
    p_gt1.add_argument("--end", default=None)
    p_gt1.add_argument("--name", default="Model")

    # gt vs pred (multiple models)
    p_gtm = sub.add_parser("gt_multi", help="Ground truth vs multiple models")
    p_gtm.add_argument("--config", required=True)
    p_gtm.add_argument("--preds", nargs="+", required=True,
                       help="List of prediction parquet paths")
    p_gtm.add_argument("--names", nargs="+", required=True,
                       help="List of model names (same length as preds)")
    p_gtm.add_argument("--target", default="price",
                       choices=["wind", "solar", "load", "price"])
    p_gtm.add_argument("--horizon", type=int, default=1)
    p_gtm.add_argument("--start", default=None)
    p_gtm.add_argument("--end", default=None)

    args = parser.parse_args()

    if args.cmd == "hourly":
        plot_hourly_profile(args.config, args.column)
    elif args.cmd == "monthly":
        plot_monthly_profile(args.config, args.column)
    elif args.cmd == "gt_single":
        plot_gt_vs_pred_single(
            cfg_path=args.config,
            pred_path=args.pred,
            target=args.target,
            horizon=args.horizon,
            start=args.start,
            end=args.end,
            model_name=args.name,
        )
    elif args.cmd == "gt_multi":
        plot_gt_vs_pred_multi(
            cfg_path=args.config,
            pred_paths=args.preds,
            model_names=args.names,
            target=args.target,
            horizon=args.horizon,
            start=args.start,
            end=args.end,
        )
    else:
        parser.print_help()
