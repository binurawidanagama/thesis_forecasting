import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def stack(pred, prefix, H):
    return np.stack([pred[f"{prefix}+h{h+1}"].values for h in range(H)], axis=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/predictions/test_predictions.parquet")
    ap.add_argument("--truth", default="data/interim/test.parquet")
    ap.add_argument("--start", required=True, help="UTC start like 2022-08-01 00:00:00")
    ap.add_argument("--hours", type=int, default=168)  # one week
    # Default to 12 to see the "Hard" prediction
    ap.add_argument("--horizon", type=int, default=12, help="Horizon step to visualize (1-12)")
    args = ap.parse_args()

    pred = pd.read_parquet(args.pred)
    truth = pd.read_parquet(args.truth)
    
    h_step = args.horizon
    print(f"Visualizing Horizon: +{h_step} hours")

    # 1. Select specific horizon columns
    cols = ["wind_mw", "solar_mw", "load_mw", "price"]
    target_cols = [f"{c}+h{h_step}" for c in cols]
    
    # Check if columns exist
    if not all(c in pred.columns for c in target_cols):
        print(f"Error: Columns {target_cols} not found in predictions.")
        exit(1)

    p_viz = pred[target_cols].copy()
    
    # 2. Rename to clean names
    p_viz.columns = cols

    # 3. CRITICAL: Shift Index
    # A prediction made at t=00:00 for horizon +12 applies to t=12:00.
    # We shift the index forward so it aligns with the Truth at 12:00.
    p_viz.index = p_viz.index + pd.Timedelta(hours=h_step)

    # 4. Align and Slice Window
    # Create the requested time range
    idx = pd.date_range(args.start, periods=args.hours, freq="h", tz="UTC")
    
    # Find intersection of (Requested Range) AND (Available Preds) AND (Available Truth)
    valid_idx = idx.intersection(p_viz.index).intersection(truth.index)
    
    if valid_idx.empty:
        print(f"No overlapping data found for {args.start} with +{h_step}h shift.")
        exit(1)

    # Slice data
    p_viz = p_viz.loc[valid_idx]
    tr = truth.loc[valid_idx, ["wind_mw","solar_mw","load_mw","price_eur_mwh"]].rename(columns={"price_eur_mwh":"price"})

    # 5. Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    labels = ["Wind (MW)", "Solar (MW)", "Load (MW)", "Price (€/MWh)"]
    
    for ax, key, label in zip(axes, cols, labels):
        # Plot Truth
        ax.plot(tr.index, tr[key], label=f"True", color="tab:blue", linewidth=1.5)
        # Plot Prediction
        ax.plot(p_viz.index, p_viz[key], label=f"Pred (h={h_step})", color="tab:orange", linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel(label)
        ax.grid(True, ls=":")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("UTC time")
    plt.suptitle(f"Forecast vs Actuals (Horizon: {h_step}h)", fontsize=16)
    plt.tight_layout()
    
    out = Path("outputs") / f"viz_week_h{h_step}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")