import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import load_config
from src.dataio.preprocess import build_master

# Set plotting style
sns.set_theme(style="white", rc={"grid.linestyle": ":"})

def plot_week(cfg_path, start_date="2022-08-01", end_date="2022-08-08", use_asp=True):
    cfg = load_config(cfg_path)
    
    # 1. Load Ground Truth (Test DF)
    print("Loading Ground Truth...")
    _, _, test_df = build_master(cfg)
    
    # 2. Load Predictions
    if use_asp:
        pred_path = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions_asp.parquet"
        title_suffix = "(ASP Refined)"
    else:
        pred_path = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions.parquet"
        title_suffix = "(Raw ELM)"
        
    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}")
        return

    print(f"Loading Predictions from {pred_path}...")
    preds = pd.read_parquet(pred_path)
    
    # 3. Align Data (Selecting Horizon h=1 explicitly)
    # FIX: Use exact column names to avoid matching +h10, +h11, etc.
    target_cols_h1 = ["wind_mw+h12", "solar_mw+h12", "load_mw+h12", "price+h12"]
    
    # Check if they exist (ASP output might change order, but names should persist)
    missing = [c for c in target_cols_h1 if c not in preds.columns]
    if missing:
        print(f"Error: Missing expected columns {missing} in predictions.")
        return

    preds_h1 = preds[target_cols_h1].copy()
    
    # Shift index: prediction made at 't' for horizon 'h1' applies to time 't+1'
    preds_h1.index = preds_h1.index + pd.Timedelta(hours=12)
    
    # Rename columns for easier access
    preds_h1.columns = ["Wind_Pred", "Solar_Pred", "Load_Pred", "Price_Pred"]
    
    # Combine into one DF for slicing
    combined = pd.DataFrame(index=preds_h1.index)
    
    # Map Truth columns
    # We must match indices. test_df might have different range.
    common_idx = combined.index.intersection(test_df.index)
    
    combined.loc[common_idx, "Wind_True"]  = test_df.loc[common_idx, "wind_mw"]
    combined.loc[common_idx, "Solar_True"] = test_df.loc[common_idx, "solar_mw"]
    combined.loc[common_idx, "Load_True"]  = test_df.loc[common_idx, "load_mw"]
    combined.loc[common_idx, "Price_True"] = test_df.loc[common_idx, "price_eur_mwh"]
    
    # Add Preds
    combined.loc[common_idx, "Wind_Pred"]  = preds_h1.loc[common_idx, "Wind_Pred"]
    combined.loc[common_idx, "Solar_Pred"] = preds_h1.loc[common_idx, "Solar_Pred"]
    combined.loc[common_idx, "Load_Pred"]  = preds_h1.loc[common_idx, "Load_Pred"]
    combined.loc[common_idx, "Price_Pred"] = preds_h1.loc[common_idx, "Price_Pred"]
    
    # Drop rows where we don't have truth (e.g. end of file)
    combined = combined.dropna()

    # 4. Slice the specific week
    plot_df = combined.loc[start_date:end_date]
    
    if plot_df.empty:
        print(f"No data found for range {start_date} to {end_date}")
        return

    # 5. Plotting
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    def plot_channel(ax, true_col, pred_col, label, unit, color_true="tab:blue", color_pred="tab:orange"):
        ax.plot(plot_df.index, plot_df[true_col], label=f"True {label}", color=color_true, linewidth=1.5)
        ax.plot(plot_df.index, plot_df[pred_col], label=f"Pred {label}", color=color_pred, linewidth=1.5, alpha=0.9)
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", which="both")

    plot_channel(axs[0], "Wind_True", "Wind_Pred", "Wind", "MW")
    plot_channel(axs[1], "Solar_True", "Solar_Pred", "Solar", "MW")
    plot_channel(axs[2], "Load_True", "Load_Pred", "Load", "MW")
    plot_channel(axs[3], "Price_True", "Price_Pred", "Price", "€/MWh")
    
    plt.xlabel("UTC Time")
    plt.suptitle(f"Forecast vs Actuals: {start_date} to {end_date} {title_suffix}", fontsize=16)
    plt.tight_layout()
    
    out_file = f"viz_week_{start_date}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--end", default="2022-08-08")
    ap.add_argument("--asp", action="store_true", help="Use ASP predictions")
    args = ap.parse_args()
    
    plot_week(args.config, args.start, args.end, args.asp)