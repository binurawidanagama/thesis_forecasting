import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from src.config import load_config

def load_finance_test_data(cfg):
    csv_path = Path(cfg["paths"]["engineered_csv"])
    df = pd.read_csv(csv_path)
    ts_col = cfg["columns"]["timestamp"]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    return df

def plot_comparison(cfg_path, start_date, horizon=12):
    cfg = load_config(cfg_path)
    
    # 1. Load Data
    print("Loading Ground Truth...")
    truth = load_finance_test_data(cfg)
    
    print("Loading Raw Predictions...")
    raw_path = Path(cfg["paths"]["outputs_dir"]) / "finance_forecast.parquet"
    pred_raw = pd.read_parquet(raw_path)
    
    print("Loading ASP Cleaned Predictions...")
    asp_path = Path(cfg["paths"]["outputs_dir"]) / "finance_forecast_asp.parquet"
    if not asp_path.exists():
        print(f"Error: ASP file not found at {asp_path}. Run run_finance_asp.py first.")
        return
    pred_asp = pd.read_parquet(asp_path)

    # 2. Setup Plotting
    # Shift predictions to visual time (t+h)
    shift = pd.Timedelta(minutes=horizon)
    
    # Slice Window (Zoom in 4 hours to see details)
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = start_ts + pd.Timedelta(hours=4)
    
    # Slice Truth
    mask_t = (truth.index >= start_ts) & (truth.index <= end_ts)
    t_slice = truth.loc[mask_t]
    
    # Slice Preds
    # We need to shift index first to match truth visual
    raw_idx = pred_raw.index + shift
    asp_idx = pred_asp.index + shift
    
    mask_p = (raw_idx >= start_ts) & (raw_idx <= end_ts)
    
    # 3. Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # --- PANEL 1: PRICE ---
    col = f"Close+h{horizon}"
    
    # True
    axes[0].plot(t_slice.index, t_slice["Close"], color="black", linewidth=2, label="True Price")
    
    # Raw (Dashed Blue)
    p_raw = pred_raw.loc[pred_raw.index[mask_p], col]
    axes[0].plot(p_raw.index + shift, p_raw, color="tab:blue", linestyle="--", linewidth=2, label="Raw ELM Output")
    
    # ASP (Solid Orange - thin to show overlap)
    p_asp = pred_asp.loc[pred_asp.index[mask_p], col]
    axes[0].plot(p_asp.index + shift, p_asp, color="tab:orange", linewidth=1.5, alpha=0.8, label="ASP Cleaned")
    
    axes[0].set_ylabel("Bitcoin Price ($)")
    axes[0].set_title(f"Price: Raw vs ASP Correction (Horizon={horizon})")
    axes[0].legend()
    axes[0].grid(True, ls=":")

    # --- PANEL 2: VOLATILITY ---
    col_vol = f"Volatility+h{horizon}"
    
    # True
    axes[1].plot(t_slice.index, t_slice["Volatility"], color="grey", linewidth=2, label="True Volatility")
    
    # Raw
    v_raw = pred_raw.loc[pred_raw.index[mask_p], col_vol]
    axes[1].plot(v_raw.index + shift, v_raw, color="tab:blue", linestyle="--", linewidth=2, label="Raw ELM Output")
    
    # ASP
    v_asp = pred_asp.loc[pred_asp.index[mask_p], col_vol]
    axes[1].plot(v_asp.index + shift, v_asp, color="tab:red", linewidth=1.5, alpha=0.9, label="ASP Cleaned")
    
    axes[1].set_ylabel("Volatility (ATR)")
    axes[1].set_title("Volatility: Raw vs ASP Correction")
    axes[1].legend()
    axes[1].grid(True, ls=":")

    plt.tight_layout()
    out_file = Path(cfg['paths']['outputs_dir']) / f"viz_comparison_{start_date.replace(' ','_').replace(':','-')}.png"
    plt.savefig(out_file)
    print(f"Saved comparison plot to {out_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finance_1m.yaml")
    ap.add_argument("--start", default="2018-05-09 12:00:00") # LUNA Crash
    args = ap.parse_args()
    plot_comparison(args.config, args.start)