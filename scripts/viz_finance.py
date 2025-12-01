import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from src.config import load_config

def load_finance_test_data(cfg):
    csv_path = Path(cfg["paths"]["engineered_csv"])
    print(f"Loading raw finance data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Force UTC
    ts_col = cfg["columns"]["timestamp"]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    return df

def plot_finance(cfg_path, requested_start, horizon=12):
    cfg = load_config(cfg_path)
    
    # 1. Load Preds
    pred_path = Path(cfg["paths"]["outputs_dir"]) / "finance_forecast.parquet"
    if not pred_path.exists():
        print(f"Error: Predictions not found at {pred_path}")
        return
    preds = pd.read_parquet(pred_path)
    
    # 2. Load Truth
    test_df = load_finance_test_data(cfg)

    # 3. Determine Plot Start Date (Auto-Detect)
    # Convert requested string to UTC timestamp
    req_ts = pd.Timestamp(requested_start).tz_localize("UTC") if pd.Timestamp(requested_start).tzinfo is None else pd.Timestamp(requested_start)
    
    # Check if requested date is inside the prediction range
    p_start = preds.index.min()
    p_end = preds.index.max()
    
    print("\n--- DIAGNOSTICS ---")
    print(f"Predictions Start: {p_start}")
    print(f"Predictions End:   {p_end}")
    print(f"Requested Start:   {req_ts}")
    
    if req_ts < p_start or req_ts > p_end:
        print(f"\n⚠️ WARNING: Requested date {req_ts} is out of range!")
        print(f"-> Switching to first available date: {p_start}")
        plot_start = p_start
    else:
        plot_start = req_ts

    # 4. Prepare Plot Data
    shift = pd.Timedelta(minutes=horizon)
    plot_end = plot_start + pd.Timedelta(hours=4) # Zoom in 4 hours
    
    # Price
    if "Close" not in test_df.columns:
        print("Error: 'Close' column missing in test data.")
        return

    # Slice Truth
    t_mask = (test_df.index >= plot_start) & (test_df.index <= plot_end)
    t_plot = test_df.loc[t_mask, "Close"]
    
    # Slice Preds
    # Align time: Prediction at t applies to t+h
    p_series = preds[f"Close+h{horizon}"].copy()
    p_series.index = p_series.index + shift
    
    p_mask = (p_series.index >= plot_start) & (p_series.index <= plot_end)
    p_plot = p_series.loc[p_mask]

    # 5. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top Panel: Price
    axes[0].plot(t_plot.index, t_plot, label="True Price", color="black", lw=1.5)
    axes[0].plot(p_plot.index, p_plot, label=f"Pred Price (h={horizon})", color="#f2a900", lw=2)
    axes[0].set_ylabel("Bitcoin Price (USD)")
    axes[0].legend()
    axes[0].grid(True, ls=":")
    axes[0].set_title(f"Price Forecast (Start: {plot_start})")

    # Bottom Panel: Volatility
    if "Volatility" in test_df.columns:
        t_vol = test_df.loc[t_mask, "Volatility"]
        
        p_vol = preds[f"Volatility+h{horizon}"].copy()
        p_vol.index = p_vol.index + shift
        p_vol = p_vol.loc[p_mask]
        
        axes[1].plot(t_vol.index, t_vol, label="True Volatility", color="grey", ls="--")
        axes[1].plot(p_vol.index, p_vol, label=f"Pred Volatility", color="tab:red")
        axes[1].set_ylabel("Volatility (ATR)")
        axes[1].legend()
        axes[1].grid(True, ls=":")

    plt.tight_layout()
    out_file = Path(cfg['paths']['outputs_dir']) / f"viz_auto_{str(plot_start).split()[0]}.png"
    plt.savefig(out_file)
    print(f"\n✅ Success! Plot saved to: {out_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finance_1m.yaml")
    ap.add_argument("--start", default="2022-01-01 00:00:00") # Default doesn't matter, it will auto-fix
    args = ap.parse_args()
    plot_finance(args.config, args.start)