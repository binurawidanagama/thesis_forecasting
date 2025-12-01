import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from src.config import load_config
from src.dataio.preprocess import build_master

def plot_weather_comparison(cfg_path, start_date, horizon=12):
    cfg = load_config(cfg_path)
    
    # 1. Load Data
    print("Loading Truth...")
    _, _, test_df = build_master(cfg)
    
    print("Loading Predictions...")
    raw_path = Path(cfg["paths"]["outputs_dir"]) / "raw_weather.parquet"
    asp_path = Path(cfg["paths"]["outputs_dir"]) / "clean_weather.parquet"
    
    if not raw_path.exists() or not asp_path.exists():
        print("Error: You need both 'raw_weather.parquet' and 'clean_weather.parquet' to compare.")
        return
        
    raw_preds = pd.read_parquet(raw_path)
    asp_preds = pd.read_parquet(asp_path)
    
    # 2. Define Variables
    # Format: (Column Name, Label, Unit)
    vars_to_plot = [
        ("shortwave_radiation_Wm2", "Radiation", "W/m²"),
        ("precipitation_mm",        "Precipitation", "mm"),
        ("relative_humidity_2m_pct","Humidity",    "%"),
        ("wind_speed_100m (m/s)",   "Wind Speed",  "m/s"),
    ]
    
    targets_in_file = [c.split('+')[0] for c in raw_preds.columns]
    plot_list = [v for v in vars_to_plot if v[0] in targets_in_file]

    # 3. Setup Plot
    fig, axes = plt.subplots(len(plot_list), 1, figsize=(14, 14), sharex=True)
    
    # Timezone fix
    t_start = pd.Timestamp(start_date).tz_localize("UTC")
    t_end   = t_start + pd.Timedelta(days=5) # 5 Day Zoom

    # 4. Loop
    for ax, (col_name, label, unit) in zip(axes, plot_list):
        
        # Prepare Columns
        pred_col = f"{col_name}+h{horizon}"
        
        # Shift Time (Prediction made at t applies to t+h)
        p_raw = raw_preds[pred_col].copy()
        p_raw.index = p_raw.index + pd.Timedelta(hours=horizon)
        
        p_asp = asp_preds[pred_col].copy()
        p_asp.index = p_asp.index + pd.Timedelta(hours=horizon)
        
        t_series = test_df[col_name]
        
        # Slice Time Window
        idx = p_raw.index.intersection(t_series.index)
        idx = idx[(idx >= t_start) & (idx <= t_end)]
        
        # Plot
        # 1. Truth (Grey/Black)
        ax.plot(t_series.loc[idx].index, t_series.loc[idx], color="black", linestyle="-", linewidth=1.5, alpha=0.5, label="Truth")
        
        # 2. Raw ELM (Blue Dashed) -> Shows the mistakes
        ax.plot(p_raw.loc[idx].index, p_raw.loc[idx], color="tab:blue", linestyle="--", linewidth=2, label="Raw ELM")
        
        # 3. ASP Cleaned (Orange Solid) -> Shows the fix
        ax.plot(p_asp.loc[idx].index, p_asp.loc[idx], color="tab:orange", linestyle="-", linewidth=2, alpha=0.9, label="ASP Cleaned")
        
        ax.set_ylabel(f"{label} ({unit})", fontsize=11, fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.7)
        
        # Highlight "Night" for Radiation
        if "Radiation" in label:
            ax.set_title("Notice: Raw ELM (Blue) has noise at night. ASP (Orange) clamps it to 0.", fontsize=10, color='red')

    axes[-1].set_xlabel("UTC Time", fontsize=12)
    plt.suptitle(f"Neuro-Symbolic Correction Analysis (Raw vs ASP)\nStart: {start_date}", fontsize=16)
    plt.tight_layout()
    
    out_file = f"{cfg['paths']['outputs_dir']}/viz_comparison_{start_date}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved Comparison Plot to {out_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/weather_full.yaml")
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()
    
    plot_weather_comparison(args.config, args.start, args.horizon)