import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from src.config import load_config
from src.dataio.preprocess import build_master

def plot_meteogram(cfg_path, start_date, horizon=12):
    cfg = load_config(cfg_path)
    
    # 1. Load Data
    print("Loading Truth...")
    _, _, test_df = build_master(cfg)
    
    print("Loading Predictions...")
    pred_path = Path(cfg["paths"]["outputs_dir"]) / "clean_weather.parquet"
    if not pred_path.exists():
        pred_path = Path(cfg["paths"]["outputs_dir"]) / "raw_weather.parquet"
        title_suffix = "(Raw ELM)"
    else:
        title_suffix = "(ASP Cleaned)"
    
    preds = pd.read_parquet(pred_path)
    
    # 2. Define Variables
    vars_to_plot = [
        ("shortwave_radiation_Wm2", ("Radiation",   "tab:orange", "W/m²")),
        ("temperature_2m_C",        ("Temperature", "tab:red",    "°C")),
        ("relative_humidity_2m_pct",("Humidity",    "tab:blue",   "%")),
        ("precipitation_mm",        ("Precipitation", "tab:cyan", "mm")),
        ("wind_speed_100m (m/s)",   ("Wind Speed",  "tab:green",  "m/s")),
        ("surface_pressure_hPa",    ("Pressure",    "tab:purple", "hPa")),
    ]
    
    targets_in_file = [c.split('+')[0] for c in preds.columns]
    plot_list = [v for v in vars_to_plot if v[0] in targets_in_file]

    # 3. Setup Plot
    fig, axes = plt.subplots(len(plot_list), 1, figsize=(14, 18), sharex=True)
    
    # Timezone fix
    t_start = pd.Timestamp(start_date).tz_localize("UTC")
    t_end   = t_start + pd.Timedelta(days=7)

    # 4. Loop
    for ax, (col_name, info) in zip(axes, plot_list):
        label_name, color, unit = info
        
        pred_col = f"{col_name}+h{horizon}"
        p_series = preds[pred_col].copy()
        p_series.index = p_series.index + pd.Timedelta(hours=horizon)
        t_series = test_df[col_name]
        
        idx = p_series.index.intersection(t_series.index)
        idx = idx[(idx >= t_start) & (idx <= t_end)]
        
        p_plot = p_series.loc[idx]
        t_plot = t_series.loc[idx]
        
        ax.plot(t_plot.index, t_plot, color="black", linestyle="-", linewidth=1.5, alpha=0.6, label="True")
        ax.plot(p_plot.index, p_plot, color=color,   linestyle="-", linewidth=2,   alpha=0.9, label=f"Pred (h={horizon})")
        
        ax.set_ylabel(f"{label_name} ({unit})", fontsize=11, fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.7)
        
        # Fill area for Precip and Radiation
        if "Precipitation" in label_name or "Radiation" in label_name:
            ax.fill_between(t_plot.index, t_plot, alpha=0.1, color=color)

    axes[-1].set_xlabel("UTC Time", fontsize=12)
    plt.suptitle(f"Multivariate Weather Forecast (12h Horizon) {title_suffix}\nStart: {start_date}", fontsize=16)
    plt.tight_layout()
    
    out_file = f"{cfg['paths']['outputs_dir']}/viz_meteogram_{start_date}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved Meteogram to {out_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/weather_full.yaml")
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()
    plot_meteogram(args.config, args.start, args.horizon)