"""
Universal Visualizer for Split Tasks.
Plots both ENERGY and WEATHER forecasts.
Saves plots to a clean separate folder structure.

Usage:
  python scripts/cnn_lstm_visualize.py --model cnn --out_dir plots --lookback 24 --horizon 12
  python scripts/cnn_lstm_visualize.py --model lstm --out_dir thesis_images --lookback 72 --horizon 24
  python scripts/cnn_lstm_visualize.py --model cnn (plots EVERYTHING to default 'plots' folder)
  python scripts/cnn_lstm_visualize.py --model cnn --horizon 72 (plots all lookbacks for H72)

  # Comparison Plots (CNN vs LSTM)
  python scripts/cnn_lstm_visualize.py --compare
  python scripts/cnn_lstm_visualize.py --compare --horizon 72

"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import os

# --- CONFIG: Features to Visualize ---
TARGETS_TO_PLOT = [
    "load_mw", "wind_mw", "solar_mw",
    "temperature_2m_C", "shortwave_radiation_Wm2",
    "relative_humidity_2m_pct", "precipitation_mm",
    "wind_speed_100m", "surface_pressure_hPa"
]

def get_subset(df, start_idx=2000, length=336):
    """Safe slicing of dataframe."""
    subset = df.iloc[start_idx : start_idx + length]
    if len(subset) == 0:
        subset = df.iloc[-length:] # Fallback
    return subset

def plot_single(df, feature_name, title, save_path, model_color="#d62728"):
    """Standard single-model plot."""
    true_col = f"True_{feature_name}"
    pred_col = f"Pred_{feature_name}"
    if true_col not in df.columns: return

    subset = get_subset(df)
    
    plt.figure(figsize=(14, 6))
    plt.plot(subset["timestamp"], subset[true_col], label="Actual", color="black", linewidth=1.5, alpha=0.6)
    plt.plot(subset["timestamp"], subset[pred_col], label="Forecast", color=model_color, linewidth=1.5, alpha=0.9)
    
    plt.title(title, fontsize=16)
    plt.ylabel(feature_name, fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparison(df_cnn, df_lstm, feature_name, title, save_path):
    """NEW: Overlays CNN (Red) and LSTM (Blue) vs Actual (Black)."""
    true_col = f"True_{feature_name}"
    pred_col = f"Pred_{feature_name}"
    
    if true_col not in df_cnn.columns: return

    # Ensure time alignment
    sub_cnn = get_subset(df_cnn)
    sub_lstm = get_subset(df_lstm)
    
    # Align timestamps (use CNN as base)
    timestamps = sub_cnn["timestamp"]
    
    plt.figure(figsize=(14, 6))
    
    # 1. Actual (Black)
    plt.plot(timestamps, sub_cnn[true_col], label="Actual", color="black", linewidth=2.0, alpha=0.5)
    
    # 2. LSTM (Blue - Dashed)
    plt.plot(timestamps, sub_lstm[pred_col], label="LSTM", color="blue", linewidth=1.5, linestyle="--", alpha=0.8)

    # 3. CNN (Red - Solid)
    plt.plot(timestamps, sub_cnn[pred_col], label="CNN (TCN)", color="red", linewidth=1.5, alpha=0.9)
    
    plt.title(title, fontsize=16)
    plt.ylabel(feature_name, fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"   [COMPARE] Saved {save_path}")
    plt.close()

def run_visualization(model_type, compare_mode, target_lb, target_hz, out_root):
    # ---------------------------
    # MODE A: Individual Plots
    # ---------------------------
    if not compare_mode:
        src_dir = f"artifacts_{model_type}_split"
        if not os.path.exists(src_dir):
            print(f"[ERROR] {src_dir} not found.")
            return

        pred_files = glob.glob(os.path.join(src_dir, "*", "preds.parquet"))
        for f in pred_files:
            folder_name = os.path.basename(os.path.dirname(f))
            
            # Filters
            if target_lb and f"LB{target_lb}" not in folder_name: continue
            if target_hz and f"H{target_hz}" not in folder_name: continue

            print(f"Processing {folder_name}...")
            try: df = pd.read_parquet(f)
            except: continue
            
            # Save Path
            dest = os.path.join(out_root, model_type.upper(), folder_name)
            os.makedirs(dest, exist_ok=True)
            
            for target in TARGETS_TO_PLOT:
                matches = [c.replace("True_", "") for c in df.columns if target in c and "True_" in c]
                for col in matches:
                    safe_col = col.replace(" ", "_").replace("/", "_per_").replace("(", "").replace(")", "")
                    out_name = f"{model_type.upper()}_{safe_col}.png"
                    plot_single(df, col, f"{model_type.upper()} ({folder_name})", os.path.join(dest, out_name))

    # ---------------------------
    # MODE B: Comparison Mode (CNN vs LSTM)
    # ---------------------------
    else:
        print("\n--- RUNNING COMPARISON MODE (CNN vs LSTM) ---")
        cnn_dir = "artifacts_cnn_split"
        lstm_dir = "artifacts_lstm_split"
        
        # Get list of folders in CNN dir
        cnn_folders = [os.path.basename(d) for d in glob.glob(os.path.join(cnn_dir, "*"))]
        
        for folder_name in cnn_folders:
            # Filters
            if target_lb and f"LB{target_lb}" not in folder_name: continue
            if target_hz and f"H{target_hz}" not in folder_name: continue
            
            # Check if matching LSTM folder exists
            path_c = os.path.join(cnn_dir, folder_name, "preds.parquet")
            path_l = os.path.join(lstm_dir, folder_name, "preds.parquet")
            
            if not os.path.exists(path_l):
                continue # Skip if no matching LSTM result

            print(f"Comparing {folder_name}...")
            try:
                df_c = pd.read_parquet(path_c)
                df_l = pd.read_parquet(path_l)
            except:
                continue
            
            # Output Directory
            dest = os.path.join(out_root, "COMPARISON", folder_name)
            os.makedirs(dest, exist_ok=True)
            
            for target in TARGETS_TO_PLOT:
                matches = [c.replace("True_", "") for c in df_c.columns if target in c and "True_" in c]
                for col in matches:
                    safe_col = col.replace(" ", "_").replace("/", "_per_").replace("(", "").replace(")", "")
                    out_name = f"COMPARE_{safe_col}.png"
                    
                    title = f"CNN vs LSTM ({folder_name})"
                    plot_comparison(df_c, df_l, col, title, os.path.join(dest, out_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Mode Selection
    parser.add_argument("--model", type=str, choices=["cnn", "lstm"], help="Plot individual model")
    parser.add_argument("--compare", action="store_true", help="Enable Comparison Mode (CNN vs LSTM)")
    
    # Options
    parser.add_argument("--out_dir", type=str, default="plots", help="Output folder")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    
    args = parser.parse_args()
    
    # Validation
    if not args.model and not args.compare:
        print("Error: You must specify either --model [cnn/lstm] OR --compare")
    else:
        run_visualization(args.model, args.compare, args.lookback, args.horizon, args.out_dir)