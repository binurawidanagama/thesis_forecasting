import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import re
from src.config import load_config

def run_asp(cfg_path):
    cfg = load_config(cfg_path)
    pred_path = Path(cfg["paths"]["outputs_dir"]) / "raw_weather.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found at: {pred_path}")
        
    preds = pd.read_parquet(pred_path)
    
    # Mapping Config Names -> ASP Names
    t_map = {
        "temperature_2m_C": "temp", 
        "relative_humidity_2m_pct": "hum", 
        "wind_speed_100m (m/s)": "wind", 
        "surface_pressure_hPa": "press",
        "shortwave_radiation_Wm2": "rad",
        "precipitation_mm": "precip"
    }
    
    # Get Night Hours from config
    night_hours = set(cfg["asp"]["pv_night_hours"])
    
    facts_path = Path("weather_facts.lp")
    hz = cfg["features"]["horizon_hours"]
    BATCH = 100
    SCALE = 100
    
    print("Running Weather ASP (Reasoning Layer)...")
    
    cleaned_preds = preds.copy()
    
    # Ensure timestamp is available for night calculation
    if "timestamp" not in cleaned_preds.columns:
        cleaned_preds["timestamp"] = cleaned_preds.index
    
    # Pre-compile Regex for speed (catch 'repair(kind, target, sample, horizon)')
    pattern = re.compile(r"repair\(\s*([a-z_]+)\s*,\s*([a-z]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

    for start in tqdm(range(0, len(preds), BATCH), desc="Processing Batches"):
        end = min(start + BATCH, len(preds))
        chunk = cleaned_preds.iloc[start:end]
        
        # --- 1. WRITE FACTS ---
        with open(facts_path, "w") as f:
            for h in range(1, hz+1): f.write(f"horizon({h}).\n")
            
            for i in range(len(chunk)):
                s_glob = start + i
                f.write(f"sample({s_glob}).\n")
                row = chunk.iloc[i]
                
                # Loop Horizon first, then Targets
                for h in range(1, hz + 1):
                    # Write Predictions
                    for t_col, asp_name in t_map.items():
                        val = row.get(f"{t_col}+h{h}", 0.0)
                        # Clingo needs integers
                        f.write(f"pred({asp_name},{s_glob},{h},{int(val*SCALE)}).\n")
                    
                    # Write Night Fact
                    ts = row["timestamp"]
                    hour_h = (ts.hour + h) % 24
                    if hour_h in night_hours:
                        f.write(f"night({s_glob},{h}).\n")

        # --- 2. SOLVE (Call Clingo) ---
        cmd = ["clingo", "src/asp/weather_physics.lp", str(facts_path), "--opt-mode=opt", "--quiet=1", "--time-limit=5"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        # --- 3. PARSE REPAIRS ---
        for line in res.stdout.splitlines():
            # Use findall because Clingo puts multiple atoms on one line
            matches = pattern.findall(line)
            for match in matches:
                kind, tgt, s, h = match
                s, h = int(s), int(h)
                
                # Map back to specific column name (e.g., 'wind_speed...+h12')
                col = ""
                for k, v in t_map.items(): 
                    if v == tgt: col = f"{k}+h{h}"
                
                if not col or col not in cleaned_preds.columns: continue
                
                # Safe Indexing
                idx_label = cleaned_preds.index[s]
                curr = cleaned_preds.at[idx_label, col]
                
                # --- APPLY REPAIR LOGIC ---
                if kind == "bound":
                    if tgt == "hum":
                        cleaned_preds.at[idx_label, col] = max(0, min(100, curr))
                    elif tgt == "press":
                        cleaned_preds.at[idx_label, col] = max(850, min(1100, curr))
                    elif tgt == "temp":
                        # New Regional Bounds
                        cleaned_preds.at[idx_label, col] = max(-30, min(45, curr))
                    else:
                        # Standard Non-Negative (Wind, Rad, Precip)
                        cleaned_preds.at[idx_label, col] = max(0, curr)
                        
                elif kind == "night" and tgt == "rad":
                    cleaned_preds.at[idx_label, col] = 0.0
                    
                elif kind == "noise_gate" and tgt == "precip":
                    # Force tiny precip (drizzle noise) to zero
                    cleaned_preds.at[idx_label, col] = 0.0
                    
                elif kind == "boost_hum" and tgt == "hum":
                    # If raining, ensure Humidity is at least 75%
                    cleaned_preds.at[idx_label, col] = max(curr, 75.0)

    # --- 4. SAFETY PASS ---
    # Fail-safe: Manually zero out radiation at night to ensure visual correctness
    print("[ASP] Running Safety Pass for Night Radiation...")
    
    start_hours = cleaned_preds["timestamp"].dt.hour.values
    # Create 2D array of hours [N, H]
    hours_2d = (start_hours[:, None] + np.arange(1, hz + 1)[None, :]) % 24
    is_night = np.isin(hours_2d, list(night_hours))

    rad_col_base = "shortwave_radiation_Wm2"
    for h_idx in range(hz):
        h = h_idx + 1
        col = f"{rad_col_base}+h{h}"
        if col in cleaned_preds.columns:
            mask_h = is_night[:, h_idx] 
            cleaned_preds.loc[mask_h, col] = 0.0

    # Cleanup
    if "timestamp" in cleaned_preds.columns:
        cleaned_preds = cleaned_preds.drop(columns=["timestamp"])
        
    out_path = Path(cfg["paths"]["outputs_dir"]) / "clean_weather.parquet"
    cleaned_preds.to_parquet(out_path)
    print(f"Saved cleaned weather to {out_path}")
    
    if facts_path.exists(): os.remove(facts_path)

if __name__ == "__main__":
    run_asp("configs/weather_full.yaml")