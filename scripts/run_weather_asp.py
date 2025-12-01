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
    preds = pd.read_parquet(pred_path)
    
    # --- UPDATED MAPPING ---
    targets = cfg["features"]["target_features"]
    t_map = {
        "temperature_2m_C": "temp", 
        "relative_humidity_2m_pct": "hum", 
        "wind_speed_100m (m/s)": "wind", 
        "surface_pressure_hPa": "press",
        "shortwave_radiation_Wm2": "rad",
        "precipitation_mm": "precip"
    }
    
    # Get Night Hours
    night_hours = set(cfg["asp"]["pv_night_hours"])
    
    facts_path = Path("weather_facts.lp")
    hz = cfg["features"]["horizon_hours"]
    BATCH = 100
    SCALE = 100
    
    print("Running Weather ASP (6 Variables)...")
    
    cleaned_preds = preds.copy()
    
    # Ensure timestamp is available for night calculation
    if "timestamp" not in cleaned_preds.columns:
        # Assuming index is timestamp
        cleaned_preds["timestamp"] = cleaned_preds.index
    
    for start in tqdm(range(0, len(preds), BATCH)):
        end = min(start+BATCH, len(preds))
        chunk = cleaned_preds.iloc[start:end]
        
        with open(facts_path, "w") as f:
            for h in range(1, hz+1): f.write(f"horizon({h}).\n")
            for i in range(len(chunk)):
                s_glob = start + i
                f.write(f"sample({s_glob}).\n")
                row = chunk.iloc[i]
                
                # Write Preds
                for t_col in targets:
                    asp_name = t_map[t_col]
                    val = row.get(f"{t_col}+h{h}", 0.0)
                    f.write(f"pred({asp_name},{s_glob},{h},{int(val*SCALE)}).\n")
                
                # Write Night Fact
                ts = row["timestamp"]
                hour_h = (ts.hour + h) % 24
                if hour_h in night_hours:
                    f.write(f"night({s_glob},{h}).\n")

        # Solve
        cmd = ["clingo", "src/asp/weather_physics.lp", str(facts_path), "--opt-mode=opt", "--quiet=1"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse Repairs
        for line in res.stdout.splitlines():
            if "repair(" in line:
                m = re.search(r"repair\(([a-z_]+),([a-z]+),(\d+),(\d+)\)", line)
                if m:
                    kind, tgt, s, h = m.groups()
                    s, h = int(s), int(h)
                    
                    # Map back to column name
                    col = ""
                    for k,v in t_map.items(): 
                        if v == tgt: col = f"{k}+h{h}"
                    
                    if not col or col not in cleaned_preds.columns: continue
                    
                    curr = cleaned_preds.at[cleaned_preds.index[s], col]
                    
                    # --- APPLY LOGIC ---
                    if kind == "bound":
                        if tgt == "hum":
                            cleaned_preds.at[cleaned_preds.index[s], col] = max(0, min(100, curr))
                        elif tgt == "press":
                            cleaned_preds.at[cleaned_preds.index[s], col] = max(850, min(1100, curr))
                        else:
                            # Generic non-negative (Wind, Rad, Precip)
                            cleaned_preds.at[cleaned_preds.index[s], col] = max(0, curr)
                            
                    elif kind == "night" and tgt == "rad":
                        cleaned_preds.at[cleaned_preds.index[s], col] = 0.0

    out_path = Path(cfg["paths"]["outputs_dir"]) / "clean_weather.parquet"
    # Drop temp column before saving
    if "timestamp" in cleaned_preds.columns:
        cleaned_preds = cleaned_preds.drop(columns=["timestamp"])
        
    cleaned_preds.to_parquet(out_path)
    print(f"Saved cleaned weather to {out_path}")
    if facts_path.exists(): os.remove(facts_path)

if __name__ == "__main__":
    run_asp("configs/weather_full.yaml")