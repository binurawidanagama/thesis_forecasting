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
    pred_path = Path(cfg["paths"]["outputs_dir"]) / "finance_forecast.parquet"
    
    if not pred_path.exists():
        print(f"Error: No predictions found at {pred_path}")
        return

    # Load data
    print("Loading predictions...")
    preds = pd.read_parquet(pred_path)
    
    # Mapping Config Names -> ASP Names
    t_map = {
        "Close": "close", 
        "Volatility": "vol"
    }
    targets = list(t_map.keys())
    
    facts_path = Path("finance_facts.lp")
    hz = cfg["features"]["horizon_hours"]
    
    # --- OPTIMIZATION: Reduce Precision to Integers ---
    # Old: SCALE = 100 (Cents). Result: Millions of unique atoms. Slow.
    # New: SCALE = 1 (Dollars). Result: Fewer unique atoms. Fast.
    BATCH = 1000 
    SCALE = 1 
    
    print(f"Running Finance ASP on {len(preds)} rows (Batch Size: {BATCH}, Scale: {SCALE})...")
    
    cleaned_preds = preds.copy()
    repairs_count = 0
    
    # Pre-compile regex for speed
    pattern = re.compile(r"repair\(\s*([a-z_]+)\s*,\s*([a-z]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

    for start in tqdm(range(0, len(preds), BATCH), desc="Processing Batches"):
        end = min(start+BATCH, len(preds))
        chunk = cleaned_preds.iloc[start:end]
        
        # --- FAST FACT GENERATION ---
        facts_buffer = []
        
        # Write horizons
        for h in range(1, hz+1): 
            facts_buffer.append(f"horizon({h}).\n")
            
        # Iterate rows
        for i in range(len(chunk)):
            s_glob = start + i
            facts_buffer.append(f"sample({s_glob}).\n")
            
            row = chunk.iloc[i]
            
            for t_col in targets:
                asp_name = t_map[t_col]
                
                # Loop through horizon
                for h in range(1, hz+1):
                    col_name = f"{t_col}+h{h}"
                    if col_name in row:
                        val = row[col_name]
                        if pd.isna(val): val = 0
                        
                        # Round to integer for cleaner ASP logic
                        scaled_val = int(round(val * SCALE))
                        
                        facts_buffer.append(f"pred({asp_name},{s_glob},{h},{scaled_val}).\n")

        # Write to disk
        with open(facts_path, "w") as f:
            f.writelines(facts_buffer)

        # --- SOLVE ---
        cmd = ["clingo", "src/asp/finance_physics.lp", str(facts_path), "--opt-mode=opt", "--quiet=1", "--time-limit=5"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        # --- PARSE REPAIRS ---
        for line in res.stdout.splitlines():
            matches = pattern.findall(line)
            for match in matches:
                kind, tgt, s, h = match
                s, h = int(s), int(h)
                repairs_count += 1
                
                # Map back to column
                col = ""
                for k,v in t_map.items(): 
                    if v == tgt: col = f"{k}+h{h}"
                
                if not col: continue
                
                # Use Label-based index safely
                idx_label = cleaned_preds.index[s]
                curr = cleaned_preds.at[idx_label, col]
                
                if kind == "bound_low":
                    # Fix Negatives
                    if tgt == "vol":
                        cleaned_preds.at[idx_label, col] = max(0.01, curr)
                    else:
                        cleaned_preds.at[idx_label, col] = max(0.0, curr)
                        
                elif kind == "bound_sanity":
                    # Fix Explosions
                    if tgt == "close":
                        cleaned_preds.at[idx_label, col] = max(100.0, min(1000000.0, curr))

    print(f"ASP Identified {repairs_count} violations.")

    # --- 4. SAFETY PASS ---
    print("Applying Safety Pass (Vectorized)...")
    
    # 1. Price > 0
    price_cols = [c for c in cleaned_preds.columns if "Close" in c]
    if price_cols:
        cleaned_preds[price_cols] = cleaned_preds[price_cols].clip(lower=0.0)

    # 2. Volatility > Epsilon
    vol_cols = [c for c in cleaned_preds.columns if "Volatility" in c]
    if vol_cols:
        cleaned_preds[vol_cols] = cleaned_preds[vol_cols].clip(lower=0.001)

    # Save
    out_path = Path(cfg["paths"]["outputs_dir"]) / "finance_forecast_asp.parquet"
    cleaned_preds.to_parquet(out_path)
    print(f"Saved ASP-cleaned finance data to {out_path}")
    
    if facts_path.exists(): os.remove(facts_path)

if __name__ == "__main__":
    run_asp("configs/finance_1m.yaml")