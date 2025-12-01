import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
import os

def process_kaggle_data():
    raw_dir = Path("data/raw")
    files = list(raw_dir.glob("*.csv"))
    csv_path = None
    for f in files:
        if "bit" in f.name.lower() or "btc" in f.name.lower():
            csv_path = f
            break
    
    if not csv_path:
        print("Error: No Bitcoin CSV found in data/raw/")
        return

    print(f"Processing: {csv_path}")
    print("Filtering for specific 18-month window (2017-01 to 2018-06)...")

    chunk_size = 1_000_000
    chunks = []
    req_cols = ["Timestamp", "Open", "High", "Low", "Close"] 

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        vol_col = "Volume_(BTC)" if "Volume_(BTC)" in chunk.columns else "Volume"
        if vol_col not in chunk.columns: continue
        
        chunk = chunk[req_cols + [vol_col]].copy()
        chunk = chunk.rename(columns={vol_col: "Volume"})
        chunk["Time (UTC)"] = pd.to_datetime(chunk["Timestamp"], unit='s')
        
        # --- STRICT FILTER ---
        # Start: Jan 1, 2017
        # End:   June 30, 2018 (18 Months total)
        mask = (chunk["Time (UTC)"] >= "2017-01-01") & (chunk["Time (UTC)"] <= "2018-06-30")
        
        if mask.any():
            chunks.append(chunk[mask])

    if not chunks:
        print("Error: No data found in 2017-2018 range.")
        return

    df = pd.concat(chunks)
    df = df.set_index("Time (UTC)").sort_index()
    df = df.drop(columns=["Timestamp"])

    print("Resampling to continuous 1-min grid...")
    df = df.resample("1min").ffill()
    df = df.dropna()

    print("Calculating Technical Indicators...")
    df["RSI"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["Volatility"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    
    minutes = df.index.minute
    hours = df.index.hour
    df["minute_sin"] = np.sin(2 * np.pi * minutes / 60)
    df["minute_cos"] = np.cos(2 * np.pi * minutes / 60)
    df["hour_sin"]   = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hours / 24)

    df = df.dropna()
    
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Finance_1m.csv"
    
    df.to_csv(out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    print("Time Range:", df.index.min(), "to", df.index.max())

if __name__ == "__main__":
    process_kaggle_data()