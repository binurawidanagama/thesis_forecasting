"""
Modular CNN Baseline (Hardened).
Runs ONE lookback configuration to allow parallel execution.

Usage:
  python scripts/run_cnn_baseline.py --lookback 24
  python scripts/run_cnn_baseline.py --lookback 72
  python scripts/run_cnn_baseline.py --lookback 168
"""

import argparse
import os
import gc
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import backend as K

# ---------------------------
# Config
# ---------------------------
@dataclass
class ExpConfig:
    csv_path: str = "data/processed/AT_engineered.csv"
    time_col: str = "Time (UTC)"
    out_root: str = "artifacts_cnn_split"
    
    # Fast CPU Settings
    batch_size: int = 128
    epochs: int = 20
    filters: int = 32      
    kernel_size: int = 5
    num_blocks: int = 5    
    dilation_base: int = 2 
    dense_units: int = 64
    dropout: float = 0.2
    learning_rate: float = 3e-4
    
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

# ---------------------------
# Helpers
# ---------------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data(cfg):
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found at: {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)
    # Rename to standard internal names
    rename_map = {
        "Actual_Load_MW": "load_mw", "Solar_MW": "solar_mw", "Wind_MW": "wind_mw",
        "Price_EUR_MWh": "price_eur_mwh", "Wind_Cap_MW": "cap_wind_mw", "Solar_Cap_MW": "cap_solar_mw"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Parse Time
    if cfg.time_col not in df.columns:
        cands = [c for c in df.columns if "time" in c.lower()]
        if cands: cfg.time_col = cands[0]
        
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=True)
    df = df.sort_values(cfg.time_col).set_index(cfg.time_col)
    
    # Return numeric only, filled
    return df.select_dtypes(include=[np.number]).ffill().bfill().fillna(0.0)

def fit_scaler(data):
    sc = StandardScaler()
    sc.fit(data)
    return sc

def make_ds(block, lookback, horizon, batch_size, shuffle=True):
    total = lookback + horizon
    if len(block) < total: return None
    
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=block, targets=None, sequence_length=total, sequence_stride=1,
        shuffle=shuffle, batch_size=batch_size
    )
    # Map to (X, Y) split
    return ds.map(lambda w: (w[:, :lookback, :], w[:, lookback:, :]), num_parallel_calls=tf.data.AUTOTUNE)

# ---------------------------
# Model
# ---------------------------
def tcn_block(x, filters, kernel, dilation, dropout):
    res = x
    x = tf.keras.layers.Conv1D(filters, kernel, padding="causal", dilation_rate=dilation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Conv1D(filters, kernel, padding="causal", dilation_rate=dilation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    
    if res.shape[-1] != filters:
        res = tf.keras.layers.Conv1D(filters, 1)(res)
    
    return tf.keras.layers.Add()([x, res])

def build_model(lb, hz, n_feats, cfg):
    inp = tf.keras.Input(shape=(lb, n_feats))
    x = inp
    for i in range(cfg.num_blocks):
        x = tcn_block(x, cfg.filters, cfg.kernel_size, cfg.dilation_base**i, cfg.dropout)
    
    x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)
    x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(hz * n_feats)(x)
    out = tf.keras.layers.Reshape((hz, n_feats))(out)
    
    return tf.keras.Model(inp, out)

# ---------------------------
# Save Preds
# ---------------------------
def save_preds_parquet(model, ds, scaler, cols, timestamps, out_path):
    # Predict in loop to handle large data
    y_pred_list, y_true_list = [], []
    for x, y in ds:
        y_pred_list.append(model.predict(x, verbose=0))
        y_true_list.append(y.numpy())
    
    if not y_pred_list: return

    y_pred = np.concatenate(y_pred_list, axis=0) # [N, H, K]
    y_true = np.concatenate(y_true_list, axis=0)
    
    # Safe Inverse Scale (Explicit Dimensions)
    # scaler.scale_ is (K,). We need to broadcast to (N, H, K).
    scale = scaler.scale_[None, None, :]
    mean  = scaler.mean_[None, None, :]
    
    pred_inv = y_pred * scale + mean
    true_inv = y_true * scale + mean
    
    # Extract NEXT HOUR (step 0) for plotting
    pred_h1 = pred_inv[:, 0, :]
    true_h1 = true_inv[:, 0, :]
    
    # Align Timestamps (Take the last N timestamps)
    n = len(pred_h1)
    if len(timestamps) >= n:
        ts = timestamps[-n:]
    else:
        # Fallback if timestamps are shorter for some reason
        ts = timestamps
        pred_h1 = pred_h1[:len(ts)]
        true_h1 = true_h1[:len(ts)]
    
    data = {"timestamp": ts}
    for i, c in enumerate(cols):
        data[f"True_{c}"] = true_h1[:, i]
        data[f"Pred_{c}"] = pred_h1[:, i]
        
    pd.DataFrame(data).to_parquet(out_path, index=False)

# ---------------------------
# Runner
# ---------------------------
def run_specific_lookback(lookback, cfg):
    print(f"\n[START] Starting job for Lookback = {lookback}...")
    set_seeds(cfg.seed)
    df = load_data(cfg)
    
    # Define Tasks
    tasks = {
        "ENERGY": ["load_mw", "solar_mw", "wind_mw"],
        
        # STRICT 6-Feature Weather Task
        "WEATHER": [
            "temperature_2m_C", 
            "shortwave_radiation_Wm2", 
            "relative_humidity_2m_pct", 
            "precipitation_mm", 
            "wind_speed_100m (m/s)", 
            "surface_pressure_hPa"
        ]
    }
    
    # Split Indices
    n = len(df)
    test_len = int(n * cfg.test_ratio)
    val_len = int(n * cfg.val_ratio)
    train_end = n - val_len - test_len
    val_end = n - test_len
    test_timestamps = df.index[val_end:]
    
    results = []
    
    for task_name, cols in tasks.items():
        print(f"  > Processing Task: {task_name} ({len(cols)} features)")
        
        # Scaling
        data = df[cols].values.astype(np.float32)
        scaler = fit_scaler(data[:train_end])
        data_scaled = scaler.transform(data).astype(np.float32)
        
        # Blocks
        train_blk = data_scaled[:train_end]
        val_blk = data_scaled[max(0, train_end - lookback):val_end]
        test_blk = data_scaled[max(0, val_end - lookback):]
        
        for horizon in [12, 24, 72]:
            print(f"    >> Horizon {horizon}...")
            
            # Datasets
            tr_ds = make_ds(train_blk, lookback, horizon, cfg.batch_size)
            va_ds = make_ds(val_blk, lookback, horizon, cfg.batch_size, shuffle=False)
            te_ds = make_ds(test_blk, lookback, horizon, cfg.batch_size, shuffle=False)
            
            # CRITICAL CHECK: Ensure data exists
            if tr_ds is None or va_ds is None or te_ds is None: 
                print("       [SKIP] Not enough data for this config.")
                continue
                
            # Train
            model = build_model(lookback, horizon, len(cols), cfg)
            model.compile(tf.keras.optimizers.Adam(cfg.learning_rate), "huber", metrics=["mae"])
            
            cb = [tf.keras.callbacks.EarlyStopping("val_loss", patience=4, restore_best_weights=True)]
            
            hist = model.fit(tr_ds, validation_data=va_ds, epochs=cfg.epochs, callbacks=cb, verbose=1)
            
            # Metrics
            val_loss = min(hist.history["val_loss"])
            test_loss = model.evaluate(te_ds, verbose=0)[0]
            
            # Save Predictions
            out_dir = os.path.join(cfg.out_root, f"LB{lookback}_H{horizon}_{task_name}")
            os.makedirs(out_dir, exist_ok=True)
            save_preds_parquet(model, te_ds, scaler, cols, test_timestamps, os.path.join(out_dir, "preds.parquet"))
            
            # Log Result
            res = {
                "task": task_name, "lookback": lookback, "horizon": horizon,
                "val_loss": val_loss, "test_loss_mae_scaled": test_loss
            }
            results.append(res)
            print(f"       [DONE] Val={val_loss:.4f} Test={test_loss:.4f}")
            
            # Cleanup
            K.clear_session()
            gc.collect()

    # Save Summary for this Lookback
    os.makedirs(cfg.out_root, exist_ok=True)
    summary_path = os.path.join(cfg.out_root, f"summary_lb{lookback}.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n[COMPLETE] Saved summary to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, required=True, help="Lookback window size (e.g., 24)")
    args = parser.parse_args()
    
    cfg = ExpConfig()
    run_specific_lookback(args.lookback, cfg)