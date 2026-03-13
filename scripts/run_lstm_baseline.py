"""
Modular LSTM Baseline (Visual Mode).
Runs ONE lookback configuration with full progress bars.
Saves predictions for visualization.

Usage:
  python scripts/run_lstm_baseline.py --lookback 24
  python scripts/run_lstm_baseline.py --lookback 72
  python scripts/run_lstm_baseline.py --lookback 168

Outputs:
  - artifacts_lstm_baseline/summary_lb{lookback}.csv
  - artifacts_lstm_baseline/LB{lookback}_H{horizon}_{task}/preds.parquet
    (ALL horizon steps, wide format)
"""

import argparse
import os
import gc
import time
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K

# ---------------------------
# Optional: process metrics (psutil)
# ---------------------------
try:
    import psutil  # pip install psutil
    _PSUTIL_OK = True
    _PROC = psutil.Process(os.getpid())

    def get_process_metrics():
        """
        Returns:
          rss_mb: current resident set size in MB
          cpu_s : cumulative CPU seconds consumed by this process (user + system)
        """
        with _PROC.oneshot():
            rss_mb = _PROC.memory_info().rss / (1024 * 1024)
            t = _PROC.cpu_times()
            cpu_s = float(t.user + t.system)
        return float(rss_mb), float(cpu_s)

except Exception:
    _PSUTIL_OK = False

    def get_process_metrics():
        return float("nan"), float("nan")


# ---------------------------
# Bench headers (MATCH others)
# ---------------------------
BASE_HEADER = [
    "task", "lookback", "horizon",
    "MAE", "RMSE", "sMAPE",
    "BASE_MAE", "BASE_RMSE", "BASE_sMAPE",
    "Params",
    "Train_Wall_Sec", "Train_CPU_Sec", "Avg_CPU_Usage_Pct",
    "Peak_RAM_MB",
    "Infer_Wall_Sec", "Infer_CPU_Sec", "Infer_Avg_CPU_Pct",
    "Latency_ms_per_sample",
    "Size_MB"
]

EXTRA_HEADER = [
    "Train_Params",
    "Deploy_Params",
    "Train_Size_MB",
    "Deploy_Size_MB"
]

HEADER_V2 = BASE_HEADER + EXTRA_HEADER


# ---------------------------
# Config
# ---------------------------
@dataclass
class ExpConfig:
    csv_path: str = "data/processed/AT_engineered.csv"
    time_col: str = "Time (UTC)"
    out_root: str = "artifacts_lstm_baseline"

    # Strict Date Splits (Matches your dCeNN configs)
    train_until: str = "2020-12-31 23:00:00"
    val_until:   str = "2021-12-31 23:00:00"
    test_until:  str = "2022-12-31 23:00:00"

    # LSTM Settings (CPU-friendly)
    batch_size: int = 256
    epochs: int = 20
    lstm_units: int = 128
    lstm_units_2: int = 64
    dense_units: int = 128
    dropout: float = 0.2
    learning_rate: float = 3e-4
    clipnorm: float = 1.0
    seed: int = 42

    # Peak RAM sampling during training
    ram_sample_every_n_batches: int = 10  # sample RAM every N train batches

    # Dataset caching (keep False if you want RAM to reflect runtime more than dataset cache)
    cache_val_test: bool = False


# ---------------------------
# Peak RAM callback (TRAIN)
# ---------------------------
class PeakRAM(tf.keras.callbacks.Callback):
    """
    Samples process RSS during training to estimate peak training RSS.
    We'll later take max() with inference start/end RSS to get end-to-end Peak_RAM_MB.
    """
    def __init__(self, sample_every_n_batches: int = 10):
        super().__init__()
        self.sample_every_n_batches = max(1, int(sample_every_n_batches))
        self.peak_mb = float("nan")
        self._b = 0

    def on_train_begin(self, logs=None):
        self._b = 0
        rss, _ = get_process_metrics()
        self.peak_mb = rss

    def on_train_batch_end(self, batch, logs=None):
        if not _PSUTIL_OK:
            return
        self._b += 1
        if self._b % self.sample_every_n_batches == 0:
            rss, _ = get_process_metrics()
            if np.isfinite(rss):
                self.peak_mb = max(float(self.peak_mb), float(rss))

    def on_train_end(self, logs=None):
        rss, _ = get_process_metrics()
        if np.isfinite(rss):
            self.peak_mb = max(float(self.peak_mb), float(rss))


# ---------------------------
# Helpers
# ---------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_cpu_pct(cpu_s: float, wall_s: float) -> float:
    if not (np.isfinite(cpu_s) and np.isfinite(wall_s)) or wall_s <= 0:
        return float("nan")
    return float(100.0 * (cpu_s / wall_s))


def update_peak(peak: float, val: float) -> float:
    if not np.isfinite(val):
        return peak
    if not np.isfinite(peak):
        return val
    return max(float(peak), float(val))


def load_data(cfg: ExpConfig) -> pd.DataFrame:
    """Loads CSV, standardizes names, parses UTC datetime index, keeps numeric+bool."""
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found at: {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)

    rename_map = {
        "Actual_Load_MW": "load_mw",
        "Solar_MW": "solar_mw",
        "Wind_MW": "wind_mw",
        "Price_EUR_MWh": "price_eur_mwh",
        "Wind_Cap_MW": "cap_wind_mw",
        "Solar_Cap_MW": "cap_solar_mw",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if cfg.time_col not in df.columns:
        candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if candidates:
            cfg.time_col = candidates[0]
        else:
            raise ValueError(f"time_col '{cfg.time_col}' not found, and no time-like column detected.")

    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).set_index(cfg.time_col)

    df_num = df.select_dtypes(include=[np.number, "bool"]).astype(np.float32)
    df_num = df_num.ffill().bfill()
    if df_num.isna().any().any():
        df_num = df_num.fillna(0.0)
    return df_num


def get_indices_by_date(df: pd.DataFrame, cfg: ExpConfig):
    """Inclusive split indices via side='right' in UTC."""
    train_end = pd.Timestamp(cfg.train_until).tz_localize("UTC")
    val_end   = pd.Timestamp(cfg.val_until).tz_localize("UTC")
    test_end  = pd.Timestamp(cfg.test_until).tz_localize("UTC")

    train_idx = df.index.searchsorted(train_end, side="right")
    val_idx   = df.index.searchsorted(val_end,   side="right")
    test_idx  = df.index.searchsorted(test_end,  side="right")
    test_idx  = min(test_idx, len(df))

    if not (0 < train_idx < val_idx <= test_idx):
        raise ValueError(
            f"Bad split indices. train={train_idx}, val={val_idx}, test={test_idx}, len={len(df)}. "
            f"Check that your split timestamps exist within the CSV range."
        )
    return train_idx, val_idx, test_idx


def fit_scaler(train_2d: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(train_2d)
    return sc


def make_ds(
    block: np.ndarray,
    lookback: int,
    horizon: int,
    batch_size: int,
    target_indices: np.ndarray,
    shuffle: bool = True,
    stride: int = 1,
    cache: bool = False,
):
    """
    Dataset:
      X: [batch, lookback, n_features]
      Y: [batch, horizon, n_targets]   (targets are gathered from feature columns)
    """
    total = lookback + horizon
    if len(block) < total:
        return None

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=block,
        targets=None,
        sequence_length=total,
        sequence_stride=stride,
        shuffle=shuffle,
        batch_size=batch_size,
    )

    idx = tf.constant(target_indices, dtype=tf.int32)

    def split_window(w):
        x = w[:, :lookback, :]
        y = w[:, lookback:, :]
        y = tf.gather(y, idx, axis=2)
        return x, y

    ds = ds.map(split_window, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------
# Scaling + Metrics
# ---------------------------
def inverse_transform_targets(y_scaled: np.ndarray, scaler: StandardScaler, target_indices: np.ndarray) -> np.ndarray:
    """Inverse-transform ONLY the target columns using the *main* scaler's params."""
    scale = scaler.scale_[target_indices].astype(np.float32)
    mean  = scaler.mean_[target_indices].astype(np.float32)

    if y_scaled.ndim == 2:   # [N, C]
        return y_scaled * scale[None, :] + mean[None, :]
    if y_scaled.ndim == 3:   # [N, H, C]
        return y_scaled * scale[None, None, :] + mean[None, None, :]
    raise ValueError(f"Unexpected y_scaled shape: {y_scaled.shape}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    smape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
    return mae, rmse, smape


def persistence_baseline_from_inputs(x_scaled: np.ndarray, target_indices: np.ndarray, horizon: int) -> np.ndarray:
    """Persistence baseline: predict future = last observed target value in input window."""
    last_step = x_scaled[:, -1, :]                   # [N, n_features]
    last_targets = last_step[:, target_indices]      # [N, n_targets]
    return np.repeat(last_targets[:, None, :], repeats=horizon, axis=1)


# ---------------------------
# Model (LSTM Seq2Seq)
# ---------------------------
def build_lstm_seq2seq(lookback: int, horizon: int, n_in_feats: int, n_out_feats: int, cfg: ExpConfig) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(lookback, n_in_feats))

    # Encoder
    x = tf.keras.layers.LSTM(cfg.lstm_units, return_sequences=False)(inp)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)

    # Bridge
    x = tf.keras.layers.RepeatVector(horizon)(x)

    # Decoder
    x = tf.keras.layers.LSTM(cfg.lstm_units_2, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)

    # Head
    x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_out_feats, dtype="float32"))(x)

    return tf.keras.Model(inp, out, name=f"lstm_s2s_lb{lookback}_h{horizon}")


# ---------------------------
# Save Preds
# ---------------------------
def save_preds_parquet(
    y_pred_inv: np.ndarray,
    y_true_inv: np.ndarray,
    cols,
    base_timestamps,
    out_path: str,
):
    """
    Save ALL horizon steps in WIDE format.

    Semantics:
      - One row = one forecast window
      - 'timestamp' = target timestamp for horizon step h1
      - Columns are named:
            True_<col>+h1, Pred_<col>+h1, ... True_<col>+hH, Pred_<col>+hH

    Backward-compatibility aliases:
            True_<col>, Pred_<col>
      are also saved as h1.
    """
    if y_pred_inv.ndim != 3 or y_true_inv.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays [N, H, C], got pred={y_pred_inv.shape}, true={y_true_inv.shape}"
        )

    n_samples, horizon, n_targets = y_pred_inv.shape

    if y_true_inv.shape != y_pred_inv.shape:
        raise ValueError(f"Shape mismatch: pred={y_pred_inv.shape}, true={y_true_inv.shape}")
    if len(cols) != n_targets:
        raise ValueError(f"Target columns mismatch: len(cols)={len(cols)} vs n_targets={n_targets}")
    if len(base_timestamps) != n_samples:
        raise AssertionError(f"Timestamp mismatch: TS={len(base_timestamps)} vs Pred={n_samples}")

    base_timestamps = pd.to_datetime(base_timestamps, utc=True)

    data = {"timestamp": base_timestamps}

    for i, c in enumerate(cols):
        # Backward-compatible aliases for h1
        data[f"True_{c}"] = y_true_inv[:, 0, i]
        data[f"Pred_{c}"] = y_pred_inv[:, 0, i]

        # Full horizon columns
        for h in range(horizon):
            h_label = h + 1
            data[f"True_{c}+h{h_label}"] = y_true_inv[:, h, i]
            data[f"Pred_{c}+h{h_label}"] = y_pred_inv[:, h, i]

    pd.DataFrame(data).to_parquet(out_path, index=False)


def append_row_schema_safe(summary_csv: str, row_dict: dict):
    """Auto-upgrade schema to HEADER_V2 and append."""
    df_row = pd.DataFrame([[row_dict.get(h, np.nan) for h in HEADER_V2]], columns=HEADER_V2)

    if os.path.exists(summary_csv):
        df_existing = pd.read_csv(summary_csv)
        for col in HEADER_V2:
            if col not in df_existing.columns:
                df_existing[col] = np.nan
        df_existing = df_existing[HEADER_V2]
        df_final = pd.concat([df_existing, df_row], ignore_index=True)
        df_final.to_csv(summary_csv, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)


# ---------------------------
# Runner
# ---------------------------
def run_specific_lookback(lookback: int, cfg: ExpConfig) -> None:
    print(f"\n[START] LSTM baseline | Lookback={lookback}")
    set_seeds(cfg.seed)
    os.makedirs(cfg.out_root, exist_ok=True)

    peak_ram_mb = float("nan")
    rss0, _ = get_process_metrics()
    peak_ram_mb = update_peak(peak_ram_mb, rss0)

    df = load_data(cfg)
    tr_idx, va_idx, te_idx = get_indices_by_date(df, cfg)
    print(f"Split indices: Train={tr_idx}, Val={va_idx}, Test={te_idx} (len={len(df)})")
    if _PSUTIL_OK:
        print("[INFO] psutil enabled: CPU-seconds + peak RAM will be recorded.")
    else:
        print("[WARN] psutil not available: CPU/RAM metrics will be NaN. Install with: pip install psutil")

    potential_drivers = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_public_holiday"]
    drivers = [c for c in potential_drivers if c in df.columns]

    tasks = {
        "ENERGY": {
            "targets": ["load_mw", "solar_mw", "wind_mw"],
            "features": [
                "load_mw", "solar_mw", "wind_mw",
                "temperature_2m_C", "shortwave_radiation_Wm2",
                "wind_speed_100m (m/s)", "precipitation_mm",
            ] + drivers,
        },
        "WEATHER": {
            "targets": [
                "temperature_2m_C", "shortwave_radiation_Wm2",
                "relative_humidity_2m_pct", "precipitation_mm",
                "wind_speed_100m (m/s)", "surface_pressure_hPa",
            ],
            "features": [
                "temperature_2m_C", "shortwave_radiation_Wm2",
                "relative_humidity_2m_pct", "precipitation_mm",
                "wind_speed_100m (m/s)", "surface_pressure_hPa",
            ] + drivers,
        },
    }

    horizons = [12, 24, 72]
    results = []

    for task_name, spec in tasks.items():
        feat_cols = spec["features"]
        target_cols = spec["targets"]

        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            print(f"[SKIP] {task_name}: Missing columns: {missing}")
            continue

        print(f"\n  > Task: {task_name} | Inputs={len(feat_cols)} | Targets={len(target_cols)}")

        data = df[feat_cols].values.astype(np.float32)
        target_indices = np.array([feat_cols.index(c) for c in target_cols], dtype=np.int32)

        scaler = fit_scaler(data[:tr_idx])
        data_scaled = scaler.transform(data).astype(np.float32)

        train_blk = data_scaled[:tr_idx]
        val_blk   = data_scaled[max(0, tr_idx - lookback):va_idx]
        test_blk  = data_scaled[max(0, va_idx - lookback):te_idx]

        rss_tmp, _ = get_process_metrics()
        peak_ram_mb = update_peak(peak_ram_mb, rss_tmp)

        for horizon in horizons:
            print(f"    >> Horizon={horizon}")

            tr_ds = make_ds(
                train_blk, lookback, horizon, cfg.batch_size, target_indices,
                shuffle=True, cache=False
            )
            va_ds = make_ds(
                val_blk, lookback, horizon, cfg.batch_size, target_indices,
                shuffle=False, cache=cfg.cache_val_test
            )
            te_ds = make_ds(
                test_blk, lookback, horizon, cfg.batch_size, target_indices,
                shuffle=False, cache=cfg.cache_val_test
            )

            if tr_ds is None or va_ds is None or te_ds is None:
                print("       [SKIP] Not enough data for this (lookback+horizon).")
                continue

            model = build_lstm_seq2seq(lookback, horizon, len(feat_cols), len(target_cols), cfg)
            opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm)
            model.compile(optimizer=opt, loss=tf.keras.losses.Huber(), metrics=["mae"])
            n_params = int(model.count_params())

            early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            peak_cb = PeakRAM(sample_every_n_batches=cfg.ram_sample_every_n_batches)

            # -----------------------
            # TRAIN: wall + CPU + peak RAM
            # -----------------------
            rss_a, cpu_a = get_process_metrics()
            peak_ram_mb = update_peak(peak_ram_mb, rss_a)

            t0 = time.time()
            model.fit(
                tr_ds,
                validation_data=va_ds,
                epochs=cfg.epochs,
                callbacks=[early, peak_cb],
                verbose=1
            )
            train_wall = float(time.time() - t0)

            rss_b, cpu_b = get_process_metrics()
            peak_ram_mb = update_peak(peak_ram_mb, rss_b)
            peak_ram_mb = update_peak(peak_ram_mb, peak_cb.peak_mb)

            train_cpu = float(cpu_b - cpu_a) if np.isfinite(cpu_a) and np.isfinite(cpu_b) else float("nan")
            avg_cpu_pct = safe_cpu_pct(train_cpu, train_wall)

            # -----------------------
            # INFER: wall + CPU (+ update Peak_RAM_MB via max(start/end))
            # -----------------------
            _ = model.predict(te_ds.take(1), verbose=0)  # warmup (not timed)

            inf_rss0, inf_cpu0 = get_process_metrics()
            peak_ram_mb = update_peak(peak_ram_mb, inf_rss0)

            t0 = time.time()
            y_pred_scaled = model.predict(te_ds, verbose=0)
            infer_wall = float(time.time() - t0)

            inf_rss1, inf_cpu1 = get_process_metrics()
            peak_ram_mb = update_peak(peak_ram_mb, inf_rss1)

            infer_cpu = float(inf_cpu1 - inf_cpu0) if np.isfinite(inf_cpu0) and np.isfinite(inf_cpu1) else float("nan")
            infer_avg_cpu_pct = safe_cpu_pct(infer_cpu, infer_wall)

            y_true_scaled = np.concatenate([y for _, y in te_ds], axis=0)

            pred_inv = inverse_transform_targets(y_pred_scaled, scaler, target_indices)
            true_inv = inverse_transform_targets(y_true_scaled, scaler, target_indices)
            mae, rmse, smape = calculate_metrics(true_inv, pred_inv)

            x_all = np.concatenate([x for x, _ in te_ds], axis=0)
            base_scaled = persistence_baseline_from_inputs(x_all, target_indices, horizon)
            base_inv = inverse_transform_targets(base_scaled, scaler, target_indices)
            base_mae, base_rmse, base_smape = calculate_metrics(true_inv, base_inv)

            n_samples = int(pred_inv.shape[0])
            latency_ms = float((infer_wall * 1000.0) / max(1, n_samples))

            temp_name = f"temp_{task_name}_lb{lookback}_h{horizon}_{os.getpid()}_{int(time.time()*1e6)}.keras"
            model_path = os.path.join(cfg.out_root, temp_name)
            model.save(model_path, include_optimizer=False)
            size_mb = float(os.path.getsize(model_path) / (1024 ** 2))
            try:
                os.remove(model_path)
            except OSError:
                pass

            # Timestamp alignment: timestamp corresponds to horizon step h1
            test_start_idx = max(0, va_idx - lookback)
            start_ts_idx = test_start_idx + lookback
            n_windows = len(test_blk) - (lookback + horizon) + 1
            ts = df.index[start_ts_idx: start_ts_idx + n_windows]

            if len(ts) != n_samples:
                raise AssertionError(
                    f"TS/pred mismatch: TS={len(ts)} vs Pred={n_samples} | {task_name} LB={lookback} H={horizon} "
                    f"(test_blk={len(test_blk)}, start_ts_idx={start_ts_idx})"
                )

            # Save preds parquet (all horizons, wide format)
            out_dir = os.path.join(cfg.out_root, f"LB{lookback}_H{horizon}_{task_name}")
            os.makedirs(out_dir, exist_ok=True)
            save_preds_parquet(pred_inv, true_inv, target_cols, ts, os.path.join(out_dir, "preds.parquet"))

            Train_Params = int(n_params)
            Deploy_Params = int(n_params)
            Train_Size_MB = float(size_mb)
            Deploy_Size_MB = float(size_mb)

            Params = int(Deploy_Params)
            Size_MB = float(Deploy_Size_MB)

            Peak_RAM_MB = float(peak_ram_mb) if np.isfinite(peak_ram_mb) else float("nan")

            results.append({
                "task": task_name,
                "lookback": int(lookback),
                "horizon": int(horizon),

                "MAE": float(mae),
                "RMSE": float(rmse),
                "sMAPE": float(smape),

                "BASE_MAE": float(base_mae),
                "BASE_RMSE": float(base_rmse),
                "BASE_sMAPE": float(base_smape),

                "Params": int(Params),

                "Train_Wall_Sec": float(train_wall),
                "Train_CPU_Sec": float(train_cpu),
                "Avg_CPU_Usage_Pct": float(avg_cpu_pct),

                "Peak_RAM_MB": float(Peak_RAM_MB),

                "Infer_Wall_Sec": float(infer_wall),
                "Infer_CPU_Sec": float(infer_cpu),
                "Infer_Avg_CPU_Pct": float(infer_avg_cpu_pct),

                "Latency_ms_per_sample": float(latency_ms),
                "Size_MB": float(Size_MB),

                "Train_Params": int(Train_Params),
                "Deploy_Params": int(Deploy_Params),
                "Train_Size_MB": float(Train_Size_MB),
                "Deploy_Size_MB": float(Deploy_Size_MB),
            })

            print(
                f"       [RES] MAE={mae:.4f} | BASE_MAE={base_mae:.4f} | "
                f"Latency={latency_ms:.4f}ms | Size={size_mb:.2f}MB | "
                f"CPU(train avg)={avg_cpu_pct:.1f}% | RAM(peak e2e)={Peak_RAM_MB:.0f}MB | Params={n_params}"
            )

            K.clear_session()
            del model
            gc.collect()

    summary_path = os.path.join(cfg.out_root, f"summary_lb{lookback}.csv")
    for r in results:
        append_row_schema_safe(summary_path, r)

    print(f"\n[COMPLETE] Saved/updated summary -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, required=True, help="Lookback window size (e.g., 24, 72, 168)")
    args = parser.parse_args()

    run_specific_lookback(args.lookback, ExpConfig())