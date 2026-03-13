"""
Modular CNN Baseline (Thesis Grade - CPU/RAM Benchmarked).
------------------------------------------------------------------------------
DESCRIPTION:
  Trains a TCN (Temporal Convolutional Network) for multivariate forecasting.
  Benchmarks against dCeNN-ELM-ASP using strict date splits and identical inputs.

  GUARANTEES:
  - Strict UTC date splits with side="right" (inclusive boundaries)
  - Multivariate inputs (features) -> Target-only outputs
  - Train-only scaling (no leakage)
  - Robust inverse scaling using the SAME scaler (feature-wise)
  - Correct timestamp alignment for saved predictions (hard assert)
  - Predictions saved in WIDE format for compatibility with plotting scripts
  - Full benchmarking:
      * Accuracy: MAE, RMSE, sMAPE
      * Baseline: Persistence MAE, RMSE, sMAPE
      * Efficiency: Params, Train_Wall_Sec, Train_CPU_Sec, Avg_CPU_Usage_Pct,
                    Peak_RAM_MB, Infer_Wall_Sec, Infer_CPU_Sec, Infer_Avg_CPU_Pct,
                    Latency_ms_per_sample, Size_MB

USAGE:
  python scripts/run_cnn_baseline.py --lookback 24
  python scripts/run_cnn_baseline.py --lookback 72
  python scripts/run_cnn_baseline.py --lookback 168

OUTPUTS:
  - artifacts_cnn_baseline/summary_lb{lookback}.csv
  - artifacts_cnn_baseline/LB{lookback}_H{horizon}_{task}/preds.parquet
    (ALL horizon steps saved in WIDE format)
------------------------------------------------------------------------------
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

# Optional but recommended for CPU/RAM benchmarking
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False


# ---------------------------
# Config
# ---------------------------
@dataclass
class ExpConfig:
    csv_path: str = "data/processed/AT_engineered.csv"
    time_col: str = "Time (UTC)"
    out_root: str = "artifacts_cnn_baseline"

    # Strict Date Splits (Matches dCeNN Default.yaml)
    train_until: str = "2020-12-31 23:00:00"
    val_until:   str = "2021-12-31 23:00:00"
    test_until:  str = "2022-12-31 23:00:00"

    # CNN/TCN Settings (CPU-friendly)
    batch_size: int = 128
    epochs: int = 20
    filters: int = 32
    kernel_size: int = 5
    num_blocks: int = 5
    dilation_base: int = 2
    dense_units: int = 64
    dropout: float = 0.2
    learning_rate: float = 3e-4
    seed: int = 42

    # Resource sampling
    ram_sample_every_n_batches: int = 10
    infer_ram_sample_every_n_batches: int = 25  # optional inference RAM sampling cadence (0 disables)


# ---------------------------
# Process Metrics (CPU seconds + RAM MB)
# ---------------------------
def get_process_metrics():
    """
    Returns (ram_mb, cpu_time_s) for this process.
    cpu_time_s = user + system CPU seconds (cumulative, all threads).
    """
    if not _HAS_PSUTIL:
        return float("nan"), float("nan")
    p = psutil.Process(os.getpid())
    with p.oneshot():
        mem_mb = p.memory_info().rss / (1024 * 1024)
        cpu = p.cpu_times()
        cpu_time_s = float(cpu.user + cpu.system)
    return float(mem_mb), float(cpu_time_s)


def _nanmax(*vals: float) -> float:
    """max that ignores NaNs; returns NaN if all are NaN."""
    good = [v for v in vals if np.isfinite(v)]
    return float(max(good)) if good else float("nan")


class ResourceMonitor(tf.keras.callbacks.Callback):
    """
    Samples RAM during training to estimate true peak RAM (process RSS).
    """
    def __init__(self, sample_every_n_batches: int = 10):
        super().__init__()
        self.sample_every_n_batches = max(1, int(sample_every_n_batches))
        self.peak_ram_mb = float("nan")
        self.start_ram_mb = float("nan")
        self._batch = 0

    def on_train_begin(self, logs=None):
        self._batch = 0
        ram, _ = get_process_metrics()
        self.start_ram_mb = ram
        self.peak_ram_mb = ram

    def on_train_batch_end(self, batch, logs=None):
        if not _HAS_PSUTIL:
            return
        self._batch += 1
        if self._batch % self.sample_every_n_batches == 0:
            ram, _ = get_process_metrics()
            self.peak_ram_mb = _nanmax(self.peak_ram_mb, ram)

    def on_train_end(self, logs=None):
        ram, _ = get_process_metrics()
        self.peak_ram_mb = _nanmax(self.peak_ram_mb, ram)


# ---------------------------
# Helpers
# ---------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
        cands = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if cands:
            cfg.time_col = cands[0]
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
            f"Check split timestamps exist within CSV range."
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
):
    """
    Dataset:
      X: [B, lookback, n_features]
      Y: [B, horizon, n_targets]
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

    return ds.map(split_window, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# ---------------------------
# Scaling + Metrics
# ---------------------------
def inverse_transform_targets(y_scaled: np.ndarray, scaler: StandardScaler, target_indices: np.ndarray) -> np.ndarray:
    """Robust inverse scaling for both 2D and 3D arrays."""
    scale = scaler.scale_[target_indices].astype(np.float32)
    mean  = scaler.mean_[target_indices].astype(np.float32)

    if y_scaled.ndim == 2:  # [N, C]
        return y_scaled * scale[None, :] + mean[None, :]
    if y_scaled.ndim == 3:  # [N, H, C]
        return y_scaled * scale[None, None, :] + mean[None, None, :]
    raise ValueError(f"Unexpected y_scaled shape: {y_scaled.shape}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    smape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
    return mae, rmse, smape


def persistence_baseline_from_inputs(x_scaled: np.ndarray, target_indices: np.ndarray, horizon: int) -> np.ndarray:
    """
    Persistence baseline: predict future = last observed target value in input window.
    x_scaled: [N, lookback, n_features]
    returns:  [N, horizon, n_targets]
    """
    last_step = x_scaled[:, -1, :]
    last_targets = last_step[:, target_indices]
    return np.repeat(last_targets[:, None, :], repeats=horizon, axis=1)


# ---------------------------
# Model (TCN)
# ---------------------------
def tcn_block(x, filters: int, kernel: int, dilation: int, dropout: float):
    res = x
    x = tf.keras.layers.Conv1D(filters, kernel, padding="causal", dilation_rate=dilation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(filters, kernel, padding="causal", dilation_rate=dilation)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    if res.shape[-1] != filters:
        res = tf.keras.layers.Conv1D(filters, 1)(res)

    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_model(lb: int, hz: int, n_in_feats: int, n_out_feats: int, cfg: ExpConfig) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(lb, n_in_feats))
    x = inp
    for i in range(cfg.num_blocks):
        x = tcn_block(x, cfg.filters, cfg.kernel_size, cfg.dilation_base**i, cfg.dropout)

    x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)
    x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(hz * n_out_feats)(x)
    out = tf.keras.layers.Reshape((hz, n_out_feats))(out)
    return tf.keras.Model(inp, out, name=f"tcn_lb{lb}_h{hz}")


# ---------------------------
# Save Preds
# ---------------------------
def save_preds_parquet(
    y_pred_inv: np.ndarray,
    y_true_inv: np.ndarray,
    cols,
    base_timestamps,
    out_path: str,
) -> None:
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
        raise ValueError(f"TS/pred mismatch: {len(base_timestamps)} vs {n_samples}")

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

    out_df = pd.DataFrame(data)
    out_df.to_parquet(out_path, index=False)


# ---------------------------
# Runner
# ---------------------------
def run_specific_lookback(lookback: int, cfg: ExpConfig) -> None:
    print(f"\n[START] CNN/TCN baseline | Lookback={lookback}")
    set_seeds(cfg.seed)

    os.makedirs(cfg.out_root, exist_ok=True)

    df = load_data(cfg)
    tr_idx, va_idx, te_idx = get_indices_by_date(df, cfg)
    print(f"Split indices: Train={tr_idx}, Val={va_idx}, Test={te_idx} (len={len(df)})")

    POTENTIAL_DRIVERS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_public_holiday"]
    drivers = [c for c in POTENTIAL_DRIVERS if c in df.columns]

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

    if not _HAS_PSUTIL:
        print("[WARN] psutil not installed -> CPU/RAM benchmarking will be NaN. Install with: pip install psutil")

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

        for horizon in horizons:
            print(f"    >> Horizon={horizon}")

            tr_ds = make_ds(train_blk, lookback, horizon, cfg.batch_size, target_indices, shuffle=True)
            va_ds = make_ds(val_blk,   lookback, horizon, cfg.batch_size, target_indices, shuffle=False)
            te_ds = make_ds(test_blk,  lookback, horizon, cfg.batch_size, target_indices, shuffle=False)

            if tr_ds is None or va_ds is None or te_ds is None:
                print("       [SKIP] Not enough data for this (lookback+horizon).")
                continue

            model = build_model(lookback, horizon, len(feat_cols), len(target_cols), cfg)
            opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=1.0)
            model.compile(optimizer=opt, loss=tf.keras.losses.Huber(), metrics=["mae"])

            n_params = int(model.count_params())

            cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
            resmon = ResourceMonitor(sample_every_n_batches=cfg.ram_sample_every_n_batches)

            # --- TRAIN BENCHMARK (Wall + CPU seconds + Peak RAM training) ---
            train_start_ram, train_start_cpu = get_process_metrics()
            t0 = time.time()
            model.fit(tr_ds, validation_data=va_ds, epochs=cfg.epochs, callbacks=cb + [resmon], verbose=1)
            train_wall = float(time.time() - t0)
            train_end_ram, train_end_cpu = get_process_metrics()

            train_cpu = float(train_end_cpu - train_start_cpu) if np.isfinite(train_start_cpu) and np.isfinite(train_end_cpu) else float("nan")
            avg_cpu_pct = float((train_cpu / train_wall) * 100.0) if (train_wall > 0 and np.isfinite(train_cpu)) else float("nan")

            # Start with best estimate from training callback + endpoints
            peak_ram = _nanmax(resmon.peak_ram_mb, train_start_ram, train_end_ram)

            # --- INFERENCE BENCHMARK (Wall + CPU seconds + RAM endpoints + optional sampling) ---
            _ = model.predict(te_ds.take(1), verbose=0)  # warmup

            inf_start_ram, inf_start_cpu = get_process_metrics()
            t0 = time.time()

            infer_peak_ram = inf_start_ram
            y_pred_scaled_list = []
            sample_every = int(cfg.infer_ram_sample_every_n_batches)
            sample_every = max(0, sample_every)

            if sample_every == 0 or not _HAS_PSUTIL:
                y_pred_scaled = model.predict(te_ds, verbose=0)
            else:
                b = 0
                for xb, _ in te_ds:
                    yb = model(xb, training=False).numpy()
                    y_pred_scaled_list.append(yb)
                    b += 1
                    if (b % sample_every) == 0:
                        ram, _ = get_process_metrics()
                        infer_peak_ram = _nanmax(infer_peak_ram, ram)
                y_pred_scaled = np.concatenate(y_pred_scaled_list, axis=0)

            infer_wall = float(time.time() - t0)
            inf_end_ram, inf_end_cpu = get_process_metrics()

            infer_cpu = float(inf_end_cpu - inf_start_cpu) if np.isfinite(inf_start_cpu) and np.isfinite(inf_end_cpu) else float("nan")
            infer_avg_cpu_pct = float((infer_cpu / infer_wall) * 100.0) if (infer_wall > 0 and np.isfinite(infer_cpu)) else float("nan")

            # END-TO-END Peak RAM (train + inference)
            peak_ram = _nanmax(peak_ram, inf_start_ram, inf_end_ram, infer_peak_ram)

            # True scaled values
            y_true_scaled = np.concatenate([y for _, y in te_ds], axis=0)

            # Inverse scale for metrics
            pred_inv = inverse_transform_targets(y_pred_scaled, scaler, target_indices)
            true_inv = inverse_transform_targets(y_true_scaled, scaler, target_indices)
            mae, rmse, smape = calculate_metrics(true_inv, pred_inv)

            # Persistence baseline
            x_all = np.concatenate([x for x, _ in te_ds], axis=0)
            base_scaled = persistence_baseline_from_inputs(x_all, target_indices, horizon=horizon)
            base_inv = inverse_transform_targets(base_scaled, scaler, target_indices)
            base_mae, base_rmse, base_smape = calculate_metrics(true_inv, base_inv)

            # Latency (ms/sample) per forecast window
            n_samples = int(pred_inv.shape[0])
            latency_ms = float((infer_wall * 1000.0) / max(1, n_samples))

            # Model size (no optimizer), unique temp name to avoid collisions
            temp_name = f"temp_{task_name}_lb{lookback}_h{horizon}_{os.getpid()}_{int(time.time()*1e6)}.keras"
            model_path = os.path.join(cfg.out_root, temp_name)
            model.save(model_path, include_optimizer=False)
            size_mb = float(os.path.getsize(model_path) / (1024 ** 2))
            try:
                os.remove(model_path)
            except OSError:
                pass

            # Timestamp alignment:
            # base timestamp corresponds to horizon step h1
            test_start_idx = max(0, va_idx - lookback)
            start_ts_idx = test_start_idx + lookback
            n_windows = len(test_blk) - (lookback + horizon) + 1
            ts = df.index[start_ts_idx : start_ts_idx + n_windows]

            if len(ts) != n_samples:
                raise AssertionError(
                    f"TS/pred mismatch: TS={len(ts)} vs Pred={n_samples} | {task_name} LB={lookback} H={horizon} "
                    f"(test_blk={len(test_blk)}, start_ts_idx={start_ts_idx})"
                )

            # Save all horizons in wide format
            out_dir = os.path.join(cfg.out_root, f"LB{lookback}_H{horizon}_{task_name}")
            os.makedirs(out_dir, exist_ok=True)
            save_preds_parquet(pred_inv, true_inv, target_cols, ts, os.path.join(out_dir, "preds.parquet"))

            results.append({
                "task": task_name,
                "lookback": lookback,
                "horizon": horizon,

                "MAE": mae,
                "RMSE": rmse,
                "sMAPE": smape,

                "BASE_MAE": base_mae,
                "BASE_RMSE": base_rmse,
                "BASE_sMAPE": base_smape,

                "Params": n_params,
                "Train_Params": n_params,
                "Deploy_Params": n_params,
                "Train_Size_MB": size_mb,
                "Deploy_Size_MB": size_mb,

                "Train_Wall_Sec": train_wall,
                "Train_CPU_Sec": train_cpu,
                "Avg_CPU_Usage_Pct": avg_cpu_pct,
                "Peak_RAM_MB": peak_ram,

                "Infer_Wall_Sec": infer_wall,
                "Infer_CPU_Sec": infer_cpu,
                "Infer_Avg_CPU_Pct": infer_avg_cpu_pct,

                "Latency_ms_per_sample": latency_ms,
                "Size_MB": size_mb,
            })

            print(
                f"       [RES] MAE={mae:.4f} | BASE_MAE={base_mae:.4f} | "
                f"Latency={latency_ms:.4f}ms | Size={size_mb:.2f}MB | "
                f"CPU(avg)={avg_cpu_pct:.1f}% | RAM(peak_e2e)={peak_ram:.0f}MB"
            )

            # Cleanup
            K.clear_session()
            del model
            gc.collect()

    # Save Summary
    summary_path = os.path.join(cfg.out_root, f"summary_lb{lookback}.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n[COMPLETE] Saved summary -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, required=True, help="Lookback window size (e.g., 24, 72, 168)")
    args = parser.parse_args()
    run_specific_lookback(args.lookback, ExpConfig())