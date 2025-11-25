import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def make_windows(df, feature_cols, target_cols, context=168, horizon=12, critical_cols=None):
    """
    Build windows; reject a window only if CRITICAL columns contain NaNs.
    Returns:
      X_dcenn: [N, T, F, 1, 1], Y: [N, H, C], idx: DatetimeIndex of window anchor times.
    """
    cols = feature_cols + target_cols
    values = df[cols].values.astype("float32")
    n = len(df)
    X, Y, idx = [], [], []

    crit_idx = []
    if critical_cols:
        for c in critical_cols:
            if c in cols:
                crit_idx.append(cols.index(c))

    for t in range(context, n - horizon):
        # IMPORTANT: end index is EXCLUSIVE; use t + horizon (NOT t + 1 + horizon)
        # This yields a window length of context + horizon
        sl = slice(t - context, t + horizon)

        win = values[sl, :]
        # only check CRITICAL columns for NaN/Inf
        if crit_idx:
            sub = win[:, crit_idx]
            if np.isnan(sub).any() or np.isinf(sub).any():
                continue

        # first 'context' rows are features, last 'horizon' rows are targets
        feat = win[:context, :len(feature_cols)]
        targ = win[context:, len(feature_cols):]  # has length == horizon

        # guard in case of short edge windows
        if feat.shape[0] != context or targ.shape[0] != horizon:
            continue

        X.append(feat)
        Y.append(targ)
        idx.append(df.index[t])

    if not X:
        return (np.empty((0, context, len(feature_cols), 1, 1), dtype="float32"),
                np.empty((0, horizon, len(target_cols)), dtype="float32"),
                pd.DatetimeIndex([]))

    X = np.array(X, dtype=np.float32)            # [N, T, F]
    Y = np.array(Y, dtype=np.float32)            # [N, H, C]
    X_dcenn = X[:, :, :, None, None]             # [N, T, F, 1, 1]
    return X_dcenn, Y, pd.DatetimeIndex(idx)
# ---------- NEW: cache helper ----------
from pathlib import Path
from .preprocess import build_master

def cache_dcenn_windows(cfg):
    """
    Build or load cached dCeNN windows for train/val/test.

    Returns:
        Xtr, Ytr, Xva, Yva, Xte, Yte
    """
    paths = cfg["paths"]
    npz_path = Path(paths.get("windows_npz", "data/interim/windows_dcenn.npz"))

    # This must match what you use in train_encoder.py / fit_elm.py
    feature_cols = [
    # Calendar
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    # Holiday / special day
    "holiday",
    # Installed capacities
    "cap_wind_mw", "cap_solar_mw",
    # Past targets (lags via 168h context)
    "wind_mw", "solar_mw", "load_mw", "price_eur_mwh",
    # Weather & proxies (direct from engineered.csv)
    "temperature_2m_C",
    "relative_humidity_2m_pct",
    "wind_speed_100m (m/s)",
    "surface_pressure_hPa",
    "precipitation_mm",
    "shortwave_radiation_Wm2",
    "air_density_kgm3",
    "pv_proxy",
    "wind_power_proxy",
    ]

    target_cols  = ["cf_wind","cf_solar","load_mw","price_eur_mwh"]

    ctx = cfg["features"]["context_hours"]
    hz  = cfg["features"]["horizon_hours"]

    # If cached file exists, just load and return
    if npz_path.exists():
        data = np.load(npz_path)
        Xtr = data["Xtr"]; Ytr = data["Ytr"]
        Xva = data["Xva"]; Yva = data["Yva"]
        Xte = data["Xte"]; Yte = data["Yte"]
        return Xtr, Ytr, Xva, Yva, Xte, Yte

    # Otherwise: build from scratch once
    train_df, val_df, test_df = build_master(cfg)
    # --- NEW: scale encoder input features ---
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    # (optional) save scaler with joblib if you want to reuse it later

    Xtr, Ytr, _ = make_windows(train_df, feature_cols, target_cols, ctx, hz)
    Xva, Yva, _ = make_windows(val_df,   feature_cols, target_cols, ctx, hz)
    Xte, Yte, _ = make_windows(test_df,  feature_cols, target_cols, ctx, hz)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        Xtr=Xtr.astype("float32"), Ytr=Ytr.astype("float32"),
        Xva=Xva.astype("float32"), Yva=Yva.astype("float32"),
        Xte=Xte.astype("float32"), Yte=Yte.astype("float32"),
    )
    print(f"[cache_dcenn_windows] Saved cached windows to {npz_path}")
    return Xtr, Ytr, Xva, Yva, Xte, Yte