import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from .preprocess import build_master

def make_windows(df, feature_cols, target_cols, context=168, horizon=12, critical_cols=None):
    """
    Build windows; reject a window only if CRITICAL columns contain NaNs.
    Returns:
      X_dcenn: [N, T, F, 1, 1], Y: [N, H, C], idx: DatetimeIndex of window anchor times.
    """
    cols = feature_cols + target_cols
    # Ensure all columns exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing in dataframe: {missing}")
        
    values = df[cols].values.astype("float32")
    n = len(df)
    X, Y, idx = [], [], []
    crit_idx = []
    
    # If not specified, treat targets as critical
    if critical_cols is None:
        critical_cols = target_cols
        
    for c in critical_cols:
        if c in cols:
            crit_idx.append(cols.index(c))
            
    for t in range(context, n - horizon):
        sl = slice(t - context, t + horizon)
        win = values[sl, :]
        
        if crit_idx:
            sub = win[:, crit_idx]
            if np.isnan(sub).any() or np.isinf(sub).any():
                continue
        
        # Split into features and targets
        feat = win[:context, :len(feature_cols)]
        targ = win[context:, len(feature_cols):]
        
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

def cache_dcenn_windows(cfg):
    """
    Build or load cached dCeNN windows.
    Generates X from SCALED features and Y from RAW targets.
    """
    paths = cfg["paths"]
    npz_path = Path(paths.get("windows_npz", "data/interim/windows_dcenn.npz"))
    
    # TEMPORARY: Keep rebuild=True until validation passes
    rebuild = True 

    feature_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "holiday",
        "cap_wind_mw", "cap_solar_mw",
        "wind_mw", "solar_mw", "load_mw", "price_eur_mwh",
        "temperature_2m_C", "relative_humidity_2m_pct", "wind_speed_100m (m/s)",
        "surface_pressure_hPa", "precipitation_mm", "shortwave_radiation_Wm2",
        "air_density_kgm3", "pv_proxy", "wind_power_proxy",
    ]
    target_cols = ["cf_wind", "cf_solar", "load_mw", "price_eur_mwh"]
    
    ctx = cfg["features"]["context_hours"]
    hz  = cfg["features"]["horizon_hours"]

    if npz_path.exists() and not rebuild:
        print(f"[cache] Loading windows from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        return (data["Xtr"], data["Ytr"], 
                data["Xva"], data["Yva"], 
                data["Xte"], data["Yte"])

    print("[cache] Building windows from scratch...")
    train_df_raw, val_df_raw, test_df_raw = build_master(cfg)

    # --- 1. Create SCALED copies for INPUTS (X) ---
    scaler = StandardScaler()
    
    train_df_scaled = train_df_raw.copy()
    val_df_scaled   = val_df_raw.copy()
    test_df_scaled  = test_df_raw.copy()
    
    # Fit scaler only on training inputs
    train_df_scaled[feature_cols] = scaler.fit_transform(train_df_scaled[feature_cols])
    val_df_scaled[feature_cols]   = scaler.transform(val_df_scaled[feature_cols])
    test_df_scaled[feature_cols]  = scaler.transform(test_df_scaled[feature_cols])

    # --- 2. Generate Windows ---
    # CRITICAL: 
    # Get X from the SCALED dataframe (normalized inputs for Neural Net)
    # Get Y from the RAW dataframe (actual MW/Euros for ground truth)
    
    def get_xy(df_scaled, df_raw):
        # We run make_windows twice. 
        # Pass 1: Get Scaled X (ignore Y)
        X_s, _, idx = make_windows(df_scaled, feature_cols, target_cols, ctx, hz)
        # Pass 2: Get Raw Y (ignore X)
        _, Y_r, _   = make_windows(df_raw,    feature_cols, target_cols, ctx, hz)
        return X_s, Y_r

    print("[cache] Generating Train windows...")
    Xtr, Ytr = get_xy(train_df_scaled, train_df_raw)
    
    print("[cache] Generating Val windows...")
    Xva, Yva = get_xy(val_df_scaled, val_df_raw)
    
    print("[cache] Generating Test windows...")
    Xte, Yte = get_xy(test_df_scaled, test_df_raw)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        Xtr=Xtr, Ytr=Ytr,
        Xva=Xva, Yva=Yva,
        Xte=Xte, Yte=Yte,
    )
    print(f"[cache] Saved new windows to {npz_path}")
    return Xtr, Ytr, Xva, Yva, Xte, Yte