import pandas as pd, numpy as np
from pathlib import Path
from .loader import load_engineered

# Treat these as CRITICAL (must be clean)
CRITICAL = ["wind_mw","solar_mw","load_mw","price_eur_mwh","cap_wind_mw","cap_solar_mw"]

# Everything else is OPTIONAL and can be imputed more aggressively
OPTIONAL = [
    "cf_wind","cf_solar",
    "temperature_2m_C","relative_humidity_2m_pct","wind_speed_10m_ms","wind_speed_100m (m/s)",
    "surface_pressure_hPa","precipitation_mm","shortwave_radiation_Wm2",
    "temperature_2m (°C)","relative_humidity_2m (%)","wind_speed_10m (m/s)",
    "surface_pressure (hPa)","precipitation (mm)","shortwave_radiation (W/mÂ²)",
    "air_density_kgm3","pv_proxy","wind_power_proxy",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "is_public_holiday","is_weekend","is_special_day"
]

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _clean_numeric(df):
    cols = [c for c in set(CRITICAL+OPTIONAL) if c in df.columns]
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df = _coerce_numeric(df, cols)

    # OPTIONAL: up to 24h interpolation, then ffill/bfill
    opt = [c for c in OPTIONAL if c in df.columns]
    if opt:
        df[opt] = df[opt].interpolate(limit=24, limit_direction="both")
        df[opt] = df[opt].ffill().bfill()

    # CRITICAL: up to 6h interpolation, then minimal ffill/bfill
    crit = [c for c in CRITICAL if c in df.columns]
    if crit:
        df[crit] = df[crit].interpolate(limit=6, limit_direction="both").ffill().bfill()

    # drop rows still missing CRITICAL after the above
    if crit:
        df = df.dropna(subset=crit)
    return df

def build_master(cfg):
    df = load_engineered(cfg)

    # add cyclical time features if missing (idempotent)
    idx = df.index.tz_convert("UTC")
    for k,v in {
        "hour_sin":  np.sin(2*np.pi*idx.hour/24),
        "hour_cos":  np.cos(2*np.pi*idx.hour/24),
        "dow_sin":   np.sin(2*np.pi*idx.dayofweek/7),
        "dow_cos":   np.cos(2*np.pi*idx.dayofweek/7),
        "month_sin": np.sin(2*np.pi*(idx.month-1)/12),
        "month_cos": np.cos(2*np.pi*(idx.month-1)/12),
    }.items():
        if k not in df.columns: df[k] = v

    # clean before split
    df = _clean_numeric(df)

    # splits
    split = cfg["time"]["split"]
    train_end = pd.Timestamp(split["train_until"], tz="UTC")
    val_end   = pd.Timestamp(split["val_until"],   tz="UTC")
    test_end  = pd.Timestamp(split["test_until"],  tz="UTC")

    df = df.loc[:test_end]
    train_df = df.loc[:train_end]
    val_df   = df.loc[train_end + pd.Timedelta(hours=1): val_end]
    test_df  = df.loc[val_end + pd.Timedelta(hours=1):]

    outdir = Path(cfg["paths"]["interim_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(outdir/"train.parquet")
    val_df.to_parquet(outdir/"val.parquet")
    test_df.to_parquet(outdir/"test.parquet")
    return train_df, val_df, test_df