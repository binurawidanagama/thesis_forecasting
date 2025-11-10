# Safe CFs + robust column handling + annual→hourly capacity broadcast
import numpy as np, pandas as pd, yaml, os
from pathlib import Path

cfg = yaml.safe_load(open("configs/default.yaml"))

RAW_CSV = cfg.get("raw_hourly_csv", "data/raw/AT_hourly_MW_and_Price.csv")
CAP_CSV = cfg.get("capacity_csv",   "data/raw/Installed Capacity per Production Type_201701010000-202301010000.csv")
HOL_CSV = cfg.get("holidays_csv",   "data/AT_public_holidays_2017_2022.csv")
OUT_CSV = cfg.get("engineered_csv", "data/processed/AT_engineered.csv")
OUT_FULL = cfg.get("engineered_full_csv")  # optional

Path("data/processed").mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def monthwise_minmax_norm(series: pd.Series, lo_q=0.02, hi_q=0.98) -> pd.Series:
    s = series.astype(float)
    out = pd.Series(index=s.index, dtype=float)
    idx = s.index
    for m in range(1, 13):
        mm = idx.month == m
        p = s[mm]
        if p.notna().sum() < 10:
            out.loc[mm] = p
            continue
        lo, hi = p.quantile([lo_q, hi_q])
        out.loc[mm] = ((p - lo) / max(hi - lo, 1e-6)).clip(0, 1.2)
    return out

def safe_cf(mw: pd.Series, cap: pd.Series, proxy: pd.Series | None = None) -> pd.Series:
    eps = 1e-6
    mw = pd.to_numeric(mw, errors="coerce").astype(float)
    cap = pd.to_numeric(cap, errors="coerce").astype(float)
    cf = pd.Series(0.0, index=mw.index)

    mask_cap = cap > 0
    cf.loc[mask_cap] = (mw.loc[mask_cap] / np.maximum(cap.loc[mask_cap], eps)).clip(0, 1.2)

    bad = (~mask_cap) & (mw > 0)
    if proxy is not None and bad.any():
        prox = monthwise_minmax_norm(pd.to_numeric(proxy, errors="coerce"))
        cf.loc[bad] = prox.loc[bad].fillna(0.0)

    return cf.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def first_existing(df, *cands):
    for c in cands:
        if c in df.columns: return c
    raise KeyError(f"None of the columns exist: {cands}")

# ---------- load raw ----------
raw = pd.read_csv(RAW_CSV, parse_dates=["Time (UTC)"]).sort_values("Time (UTC)")
raw["Time (UTC)"] = pd.to_datetime(raw["Time (UTC)"], utc=True)
raw = raw.set_index("Time (UTC)").asfreq("H")

# normalize odd encodings once (keep both if present)
rename_map = {
  "temperature_2m (°C)": "temperature_2m_C",
  "relative_humidity_2m (%)": "relative_humidity_2m_pct",
  "wind_speed_10m (m/s)": "wind_speed_10m_ms",
  "wind_speed_100m (m/s)": "wind_speed_100m_ms",
  "surface_pressure (hPa)": "surface_pressure_hPa",
  "shortwave_radiation (W/mÂ²)": "shortwave_radiation_Wm2",
  "precipitation (mm)": "precipitation_mm",
  "DA_Price_EUR_MWh": "Price_EUR_MWh",
}
raw = raw.rename(columns={k:v for k,v in rename_map.items() if k in raw.columns})

# Mandatory base columns (support legacy names too)
PRICE = first_existing(raw, "Price_EUR_MWh", "DA_Price_EUR_MWh")
WIND  = first_existing(raw, "Wind_MW")
SOLAR = first_existing(raw, "Solar_MW")
LOAD  = first_existing(raw, "Actual_Load_MW")

# ---------- derived met & proxies ----------
# choose temp/pressure/wind columns (works whether you renamed or not)
T_C   = raw.get("temperature_2m_C",   raw.get("temperature_2m (°C)"))
p_hPa = raw.get("surface_pressure_hPa", raw.get("surface_pressure (hPa)"))
RHpct = raw.get("relative_humidity_2m_pct", raw.get("relative_humidity_2m (%)"))
v10   = raw.get("wind_speed_10m_ms",  raw.get("wind_speed_10m (m/s)"))
sw    = raw.get("shortwave_radiation_Wm2", raw.get("shortwave_radiation (W/mÂ²)"))

# guard against missing weather columns
T_C = pd.to_numeric(T_C, errors="coerce").fillna(method="ffill") if T_C is not None else pd.Series(15.0, index=raw.index)
p_Pa = pd.to_numeric(p_hPa, errors="coerce").fillna(method="ffill").astype(float)*100 if p_hPa is not None else pd.Series(1013.25*100, index=raw.index)
RH = (pd.to_numeric(RHpct, errors="coerce").clip(0,100)/100.0) if RHpct is not None else pd.Series(0.6, index=raw.index)
v10 = pd.to_numeric(v10, errors="coerce").clip(lower=0) if v10 is not None else pd.Series(3.0, index=raw.index)
T_K = T_C + 273.15
e_s = 610.94 * np.exp((17.625*T_C)/(T_C+243.04))
e   = RH * e_s
q   = 0.622 * e / (p_Pa - (1-0.622)*e).clip(lower=1.0)
rho_air = p_Pa / (287.05 * (T_K*(1+0.61*q)))
raw["air_density_kgm3"] = rho_air

v100 = v10 * (100/10)**0.143
raw["wind_speed_100m_ms"] = v100

if sw is None:
    sw = pd.Series(0.0, index=raw.index)
raw["pv_proxy"] = (pd.to_numeric(sw, errors="coerce").clip(lower=0) *
                   (1.0 - 0.005*(T_C-25).clip(lower=0))).clip(lower=0)
raw["wind_power_proxy"] = (raw["air_density_kgm3"]*(v100**3)).clip(upper=np.nanpercentile((rho_air*(v100**3)).fillna(0), 99))

# ---------- holidays ----------
if Path(HOL_CSV).exists():
    hol = pd.read_csv(HOL_CSV, parse_dates=["date"])
    hol["date"] = pd.to_datetime(hol["date"], utc=True).dt.floor("D")
    raw["is_public_holiday"] = raw.index.floor("D").isin(hol["date"]).astype(int)
else:
    raw["is_public_holiday"] = 0
raw["is_weekend"] = (raw.index.dayofweek >= 5).astype(int)
raw["is_special_day"] = ((raw["is_public_holiday"]==1) | (raw["is_weekend"]==1)).astype(int)

# ---------- annual capacity -> hourly ----------
cap = pd.read_csv(CAP_CSV)
cap = cap.melt(id_vars=[c for c in cap.columns if "Production" in c], var_name="year", value_name="MW")
cap["year"] = cap["year"].astype(str).str.extract(r"(\d{4})").astype(int)
ptype = [c for c in cap.columns if "Production" in c][0]
cap["ptype_norm"] = cap[ptype].astype(str).str.lower()

solar_cap = cap[cap["ptype_norm"].str.contains("solar")]["MW"].groupby(cap["year"]).sum()
wind_cap  = cap[cap["ptype_norm"].str.contains("wind")]["MW"].groupby(cap["year"]).sum()

def annual_to_hourly(s, idx):
    y = pd.Series(index=idx, data=idx.year).map(s).astype(float)
    return y.ffill().bfill()

df = raw.copy()
df["Solar_Cap_MW"] = annual_to_hourly(solar_cap, df.index).fillna(0.0)
df["Wind_Cap_MW"]  = annual_to_hourly(wind_cap,  df.index).fillna(0.0)

# ---------- safe CFs (use proxies when cap==0 & MW>0) ----------
df["CF_Solar"] = safe_cf(df[SOLAR], df["Solar_Cap_MW"], proxy=df["pv_proxy"])
df["CF_Wind"]  = safe_cf(df[WIND],  df["Wind_Cap_MW"],  proxy=df["wind_power_proxy"])

# ---------- calendar sines/cosines ----------
idx = df.index
df["hour_sin"]  = np.sin(2*np.pi*idx.hour/24);  df["hour_cos"]  = np.cos(2*np.pi*idx.hour/24)
df["dow_sin"]   = np.sin(2*np.pi*idx.dayofweek/7); df["dow_cos"] = np.cos(2*np.pi*idx.dayofweek/7)
df["month_sin"] = np.sin(2*np.pi*(idx.month-1)/12); df["month_cos"] = np.cos(2*np.pi*(idx.month-1)/12)

# ---------- save ----------
df_out = df.reset_index().rename(columns={"index":"Time (UTC)", "wind_speed_100m_ms":"wind_speed_100m (m/s)"})
df_out.to_csv(OUT_CSV, index=False)
if OUT_FULL:
    df_out.to_csv(OUT_FULL, index=False)
print(f"Saved engineered CSV to {OUT_CSV}" + (f" and {OUT_FULL}" if OUT_FULL else ""))