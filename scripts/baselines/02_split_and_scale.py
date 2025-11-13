import numpy as np, pandas as pd, yaml, os
from sklearn.preprocessing import StandardScaler

cfg = yaml.safe_load(open("configs/base.yaml"))
f_cfg = yaml.safe_load(open("configs/features.yaml"))

df = pd.read_csv(cfg["engineered_csv"], parse_dates=["Time (UTC)"]).set_index("Time (UTC)").sort_index()

FEATS_BASE = [
  "Actual_Load_MW","Solar_MW","Wind_MW","Price_EUR_MWh",
  "temperature_2m (°C)","relative_humidity_2m (%)",
  "surface_pressure (hPa)","precipitation (mm)","shortwave_radiation (W/m²)",
  "air_density_kgm3","wind_speed_100m (m/s)","wind_power_proxy","pv_proxy",
  "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
  "is_public_holiday","is_weekend","is_special_day"
]

# Winsorize
for c in f_cfg["winsorize_cols"]:
    a,b = df[c].quantile([0.01,0.99]); df[c] = df[c].clip(a,b)

def make_supervised(df, horizon=1, lags=(1,2,3,6,12,24,48,72), rolls=(3,6,12,24,48)):
    X = df[FEATS_BASE].copy()
    core = ["Actual_Load_MW","Solar_MW","Wind_MW","Price_EUR_MWh",
            "temperature_2m (°C)","wind_speed_10m (m/s)","shortwave_radiation (W/m²)"]
    for c in core:
        for L in lags:  X[f"{c}_lag{L}"] = df[c].shift(L)
        for R in rolls: X[f"{c}_rmean{R}"] = df[c].rolling(R, min_periods=max(2,int(0.6*R))).mean()
    Y = pd.DataFrame({
        "CF_Solar": df["CF_Solar"].shift(-horizon),
        "CF_Wind":  df["CF_Wind"].shift(-horizon),
        "Load_MW":  df["Actual_Load_MW"].shift(-horizon),
        "Price":    df["Price_EUR_MWh"].shift(-horizon),
    }, index=df.index)
    XY = X.join(Y).dropna()
    return XY.iloc[:, :X.shape[1]], XY.iloc[:, X.shape[1]:]

def sub(df,s,e): return df.loc[s:e]

train = sub(df, cfg["split"]["train_start"], cfg["split"]["train_end"])
val   = sub(df, cfg["split"]["val_start"],   cfg["split"]["val_end"])
test  = sub(df, cfg["split"]["test_start"],  cfg["split"]["test_end"])

Xtr,Ytr = make_supervised(train, cfg["horizon"], f_cfg["lags"], f_cfg["rolls"])
Xva,Yva = make_supervised(val,   cfg["horizon"], f_cfg["lags"], f_cfg["rolls"])
Xte,Yte = make_supervised(test,  cfg["horizon"], f_cfg["lags"], f_cfg["rolls"])

scaler = StandardScaler().fit(Xtr.values)
Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr.values), scaler.transform(Xva.values), scaler.transform(Xte.values)

os.makedirs("artifacts", exist_ok=True)
np.savez_compressed(
    cfg["npz_path"],
    X_train=Xtr_s, Y_train=Ytr.values.astype(np.float32),
    X_val=Xva_s,   Y_val=Yva.values.astype(np.float32),
    X_test=Xte_s,  Y_test=Yte.values.astype(np.float32),
    feature_names=np.array(Xtr.columns.tolist()),
    target_names=np.array(Ytr.columns.tolist()),
)
print(f"Saved {cfg['npz_path']}")
