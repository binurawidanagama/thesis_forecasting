import pandas as pd
from pathlib import Path

def load_engineered(cfg):
    p = Path(cfg["paths"]["engineered_csv"])
    c = cfg["columns"]
    ts = c["timestamp"]
    df = pd.read_csv(p, parse_dates=[ts])
    # force UTC (your file says (UTC) already, this keeps it explicit)
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.set_index(ts).sort_index().asfreq("h")

    # rename to canonical internal names
    ren = {
        c["wind_mw"]: "wind_mw",
        c["solar_mw"]: "solar_mw",
        c["load_mw"]: "load_mw",
        c["price_eur_mwh"]: "price_eur_mwh",
        c["cf_wind"]: "cf_wind",
        c["cf_solar"]: "cf_solar",
        c["cap_wind_mw"]: "cap_wind_mw",
        c["cap_solar_mw"]: "cap_solar_mw",
    }
    df = df.rename(columns=ren)

    # unify holiday flags into a single 'holiday' column if provided
    hol_keys = cfg["columns"].get("holiday_flags", [])
    if hol_keys:
        for h in hol_keys:
            if h not in df.columns:
                continue
            df[h] = (df[h].astype(int) > 0).astype(int)
        df["holiday"] = (df[hol_keys].sum(axis=1) > 0).astype(int)
    else:
        df["holiday"] = 0

    return df

