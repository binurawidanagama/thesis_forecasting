import numpy as np
import pandas as pd

def apply_asp_bounds(df, cap_wind_col="cap_wind_mw", cap_solar_col="cap_solar_mw",
                     wind_cols=None, solar_cols=None, load_cols=None, price_cols=None,
                     pv_night_hours=(0, 1, 2, 3, 4, 21, 22, 23), ramp_limits=None):
    """
    Apply ASP bounds to the predictions, fixing physical violations like:
    - capacity bounds for wind/solar
    - PV predictions at night
    - ramp limitations

    df: DataFrame indexed by timestamp with prediction columns.
    """

    # Ensure df.index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    flags = pd.DataFrame(index=df.index)
    flags["adjusted"] = 0

    # bounds by capacity
    if wind_cols:
        for c in wind_cols:
            before = df[c].copy()
            df[c] = df[c].clip(lower=0, upper=df[cap_wind_col])
            flags["adjusted"] |= (df[c] != before).astype(int)
    
    if solar_cols:
        for c in solar_cols:
            before = df[c].copy()
            df[c] = df[c].clip(lower=0, upper=df[cap_solar_col])
            flags["adjusted"] |= (df[c] != before).astype(int)

    # PV at night ≈ 0
    if solar_cols:
        hrs = df.index.hour  # This now works because df.index is a DatetimeIndex
        night_mask = hrs.isin(pv_night_hours)
        for c in solar_cols:
            before = df[c].copy()
            df.loc[night_mask, c] = 0.0
            flags["adjusted"] |= (df[c] != before).astype(int)

    # simple ramp caps (optional)
    if ramp_limits:
        for name, cols in [("wind", wind_cols), ("solar", solar_cols), ("load", load_cols), ("price", price_cols)]:
            if not cols:
                continue
            cap = ramp_limits.get(name, None)
            if not cap:
                continue
            for c in cols:
                dif = df[c].diff().abs()
                viol = dif > cap
                df.loc[viol, c] = df[c].shift(1) + np.sign(df[c] - df[c].shift(1)) * cap
                flags["adjusted"] |= viol.astype(int)

    return df, flags
