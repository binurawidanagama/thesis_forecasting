import numpy as np
import pandas as pd

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
