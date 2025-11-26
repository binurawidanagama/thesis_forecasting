import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

def stack(pred, prefix, H):
    return np.stack([pred[f"{prefix}+h{h+1}"].values for h in range(H)], axis=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/predictions/test_predictions.parquet")
    ap.add_argument("--truth", default="data/interim/test.parquet")
    ap.add_argument("--start", required=True, help="UTC start like 2022-08-01 00:00:00")
    ap.add_argument("--hours", type=int, default=168)  # one week
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    pred = pd.read_parquet(args.pred)
    truth = pd.read_parquet(args.truth)
    H = args.horizon

    # Build aligned truth (wind/solar CF→MW using capacity at window end)
    # We mimic src/eval/evaluate.py’s window alignment:
    # For display, we’ll plot the 1-step ahead series (h=1) to keep it simple.
    cols = ["wind_mw","solar_mw","load_mw","price"]
    p1 = pred[[f"{c}+h1" for c in cols]].rename(columns=lambda c: c.replace("+h1",""))
    # Align index to truth
    p1 = p1.loc[p1.index.intersection(truth.index)]

    # slice window
    idx = pd.date_range(args.start, periods=args.hours, freq="h", tz="UTC")
    p1 = p1.loc[idx]
    tr = truth.loc[idx, ["wind_mw","solar_mw","load_mw","price_eur_mwh"]].rename(columns={"price_eur_mwh":"price"})

    # Plot
    fig, axes = plt.subplots(4,1, figsize=(12,10), sharex=True)
    for ax, key, label in zip(axes, ["wind_mw","solar_mw","load_mw","price"], ["Wind (MW)","Solar (MW)","Load (MW)","Price (€/MWh)"]):
        ax.plot(tr.index, tr[key], label=f"True {label}")
        ax.plot(p1.index, p1[key], label=f"Pred {label}", alpha=0.8)
        ax.set_ylabel(label); ax.grid(True, ls=":")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("UTC time")
    plt.tight_layout()
    out = Path("outputs")/"viz_week.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
