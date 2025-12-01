import json
import pandas as pd, numpy as np
from pathlib import Path
from ..config import load_config
from ..utils.metrics import mae, smape
from ..dataio.preprocess import build_master
from ..dataio.window import make_windows


def run(cfg_path):
    cfg = load_config(cfg_path)
    out_dir = Path(cfg["paths"]["outputs_dir"])
    pred_p = out_dir / "predictions" / "test_predictions.parquet"
    if not pred_p.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_p}. Run the inference script first.")
    pred = pd.read_parquet(pred_p)

    # reconstruct ground truth windows for the test split
    train_df, val_df, test_df = build_master(cfg)
    feature_cols = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
                    "holiday","cap_wind_mw","cap_solar_mw",
                    "wind_mw","solar_mw","load_mw","price_eur_mwh"]
    target_cols  = ["cf_wind","cf_solar","load_mw","price_eur_mwh"]
    ctx = cfg["features"]["context_hours"]; hz = cfg["features"]["horizon_hours"]

    Xte, Yte, te_index = make_windows(test_df, feature_cols, target_cols, ctx, hz)

    # pred DataFrame columns use the naming scheme produced by the predictor: e.g. "wind_mw+h1", "solar_mw+h1", ...
    # We'll gather per-target predicted arrays by collecting columns for each horizon.
    prefixes = {
        "wind": "wind_mw",
        "solar": "solar_mw",
        "load": "load_mw",
        "price": "price",
    }

    metrics = {k: {} for k in prefixes.keys()}

    # for wind/solar predictions were saved in MW; ground truth Yte contains CFs for wind/solar (targets 0/1)
    # align capacities for test set (same logic as in inference)
    N = Yte.shape[0]
    caps_te_w = test_df["cap_wind_mw"].values[ctx: ctx + N]
    caps_te_s = test_df["cap_solar_mw"].values[ctx: ctx + N]

    for h in range(1, hz+1):
        # build true arrays for each target at horizon h
        idx = h-1
        true_w = Yte[:, idx, 0] * caps_te_w
        true_s = Yte[:, idx, 1] * caps_te_s
        true_load = Yte[:, idx, 2]
        true_price = Yte[:, idx, 3]

        # read corresponding predicted columns (safe even if columns are in different order)
        pw = pred[f"{prefixes['wind']}+h{h}"].values
        ps = pred[f"{prefixes['solar']}+h{h}"].values
        pl = pred[f"{prefixes['load']}+h{h}"].values
        pp = pred[f"{prefixes['price']}+h{h}"].values

        metrics["wind"][f"h{h}"] = {"MAE": float(mae(pw, true_w)), "SMAPE": float(smape(pw, true_w))}
        metrics["solar"][f"h{h}"] = {"MAE": float(mae(ps, true_s)), "SMAPE": float(smape(ps, true_s))}
        metrics["load"][f"h{h}"] = {"MAE": float(mae(pl, true_load)), "SMAPE": float(smape(pl, true_load))}
        metrics["price"][f"h{h}"] = {"MAE": float(mae(pp, true_price)), "SMAPE": float(smape(pp, true_price))}

    outp = out_dir / "eval" / "metrics_test.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, indent=2))
    print(f"Saved evaluation metrics to {outp}")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    run(args.config)

