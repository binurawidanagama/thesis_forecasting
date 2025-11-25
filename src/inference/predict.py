# src/inference/predict.py

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path

from ..config import load_config
from ..dataio.preprocess import build_master
from ..dataio.window import make_windows
from ..models.dcenn import TinyDCENN
from ..train.fit_elm import extract_latents


def _ensemble_predict(models, X_np):
    """
    models: list[RandomFeatureELM]
    X_np: numpy array [N, in_dim]
    returns: numpy [N, out_dim] averaged over ensemble
    """
    preds = []
    for m in models:
        with torch.no_grad():
            y = m.predict(torch.from_numpy(X_np.astype(np.float32)))  # torch [N, out_dim]
            preds.append(y.cpu().numpy())
    return np.mean(preds, axis=0)


def run(cfg_path: str):
    cfg = load_config(cfg_path)

    # ---- 1. Load data and build test windows (same setup as encoder/ELM training) ----
    feature_cols = [
        # Calendar
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        # Holiday
        "holiday",
        # Capacities
        "cap_wind_mw", "cap_solar_mw",
        # Past targets
        "wind_mw", "solar_mw", "load_mw", "price_eur_mwh",
        # Weather + proxies
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
    target_cols = ["cf_wind", "cf_solar", "load_mw", "price_eur_mwh"]

    ctx = cfg["features"]["context_hours"]
    hz = cfg["features"]["horizon_hours"]

    train_df, val_df, test_df = build_master(cfg)

    Xte, Yte, te_index = make_windows(test_df, feature_cols, target_cols, ctx, hz)
    N = Xte.shape[0]
    in_channels = Xte.shape[2]

    # ---- 2. Load encoder and compute test latents ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = TinyDCENN(
        in_channels=in_channels,
        hidden_channels=cfg["training"]["encoder"]["latent_channels"],
    ).to(device)

    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])

    Zte = extract_latents(enc, Xte, device=device)  # [N, latent_dim]

    # ---- 3. Load ELMs and latent normalization stats ----
    elms_path = Path(cfg["paths"]["checkpoints_dir"]) / "elms.joblib"
    saved = joblib.load(elms_path)

    mu = saved["mu"]
    sigma = saved["sigma"]
    Zte_n = (Zte - mu) / sigma

    models_wind = saved["elm_wind"]
    models_solar = saved["elm_solar"]
    models_load = saved["elm_load"]
    models_price = saved["elm_price"]

    # ---- 4. Build extra price features for test (last price + price-24h) ----
    p_test = test_df["price_eur_mwh"]
    p_test_shift24 = p_test.shift(24)

    last_price_te = p_test.reindex(te_index).values
    price_24h_te = p_test_shift24.reindex(te_index)
    price_24h_te = price_24h_te.fillna(p_test.reindex(te_index)).values

    extra_price_te = np.stack([last_price_te, price_24h_te], axis=1)  # [N, 2]

    # ---- 5. Predict CF_wind, CF_solar, load, price ----
    X_wind_te  = Zte_n
    X_solar_te = Zte_n
    X_load_te  = Zte_n
    X_price_te = np.concatenate([Zte_n, extra_price_te], axis=1)

    cf_wind_te  = _ensemble_predict(models_wind,  X_wind_te)   # [N, H]
    cf_solar_te = _ensemble_predict(models_solar, X_solar_te)  # [N, H]
    load_te     = _ensemble_predict(models_load,  X_load_te)   # [N, H]
    price_te    = _ensemble_predict(models_price, X_price_te)  # [N, H]

    # ---- 6. Convert CFs to MW using capacities at the anchor time ----
    cap_w_series = test_df["cap_wind_mw"]
    cap_s_series = test_df["cap_solar_mw"]

    caps_te_w = cap_w_series.reindex(te_index).values  # [N]
    caps_te_s = cap_s_series.reindex(te_index).values  # [N]

    # broadcast to [N, H]
    caps_te_w_2d = np.repeat(caps_te_w[:, None], hz, axis=1)
    caps_te_s_2d = np.repeat(caps_te_s[:, None], hz, axis=1)

    wind_mw_te  = cf_wind_te  * caps_te_w_2d
    solar_mw_te = cf_solar_te * caps_te_s_2d

    # ---- 7. Flatten into [N, H*4] and save ----
    cols = []
    blocks = []

    for h in range(hz):
        cols.extend([
            f"wind_mw+h{h+1}",
            f"solar_mw+h{h+1}",
            f"load_mw+h{h+1}",
            f"price+h{h+1}",
        ])
        blocks.append(
            np.column_stack([
                wind_mw_te[:, h],
                solar_mw_te[:, h],
                load_te[:, h],
                price_te[:, h],
            ])
        )

    arr = np.concatenate(blocks, axis=1)  # [N, H*4]

    pred_df = pd.DataFrame(arr, index=te_index, columns=cols)
    out_path = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path)
    print(f"[predict] Saved test predictions to {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
