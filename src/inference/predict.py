import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from ..config import load_config
from ..dataio.window import cache_dcenn_windows
from ..models.dcenn import TinyDCENN
from ..train.fit_elm import extract_latents

def _ensemble_predict(models, X_np):
    preds = []
    for m in models:
        with torch.no_grad():
            y = m.predict(torch.from_numpy(X_np.astype(np.float32)))
            preds.append(y.cpu().numpy())
    return np.mean(preds, axis=0)

def run(cfg_path: str):
    cfg = load_config(cfg_path)
    
    # 1. Load Data
    _, _, _, _, Xte, _ = cache_dcenn_windows(cfg)
    
    # Reconstruct Index
    from ..dataio.preprocess import build_master
    _, _, test_df = build_master(cfg)
    ctx, hz = cfg["features"]["context_hours"], cfg["features"]["horizon_hours"]
    te_index = test_df.index[list(range(ctx, len(test_df) - hz))]
    if len(te_index) > Xte.shape[0]: te_index = te_index[:Xte.shape[0]]

    # 2. Encoder & Latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = TinyDCENN(Xte.shape[2], cfg["training"]["encoder"]["latent_channels"]).to(device)
    ckpt = torch.load(Path(cfg["paths"]["checkpoints_dir"]) / "encoder.pt", map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    
    Zte = extract_latents(enc, Xte, device=device)
    saved = joblib.load(Path(cfg["paths"]["checkpoints_dir"]) / "elms.joblib")
    Zte_n = (Zte - saved["mu"]) / saved["sigma"]
    
    # 3. Prepare Inputs (Lags + Latents)
    def get_lags(X_tensor, col_idx):
        lag1  = X_tensor[:, -1, col_idx, 0, 0]
        lag24 = X_tensor[:, -24, col_idx, 0, 0]
        return np.stack([lag1, lag24], axis=1)

    # Note: Using implicit price lags (Golden Copy version)
    lag_wind  = get_lags(Xte, 9)
    lag_solar = get_lags(Xte, 10)
    lag_load  = get_lags(Xte, 11)
    lag_price = get_lags(Xte, 12)

    X_wind  = np.concatenate([Zte_n, lag_wind], axis=1)
    X_solar = np.concatenate([Zte_n, lag_solar], axis=1)
    X_load  = np.concatenate([Zte_n, lag_load], axis=1)
    X_price = np.concatenate([Zte_n, lag_price], axis=1)

    # 4. Predict
    p_wind  = _ensemble_predict(saved["elm_wind"],  X_wind)
    p_solar = _ensemble_predict(saved["elm_solar"], X_solar)
    p_load  = _ensemble_predict(saved["elm_load"],  X_load)
    p_price = _ensemble_predict(saved["elm_price"], X_price)

    # 5. Inverse Scale & "Soft" Clamping
    # We remove negatives (physics), but we DO NOT force night to zero.
    # This allows the "smooth/noisy" curve you preferred.
    N, H = p_wind.shape
    stacked = np.stack([p_wind, p_solar, p_load, p_price], axis=2).reshape(-1, 4)
    inverse = saved["target_scaler"].inverse_transform(stacked).reshape(N, H, 4)
    
    final_wind  = np.maximum(inverse[:,:,0], 0.0)
    final_solar = np.maximum(inverse[:,:,1], 0.0) # <--- No strict night mask here
    final_load  = np.maximum(inverse[:,:,2], 0.0)
    final_price = np.maximum(inverse[:,:,3], -100.0)

    # 6. Save
    cap_w = test_df.loc[te_index, "cap_wind_mw"].values[:, None]
    cap_s = test_df.loc[te_index, "cap_solar_mw"].values[:, None]
    mw_wind = final_wind * cap_w
    mw_solar = final_solar * cap_s

    cols = []
    blocks = []
    for h in range(hz):
        cols.extend([f"wind_mw+h{h+1}", f"solar_mw+h{h+1}", f"load_mw+h{h+1}", f"price+h{h+1}"])
        blocks.append(np.column_stack([mw_wind[:, h], mw_solar[:, h], final_load[:, h], final_price[:, h]]))
        
    out_path = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.concatenate(blocks, axis=1), index=te_index, columns=cols).to_parquet(out_path)
    print(f"[predict] Saved to {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); run(args.config)