import torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from ..config import load_config
from ..utils.log import get_logger
from ..dataio.preprocess import build_master
from ..dataio.window import make_windows
from ..dataio.dataset import WindowDataset
from ..models.dcenn import TinyDCENN
from ..models.elm import RandomFeatureELM
import joblib
from ..dataio.window import cache_dcenn_windows


def extract_latents(encoder, X, batch=512, device="cpu"):
    encoder.eval()
    zs=[]
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            z = encoder(xb)
            zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)

def run(cfg_path):
    cfg = load_config(cfg_path); log = get_logger("fit-elm")
    train_df, val_df, test_df = build_master(cfg)  # still needed for capacities later
    Xtr, Ytr, Xva, Yva, Xte, Yte = cache_dcenn_windows(cfg)


    device = torch.device("cpu")
    enc = TinyDCENN(in_channels=Xtr.shape[2], hidden_channels=cfg["training"]["encoder"]["latent_channels"]).to(device)
    ckpt = torch.load(Path(cfg["paths"]["checkpoints_dir"])/"encoder.pt", map_location=device)
    enc.load_state_dict(ckpt["encoder"])

    Ztr = extract_latents(enc, Xtr, device=device)
    Zva = extract_latents(enc, Xva, device=device)
    Zte = extract_latents(enc, Xte, device=device)

    # Prepare targets
    Ytr_cf_wind  = Ytr[:,:,0]
    Ytr_cf_solar = Ytr[:,:,1]
    Ytr_load     = Ytr[:,:,2]
    Ytr_price    = Ytr[:,:,3]

    Yva_cf_wind  = Yva[:,:,0]; Yva_cf_solar = Yva[:,:,1]; Yva_load = Yva[:,:,2]; Yva_price = Yva[:,:,3]

    # Fit VRE & Load ELMs
    elm_cfg = cfg["training"]["elm"]
    elm_wind  = RandomFeatureELM(in_dim=Ztr.shape[1], out_dim=Ztr.shape[1], hidden=elm_cfg["hidden"], ridge_lambda=elm_cfg["ridge_lambda"])
    elm_solar = RandomFeatureELM(in_dim=Ztr.shape[1], out_dim=Ztr.shape[1], hidden=elm_cfg["hidden"], ridge_lambda=elm_cfg["ridge_lambda"])
    elm_load  = RandomFeatureELM(in_dim=Ztr.shape[1], out_dim=Ztr.shape[1], hidden=elm_cfg["hidden"], ridge_lambda=elm_cfg["ridge_lambda"])

    # Note: each ELM outputs horizon-dim directly
    elm_wind  = RandomFeatureELM(Ztr.shape[1], Ytr_cf_wind.shape[1], elm_cfg["hidden"], "tanh", elm_cfg["ridge_lambda"])
    elm_solar = RandomFeatureELM(Ztr.shape[1], Ytr_cf_solar.shape[1], elm_cfg["hidden"], "tanh", elm_cfg["ridge_lambda"])
    elm_load  = RandomFeatureELM(Ztr.shape[1], Ytr_load.shape[1],     elm_cfg["hidden"], "tanh", elm_cfg["ridge_lambda"])

    elm_wind.fit(torch.from_numpy(Ztr), torch.from_numpy(Ytr_cf_wind))
    elm_solar.fit(torch.from_numpy(Ztr), torch.from_numpy(Ytr_cf_solar))
    elm_load.fit(torch.from_numpy(Ztr), torch.from_numpy(Ytr_load))

    # Predict to build residual features
    cf_wind_tr  = elm_wind.predict(torch.from_numpy(Ztr)).numpy()
    cf_solar_tr = elm_solar.predict(torch.from_numpy(Ztr)).numpy()
    load_tr     = elm_load.predict(torch.from_numpy(Ztr)).numpy()

    # multiply CF by capacity (aligned to prediction hours)
    # we use the last context timestamp's capacities as proxy for t+h (annual caps ⇒ same within year)
    caps_tr_w = train_df["cap_wind_mw"].values[cfg["features"]["context_hours"] : cfg["features"]["context_hours"] + cf_wind_tr.shape[0]]
    caps_tr_s = train_df["cap_solar_mw"].values[cfg["features"]["context_hours"] : cfg["features"]["context_hours"] + cf_solar_tr.shape[0]]
    # Broadcast to horizon
    caps_tr_w = np.repeat(caps_tr_w[:,None], cf_wind_tr.shape[1], axis=1)
    caps_tr_s = np.repeat(caps_tr_s[:,None], cf_solar_tr.shape[1], axis=1)

    wind_mw_tr  = cf_wind_tr  * caps_tr_w
    solar_mw_tr = cf_solar_tr * caps_tr_s
    residual_tr = load_tr - (wind_mw_tr + solar_mw_tr)

    # Price ELM uses [z, residual_vector]
    Xtr_price = np.concatenate([Ztr, residual_tr], axis=1)
    elm_price = RandomFeatureELM(Xtr_price.shape[1], Ytr_price.shape[1], elm_cfg["hidden"], "tanh", elm_cfg["ridge_lambda"])
    elm_price.fit(torch.from_numpy(Xtr_price), torch.from_numpy(Ytr_price))

    # Save models
    outdir = Path(cfg["paths"]["checkpoints_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"elm_wind": elm_wind, "elm_solar": elm_solar, "elm_load": elm_load, "elm_price": elm_price}, outdir/"elms.joblib")
    np.savez(outdir/"latents.npz", Ztr=Ztr, Zva=Zva, Zte=Zte)
    print("Saved ELMs and latents.")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    run(args.config)

