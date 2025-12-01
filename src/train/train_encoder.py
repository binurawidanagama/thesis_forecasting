import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from ..config import load_config
from ..utils.log import get_logger
from ..dataio.dataset import WindowDataset
from ..models.dcenn import TinyDCENN
from ..dataio.window import cache_dcenn_windows

def run(cfg_path):
    cfg = load_config(cfg_path)
    log = get_logger("train-encoder")

    # 1. Load Data (Force Rebuild enabled in window.py)
    Xtr, Ytr, Xva, Yva, _, _ = cache_dcenn_windows(cfg)
    
    # 2. Scale Targets for Encoder Training
    # Reshape [N, H, C] -> [N*H, C]
    N_tr, H, C = Ytr.shape
    N_va = Yva.shape[0]
    
    scaler = StandardScaler()
    Ytr_flat = Ytr.reshape(-1, C)
    
    # Fit on Train, transform Train
    Ytr_scaled = scaler.fit_transform(Ytr_flat).reshape(N_tr, H, C).astype(np.float32)
    # Transform Val using Train stats
    Yva_scaled = scaler.transform(Yva.reshape(-1, C)).reshape(N_va, H, C).astype(np.float32)
    
    log.info(f"Target Scaler Mean: {scaler.mean_}")
    log.info(f"Target Scaler Scale: {scaler.scale_}")

    # 3. Datasets
    in_channels = Xtr.shape[2]
    train_ds = WindowDataset(Xtr, Ytr_scaled) 
    val_ds   = WindowDataset(Xva, Yva_scaled)
    
    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["encoder"]["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["training"]["encoder"]["batch_size"], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TinyDCENN(
        in_channels=in_channels,
        hidden_channels=cfg["training"]["encoder"]["latent_channels"]
    ).to(device)
    
    heads = nn.ModuleDict({
        "cf_wind":  nn.Linear(cfg["training"]["encoder"]["latent_channels"], H),
        "cf_solar": nn.Linear(cfg["training"]["encoder"]["latent_channels"], H),
        "load":     nn.Linear(cfg["training"]["encoder"]["latent_channels"], H),
        "price":    nn.Linear(cfg["training"]["encoder"]["latent_channels"], H),
    }).to(device)
    
    optim = torch.optim.Adam(list(model.parameters())+list(heads.parameters()), lr=cfg["training"]["encoder"]["lr"])
    mae = nn.L1Loss()
    price_w = cfg["training"]["encoder"]["price_loss_weight"]
    
    best_val = 1e9
    
    for epoch in range(cfg["training"]["encoder"]["epochs"]):
        model.train(); heads.train(); total=0
        for xb, yb in tqdm(train_dl, desc=f"epoch {epoch+1}"):
            xb = xb.to(device)
            yb = yb.to(device) # yb is SCALED
            z = model(xb)
            
            pred = torch.stack([
                heads["cf_wind"](z), heads["cf_solar"](z),
                heads["load"](z),    heads["price"](z),
            ], dim=-1)
            
            loss = mae(pred[:,:,0], yb[:,:,0]) + \
                   mae(pred[:,:,1], yb[:,:,1]) + \
                   mae(pred[:,:,2], yb[:,:,2]) + \
                   price_w * mae(pred[:,:,3], yb[:,:,3])
                   
            optim.zero_grad(); loss.backward(); optim.step(); total += loss.item()
            
        # Validation
        model.eval(); heads.eval()
        vloss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb=xb.to(device); yb=yb.to(device)
                z = model(xb)
                pred = torch.stack([
                    heads["cf_wind"](z), heads["cf_solar"](z),
                    heads["load"](z),    heads["price"](z),
                ], dim=-1)
                
                vloss += (mae(pred[:,:,0], yb[:,:,0]) + \
                          mae(pred[:,:,1], yb[:,:,1]) + \
                          mae(pred[:,:,2], yb[:,:,2]) + \
                          price_w*mae(pred[:,:,3], yb[:,:,3])).item()
                          
        log.info(f"epoch {epoch+1} train_loss={total/len(train_dl):.4f} val_loss={vloss/len(val_dl):.4f}")
        
        if vloss < best_val:
            best_val = vloss
            Path(cfg["paths"]["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save({"encoder": model.state_dict()}, Path(cfg["paths"]["checkpoints_dir"])/"encoder.pt")

    log.info("Saved best encoder.")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args=ap.parse_args(); run(args.config)