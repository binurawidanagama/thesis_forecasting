# Austria ST Forecasting — dCeNN → ELM → ASP (Minimal)

## What this is
A minimal, reproducible pipeline for 12-hour forecasting of **wind, solar, load, price** using a tiny **dCeNN** encoder, efficient **ELM** heads, and **ASP** plausibility rules. Baselines (CNN/LSTM) can be added later.

## Steps
1. `scripts/prepare_data.py` — merge your 3 CSVs, create CF labels, split train/val/test.
2. `scripts/train_encoder.py` — train the tiny dCeNN encoder with small linear heads (Phase A).
3. `scripts/fit_elm.py` — freeze encoder, extract latents, fit ELM heads (Phase B).
4. `scripts/predict.py` — run on test, save horizon-wise predictions.
5. `scripts/run_eval.py` — compute metrics (stub provided).
6. `scripts/run_asp.py` — enforce simple plausibility/bounds (ASP).

## Notes
- We currently use a **1×1 spatial grid**. When you add gridded weather, bump `spatial_hw` and feed channels into dCeNN.
- VRE is trained in **capacity-factor** space; multiply by (annual) capacity to get MW.
- All times are unified to **UTC**; ensure your CSV timestamp column is mapped in `configs/default.yaml`.

