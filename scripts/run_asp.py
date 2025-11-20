# scripts/run_asp.py

import subprocess
import pandas as pd
from pathlib import Path
from src.config import load_config
from src.dataio.preprocess import build_master
from src.utils.asp_rules import apply_asp_bounds  # import your current asp_rules function

def write_facts(cfg, preds, caps, facts_path):
    """
    preds: DataFrame with datetime index and columns wind_mw+h*, solar_mw+h*, ...
    caps:  DataFrame with columns cap_wind_mw, cap_solar_mw; same length/order as preds
    """
    hz = cfg["features"]["horizon_hours"]
    N = len(preds)
    night_hours = set(cfg["asp"]["pv_night_hours"])

    SCALE = 100  # encode values as integer = value * SCALE

    with open(facts_path, "w") as f:
        f.write("% auto-generated facts\n")

        for s in range(N):
            f.write(f"sample({s}).\n")
        for h in range(1, hz + 1):
            f.write(f"horizon({h}).\n")

        # capacities (also scaled for consistency)
        for s in range(N):
            cw = float(caps.iloc[s]["cap_wind_mw"])
            cs = float(caps.iloc[s]["cap_solar_mw"])
            cw_int = int(round(cw * SCALE))
            cs_int = int(round(cs * SCALE))
            f.write(f"cap(wind,{s},{cw_int}).\n")
            f.write(f"cap(solar,{s},{cs_int}).\n")

        # predictions + night flags
        for s in range(N):
            ts = preds.index[s]
            hour0 = int(ts.hour)

            for h in range(1, hz + 1):
                w  = float(preds.iloc[s][f"wind_mw+h{h}"])
                so = float(preds.iloc[s][f"solar_mw+h{h}"])
                ld = float(preds.iloc[s][f"load_mw+h{h}"])
                pr = float(preds.iloc[s][f"price+h{h}"])

                w_int  = int(round(w  * SCALE))
                so_int = int(round(so * SCALE))
                ld_int = int(round(ld * SCALE))
                pr_int = int(round(pr * SCALE))

                f.write(f"pred(wind,{s},{h},{w_int}).\n")
                f.write(f"pred(solar,{s},{h},{so_int}).\n")
                f.write(f"pred(load,{s},{h},{ld_int}).\n")
                f.write(f"pred(price,{s},{h},{pr_int}).\n")

                # simple horizon hour-of-day
                hod_h = (hour0 + h) % 24
                if hod_h in night_hours:
                    f.write(f"night({s},{h}).\n")

def parse_repairs(clingo_output):
    """
    Parse the repair actions returned by clingo.
    Example of output: repair(bound_cap, solar, 12, 3)
    """
    repairs = []
    for line in clingo_output.splitlines():
        line = line.strip()
        if not line or line.startswith("Answer") or line.startswith("Optimization"):
            continue
        atoms = line.split()
        for a in atoms:
            if not a.startswith("repair("):
                continue
            inner = a[len("repair("):-1]  # drop 'repair(' and trailing ')'
            kind, target, s, h = inner.split(",")
            repairs.append((kind, target, int(s), int(h)))
    return repairs

def apply_repairs(preds, caps, repairs):
    preds = preds.copy()
    for kind, target, s, h in repairs:
        if target in ("wind", "solar", "load"):
            col = f"{target}_mw+h{h}"
        else:
            col = f"price+h{h}"

        col_idx = preds.columns.get_loc(col)

        if kind == "bound_low":
            preds.iloc[s, col_idx] = max(preds.iloc[s, col_idx], 0.0)

        elif kind == "bound_cap" and target in ("wind", "solar"):
            cap_col = "cap_wind_mw" if target == "wind" else "cap_solar_mw"
            cap_val = float(caps.iloc[s][cap_col])
            preds.iloc[s, col_idx] = min(preds.iloc[s, col_idx], cap_val)

        elif kind == "pv_night" and target == "solar":
            preds.iloc[s, col_idx] = 0.0

    return preds

def run(cfg_path):
    cfg = load_config(cfg_path)

    preds_p = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions.parquet"
    if not preds_p.exists():
        raise FileNotFoundError(preds_p)

    preds = pd.read_parquet(preds_p)

    # get test_df with capacities (rebuild from engineered CSV via your pipeline)
    train_df, val_df, test_df = build_master(cfg)
    ctx = cfg["features"]["context_hours"]
    N = len(preds)
    caps = test_df[["cap_wind_mw", "cap_solar_mw"]].iloc[ctx:ctx+N].reset_index(drop=True)

    # Generate ASP facts
    facts_path = Path("src") / "asp" / "facts.lp"
    facts_path.parent.mkdir(exist_ok=True)
    write_facts(cfg, preds, caps, facts_path)


    # Call clingo
    result = subprocess.run(
        [
            "clingo",
            "src/asp/core_asp.lp",
            "src/asp/weather_asp.lp",
            "src/asp/market_asp.lp",
            str(facts_path),
            "--opt-mode=opt",
            "--quiet=1",
        ],
        text=True,
        capture_output=True,
        check=False,  # don't raise on non-zero exit; we'll handle it
    )
    
    if result.returncode != 0:
        print(f"clingo exited with code {result.returncode}")
        print(f"STDERR:\n{result.stderr}")
        print(f"STDOUT:\n{result.stdout}")
        if result.returncode == 65:
            print("(Exit code 65 usually means UNSATISFIABLE: the ASP constraints cannot be satisfied)")
            print("Falling back to Python ASP rules...")
            # Use Python fallback
            preds_reset = preds.reset_index(drop=True)
            preds_with_caps = preds.copy()
            preds_with_caps["cap_wind_mw"] = caps["cap_wind_mw"].values
            preds_with_caps["cap_solar_mw"] = caps["cap_solar_mw"].values

            
            wind_cols = [c for c in preds_with_caps.columns if c.startswith("wind_mw+h")]
            solar_cols = [c for c in preds_with_caps.columns if c.startswith("solar_mw+h")]
            load_cols = [c for c in preds_with_caps.columns if c.startswith("load_mw+h")]
            price_cols = [c for c in preds_with_caps.columns if c.startswith("price+h")]
            
            cleaned, flags = apply_asp_bounds(
                preds_with_caps,
                wind_cols=wind_cols,
                solar_cols=solar_cols,
                load_cols=load_cols,
                price_cols=price_cols,
                pv_night_hours=tuple(cfg["asp"]["pv_night_hours"]),
                ramp_limits=None,
            )
            out_p = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions_asp.parquet"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cleaned.set_index(preds.index, inplace=True)
            cleaned = cleaned.drop(columns=["cap_wind_mw", "cap_solar_mw"], errors="ignore")
            cleaned.to_parquet(out_p)
            print(f"Python ASP applied; saved {out_p}, num adjusted rows={flags['adjusted'].sum()}")
            return

    repairs = parse_repairs(result.stdout)

    # Apply repairs to predictions
    cleaned = apply_repairs(preds, caps, repairs)
    out_p = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions_asp.parquet"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cleaned.set_index(preds.index).to_parquet(out_p)
    print(f"ASP applied via clingo; saved {out_p}, num repairs={len(repairs)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
