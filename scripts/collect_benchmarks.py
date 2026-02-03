"python scripts/collect_benchmarks.py"


from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Canonical columns (union schema)
# -----------------------------
CANON_COLS = [
    "task", "lookback", "horizon",

    "MAE", "RMSE", "sMAPE",
    "BASE_MAE", "BASE_RMSE", "BASE_sMAPE",

    "Params",
    "Train_Params", "Deploy_Params",
    "Size_MB",
    "Train_Size_MB", "Deploy_Size_MB",

    "Train_Wall_Sec", "Train_CPU_Sec", "Avg_CPU_Usage_Pct", "Peak_RAM_MB",
    "Infer_Wall_Sec", "Infer_CPU_Sec", "Infer_Avg_CPU_Pct",
    "Latency_ms_per_sample",
]

ALIASES = {
    # older baseline naming variants (if any)
    "Train_Avg_CPU_Pct": "Avg_CPU_Usage_Pct",
    "Train_Peak_RSS_MB": "Peak_RAM_MB",
}

NUMERIC_COLS = [
    "lookback", "horizon",
    "MAE", "RMSE", "sMAPE", "BASE_MAE", "BASE_RMSE", "BASE_sMAPE",
    "Params", "Train_Params", "Deploy_Params",
    "Size_MB", "Train_Size_MB", "Deploy_Size_MB",
    "Train_Wall_Sec", "Train_CPU_Sec", "Avg_CPU_Usage_Pct", "Peak_RAM_MB",
    "Infer_Wall_Sec", "Infer_CPU_Sec", "Infer_Avg_CPU_Pct",
    "Latency_ms_per_sample",
]

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Apply aliases
    for old, new in ALIASES.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Ensure canonical columns exist
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Recover legacy Params/Size_MB if missing
    if df["Params"].isna().all() and not df["Deploy_Params"].isna().all():
        df["Params"] = df["Deploy_Params"]
    if df["Size_MB"].isna().all() and not df["Deploy_Size_MB"].isna().all():
        df["Size_MB"] = df["Deploy_Size_MB"]

    # Fill train/deploy if only legacy exists (safety)
    if df["Train_Params"].isna().all() and not df["Params"].isna().all():
        df["Train_Params"] = df["Params"]
    if df["Deploy_Params"].isna().all() and not df["Params"].isna().all():
        df["Deploy_Params"] = df["Params"]
    if df["Train_Size_MB"].isna().all() and not df["Size_MB"].isna().all():
        df["Train_Size_MB"] = df["Size_MB"]
    if df["Deploy_Size_MB"].isna().all() and not df["Size_MB"].isna().all():
        df["Deploy_Size_MB"] = df["Size_MB"]

    # Coerce numeric
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize task label
    df["task"] = df["task"].astype(str).str.upper()

    return df

def tag_model_variant(path: Path) -> tuple[str, str]:
    p = str(path).lower()
    name = path.name.lower()

    # Baselines by folder
    if "artifacts_lstm_baseline" in p:
        return "LSTM", "baseline"
    if "artifacts_cnn_baseline" in p:
        return "CNN", "baseline"

    # dCeNN summaries by filename
    if name.startswith("summary_dcenn_"):
        variant = "asp" if "asp" in name else "raw"
        model = "dCeNN+ELM+ASP" if variant == "asp" else "dCeNN+ELM"
        return model, variant

    return "UNKNOWN", "unknown"

def collect_paths() -> list[Path]:
    paths = []

    # Baselines (your exact structure)
    paths += sorted(Path("artifacts_lstm_baseline").glob("summary_lb*.csv"))
    paths += sorted(Path("artifacts_cnn_baseline").glob("summary_lb*.csv"))

    # dCeNN (your exact structure)
    paths += sorted(Path("outputs_energy_full").glob("summary_dcenn_*.csv"))
    paths += sorted(Path("outputs_weather_full").glob("summary_dcenn_*.csv"))

    # De-dup
    seen = set()
    uniq = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen and p.exists():
            uniq.append(p)
            seen.add(rp)

    return uniq

def build_master(out_csv: str = "benchmarks_master.csv") -> pd.DataFrame:
    paths = collect_paths()
    if not paths:
        raise SystemExit("No summary CSVs found. Check folder names / file patterns.")

    frames = []
    for p in paths:
        df = normalize_df(pd.read_csv(p))
        model, variant = tag_model_variant(p)
        df["model"] = model
        df["variant"] = variant
        df["source_file"] = str(p)
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)

    # Cross-task comparable metrics (unitless)
    master["RMSE_ratio"] = master["RMSE"] / (master["BASE_RMSE"] + 1e-12)
    master["MAE_ratio"]  = master["MAE"]  / (master["BASE_MAE"]  + 1e-12)
    master["RMSE_gain"]  = 1.0 - master["RMSE_ratio"]  # higher is better

    # Remove duplicates if re-run
    key = ["model", "variant", "task", "lookback", "horizon"]
    master = master.sort_values("source_file").drop_duplicates(subset=key, keep="last")

    master.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with {len(master)} rows from {len(paths)} files.")

    # Quick sanity print
    print(master.groupby(["model","variant"]).size())

    return master

if __name__ == "__main__":
    build_master()
