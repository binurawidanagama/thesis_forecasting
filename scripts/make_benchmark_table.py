"python scripts/make_benchmark_table.py"


import pandas as pd

MASTER = "benchmarks_master.csv"

def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join([str(x) for x in col]).strip() for col in df.columns.values]
    return df

def main():
    df = pd.read_csv(MASTER)

    # keep the known grid (optional safety)
    df = df[df["lookback"].isin([24, 72, 168])]
    df = df[df["horizon"].isin([12, 24, 72])]
    df["task"] = df["task"].astype(str).str.upper()

    idx = ["task","lookback","horizon"]
    cols = ["model","variant"]

    # Accuracy table (unitless)
    t_acc = (
        df.pivot_table(index=idx, columns=cols, values="RMSE_ratio", aggfunc="mean")
          .sort_index()
    )
    t_acc = flatten_cols(t_acc)
    t_acc.to_csv("benchmark_table_rmse_ratio.csv")
    print("[OK] Saved benchmark_table_rmse_ratio.csv")

    # Latency table
    t_lat = (
        df.pivot_table(index=idx, columns=cols, values="Latency_ms_per_sample", aggfunc="mean")
          .sort_index()
    )
    t_lat = flatten_cols(t_lat)
    t_lat.to_csv("benchmark_table_latency_ms.csv")
    print("[OK] Saved benchmark_table_latency_ms.csv")

    # Winners table (lowest RMSE_ratio per setting)
    winners = (
        df.dropna(subset=["RMSE_ratio"])
          .assign(model_tag=lambda x: x["model"] + " (" + x["variant"] + ")")
          .sort_values(idx + ["RMSE_ratio"])
          .groupby(idx, as_index=False)
          .first()
          .loc[:, idx + ["model_tag","RMSE_ratio"]]
          .rename(columns={"model_tag": "best_model", "RMSE_ratio": "best_RMSE_ratio"})
          .sort_values(idx)
    )
    winners.to_csv("benchmark_winners_rmse_ratio.csv", index=False)
    print("[OK] Saved benchmark_winners_rmse_ratio.csv")

if __name__ == "__main__":
    main()
