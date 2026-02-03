"python scripts/rank_and_plot_benchmarks.py"


import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASTER = "benchmarks_master.csv"

def safe_log(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.log10(np.clip(x, 1e-12, None))

def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    vals = s.values.astype(float)

    if np.all(~np.isfinite(vals)):
        return pd.Series(np.full(len(s), 0.5), index=s.index)

    lo = np.nanmin(vals)
    hi = np.nanmax(vals)

    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.full(len(s), 0.5), index=s.index)

    return (s - lo) / (hi - lo)

def pareto_front_2d(df, x_col, y_col):
    """Lower is better for both x and y."""
    d = df.dropna(subset=[x_col, y_col]).copy().sort_values(x_col, ascending=True)
    best_y = np.inf
    pareto = np.zeros(len(d), dtype=bool)
    for i, y in enumerate(d[y_col].values):
        if y < best_y:
            pareto[i] = True
            best_y = y
    d["is_pareto"] = pareto
    return d

def build_scores(df, w_acc=0.7, w_comp=0.3, w_lat=0.5, w_comp_lat=0.5):
    """
    IMPORTANT:
      Uses Latency_ms_per_sample as-is.
      Your ASP scripts already store E2E latency in ASP summary CSV (combined raw + asp time),
      so we do NOT add ASP time here.
    """
    df = df.copy()
    group_cols = ["task", "lookback", "horizon"]

    # Ensure numeric columns
    for c in ["Train_CPU_Sec","Infer_CPU_Sec","Peak_RAM_MB","Latency_ms_per_sample","RMSE","BASE_RMSE","Params","Deploy_Params"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Backward compat: if Deploy_Params missing, fallback to Params
    if "Deploy_Params" not in df.columns or df["Deploy_Params"].isna().all():
        df["Deploy_Params"] = df.get("Params", np.nan)

    # Ensure RMSE_ratio exists (you already have it, but keep safe)
    if "RMSE_ratio" not in df.columns or df["RMSE_ratio"].isna().all():
        df["RMSE_ratio"] = df["RMSE"] / (df["BASE_RMSE"] + 1e-12)

    # Log transforms (stabilize scales)
    df["_log_latency"]  = safe_log(df["Latency_ms_per_sample"])
    df["_log_traincpu"] = safe_log(df["Train_CPU_Sec"])
    df["_log_infercpu"] = safe_log(df["Infer_CPU_Sec"])
    df["_log_ram"]      = safe_log(df["Peak_RAM_MB"])
    df["_log_params"]   = safe_log(df["Deploy_Params"])

    # Normalize within each (task,LB,H) for fairness
    df["acc_n"]      = df.groupby(group_cols)["RMSE_ratio"].transform(minmax_norm)
    df["lat_n"]      = df.groupby(group_cols)["_log_latency"].transform(minmax_norm)
    df["traincpu_n"] = df.groupby(group_cols)["_log_traincpu"].transform(minmax_norm)
    df["infercpu_n"] = df.groupby(group_cols)["_log_infercpu"].transform(minmax_norm)
    df["ram_n"]      = df.groupby(group_cols)["_log_ram"].transform(minmax_norm)
    df["param_n"]    = df.groupby(group_cols)["_log_params"].transform(minmax_norm)

    # Compute power proxy (no latency inside)
    df["compute_power_n"] = df[["traincpu_n","infercpu_n","ram_n","param_n"]].mean(axis=1)

    # Rankings
    df["compute_only_score"] = df["compute_power_n"]
    df["compute_latency_score"] = w_comp_lat * df["compute_power_n"] + w_lat * df["lat_n"]
    df["final_score"] = w_acc * df["acc_n"] + w_comp * df["compute_power_n"]

    return df

def summarize_by_model(df_scored):
    group_cols = ["task","lookback","horizon"]

    # Pareto per setting: latency vs accuracy ratio
    pareto_rows = []
    for _, g in df_scored.groupby(group_cols):
        pareto_rows.append(pareto_front_2d(g, "Latency_ms_per_sample", "RMSE_ratio"))
    pareto_df = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else df_scored.copy()

    df_scored = df_scored.merge(
        pareto_df[["source_file","model","variant","task","lookback","horizon","is_pareto"]],
        on=["source_file","model","variant","task","lookback","horizon"],
        how="left"
    )

    g = df_scored.groupby(["model","variant"], dropna=False)
    summary = g.agg(
        n_rows=("final_score","count"),

        mean_final_score=("final_score","mean"),
        mean_compute_only=("compute_only_score","mean"),
        mean_compute_latency=("compute_latency_score","mean"),

        mean_RMSE_ratio=("RMSE_ratio","mean"),
        median_RMSE_ratio=("RMSE_ratio","median"),

        mean_latency_ms=("Latency_ms_per_sample","mean"),
        median_latency_ms=("Latency_ms_per_sample","median"),

        mean_train_cpu_sec=("Train_CPU_Sec","mean"),
        mean_infer_cpu_sec=("Infer_CPU_Sec","mean"),
        mean_peak_ram_mb=("Peak_RAM_MB","mean"),
        mean_deploy_params=("Deploy_Params","mean"),

        pareto_share=("is_pareto", lambda s: float(np.nanmean(s.astype(float))) if len(s) else np.nan),
    ).reset_index()

    summary["tag"] = summary["model"].astype(str) + " (" + summary["variant"].astype(str) + ")"

    # Ranks
    summary = summary.sort_values("mean_final_score", ascending=True).reset_index(drop=True)
    summary["overall_rank"] = np.arange(1, len(summary) + 1)

    tmp = summary.sort_values("mean_compute_only", ascending=True).reset_index(drop=True)
    tmp["compute_rank"] = np.arange(1, len(tmp) + 1)
    summary = summary.merge(tmp[["tag","compute_rank"]], on="tag", how="left")

    tmp = summary.sort_values("mean_compute_latency", ascending=True).reset_index(drop=True)
    tmp["compute_latency_rank"] = np.arange(1, len(tmp) + 1)
    summary = summary.merge(tmp[["tag","compute_latency_rank"]], on="tag", how="left")

    tmp = summary.sort_values("mean_latency_ms", ascending=True).reset_index(drop=True)
    tmp["latency_rank"] = np.arange(1, len(tmp) + 1)
    summary = summary.merge(tmp[["tag","latency_rank"]], on="tag", how="left")

    summary = summary.sort_values("overall_rank", ascending=True).reset_index(drop=True)
    return summary, df_scored

def plot_bar(summary, value_col, title, out_png):
    s = summary.sort_values(value_col, ascending=True).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(s["tag"], s[value_col].values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(value_col + " (lower is better)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

def plot_pareto(df_scored, out="pareto_points.png"):
    d = df_scored.dropna(subset=["RMSE_ratio","Latency_ms_per_sample"]).copy()
    plt.figure(figsize=(10, 6))

    npart = d[d["is_pareto"] != True]
    plt.scatter(npart["Latency_ms_per_sample"], npart["RMSE_ratio"], alpha=0.15)

    p = d[d["is_pareto"] == True].copy()
    for (m,v), g in p.groupby(["model","variant"]):
        plt.scatter(g["Latency_ms_per_sample"], g["RMSE_ratio"], alpha=0.85, label=f"{m} ({v})")

    plt.xscale("log")
    plt.xlabel("Latency (ms/sample) [log]")
    plt.ylabel("RMSE_ratio (lower is better)")
    plt.title("Pareto Highlight: Accuracy vs Latency")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_acc", type=float, default=0.7)
    ap.add_argument("--w_comp", type=float, default=0.3)
    ap.add_argument("--w_lat", type=float, default=0.5)
    ap.add_argument("--w_comp_lat", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(MASTER)

    # normalize task casing
    df["task"] = df["task"].astype(str).str.upper()

    # guard grid if you want
    if "lookback" in df.columns:
        df = df[df["lookback"].isin([24, 72, 168])]
    if "horizon" in df.columns:
        df = df[df["horizon"].isin([12, 24, 72])]

    df_scored = build_scores(df, w_acc=args.w_acc, w_comp=args.w_comp, w_lat=args.w_lat, w_comp_lat=args.w_comp_lat)
    summary, df_scored = summarize_by_model(df_scored)

    summary.to_csv("model_ranking_with_compute_latency.csv", index=False)
    df_scored.to_csv("benchmarks_scored.csv", index=False)

    print("[OK] Saved:")
    print(" - model_ranking_with_compute_latency.csv")
    print(" - benchmarks_scored.csv")

    plot_bar(summary, "mean_final_score", "Overall Ranking (Accuracy + Compute Power)", "rank_overall.png")
    plot_bar(summary, "mean_compute_only", "Compute-Only Ranking (CPU+RAM+Params)", "rank_compute_only.png")
    plot_bar(summary, "mean_compute_latency", "Compute+Latency Ranking (CPU+RAM+Params + Latency)", "rank_compute_latency.png")
    plot_bar(summary, "mean_latency_ms", "Latency Ranking (mean ms/sample)", "rank_latency.png")

    plot_pareto(df_scored, out="pareto_points.png")

if __name__ == "__main__":
    main()
