"python scripts/plot_pareto.py"


import pandas as pd
import matplotlib.pyplot as plt

MASTER = "benchmarks_master.csv"

def main():
    df = pd.read_csv(MASTER)

    # Filter to usable rows
    df = df.dropna(subset=["RMSE_ratio", "Latency_ms_per_sample", "task", "lookback", "horizon"])
    df = df[df["lookback"].isin([24, 72, 168])]
    df = df[df["horizon"].isin([12, 24, 72])]

    # Clip ratios so one crazy run doesn't ruin the entire scale
    df["RMSE_ratio"] = df["RMSE_ratio"].clip(0, 5)

    plt.figure(figsize=(10, 6))

    for (model, variant), g in df.groupby(["model", "variant"]):
        plt.scatter(
            g["Latency_ms_per_sample"],
            g["RMSE_ratio"],
            label=f"{model} ({variant})",
            alpha=0.75,
        )

    plt.xscale("log")
    plt.xlabel("Latency (ms per sample) [log scale]")
    plt.ylabel("RMSE / Persistence_RMSE (lower is better)")
    plt.title("Accuracy–Efficiency Tradeoff Across All Tasks / Lookbacks / Horizons")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    out = "benchmark_pareto_all.png"
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"[OK] Saved {out}")

if __name__ == "__main__":
    main()
