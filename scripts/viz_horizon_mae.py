import json, matplotlib.pyplot as plt
from pathlib import Path

m = json.loads(Path("outputs/eval/metrics_test.json").read_text())

def series(maes):
    # keep horizons in order h1..h12
    xs, ys = [], []
    for k in sorted(maes.keys(), key=lambda s: int(s[1:])):
        xs.append(k)
        ys.append(maes[k]["MAE"])
    return xs, ys

plt.figure(figsize=(10,5))
for key, label in [("wind","Wind MAE"),("solar","Solar MAE"),("load","Load MAE"),("price","Price MAE")]:
    xs, ys = series(m[key])
    plt.plot(xs, ys, marker="o", label=label)
plt.grid(True, ls=":")
plt.legend()
plt.title("MAE per Horizon")
plt.xlabel("Horizon")
plt.ylabel("MAE")
out = Path("outputs")/"viz_mae_per_horizon.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
