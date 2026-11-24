import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fdprc.reservoir import FDReservoir, FDNoFib, FDNoMirror, FDNoLimitCycle
from fdprc.readout import ridge_fit, ridge_predict, rmse
from fdprc.data import sine_signal, collect

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

u = sine_signal(noise=0.1)
n_tr = int(0.7 * len(u))

models = {
    "Full FD-PRC":    FDReservoir(),
    "No Fibonacci":   FDNoFib(),
    "No Mirror":      FDNoMirror(),
    "No LimitCycle":  FDNoLimitCycle(),
    "No Dissipation": FDReservoir(gamma=0.0),
}

results = {}
for name, fd in models.items():
    X = collect(fd, u)
    w = ridge_fit(X[:n_tr - 1], u[1:n_tr])
    results[name] = rmse(u[n_tr + 1:], ridge_predict(X[n_tr:-1], w))

names = list(results.keys())
vals = [results[k] for k in names]
colors = ["tab:blue"] + ["tab:gray"] * (len(names) - 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(names, vals, color=colors)
ax.set_ylabel("RMSE")
ax.set_title("Pillar ablation")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig4_ablation.png", dpi=150)
