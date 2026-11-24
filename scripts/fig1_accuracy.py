import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fdprc.reservoir import FDReservoir
from fdprc.healing import HealingLayer
from fdprc.readout import ridge_fit, ridge_predict, rmse, r2
from fdprc.data import sine_signal, collect

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

u = sine_signal(noise=0.05)
fd = FDReservoir()
heal = HealingLayer(fd.psi)
X = collect(fd, u, heal=heal)
n_tr = int(0.7 * len(u))
w = ridge_fit(X[:n_tr - 1], u[1:n_tr])
yhat = ridge_predict(X[n_tr:-1], w)
truth = u[n_tr + 1:]

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(truth, label="truth", lw=1.2, color="black")
ax.plot(yhat, label="FD-PRC", lw=1.2, alpha=0.85, color="tab:blue")
ax.set_title(f"One-step prediction   RMSE={rmse(truth, yhat):.4f}   R²={r2(truth, yhat):.3f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig1_accuracy.png", dpi=150)
