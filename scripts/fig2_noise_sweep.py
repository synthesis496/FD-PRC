import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fdprc.reservoir import FDReservoir
from fdprc.healing import HealingLayer
from fdprc.readout import ridge_fit, ridge_predict, rmse
from fdprc.data import sine_signal, collect

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

noises = [0.0, 0.1, 0.2, 0.3, 0.5]
rmse_no_heal = []
rmse_heal = []

for nz in noises:
    u = sine_signal(noise=nz)
    n_tr = int(0.7 * len(u))

    fd = FDReservoir()
    X = collect(fd, u)
    w = ridge_fit(X[:n_tr - 1], u[1:n_tr])
    rmse_no_heal.append(rmse(u[n_tr + 1:], ridge_predict(X[n_tr:-1], w)))

    fd2 = FDReservoir()
    heal2 = HealingLayer(fd2.psi)
    X2 = collect(fd2, u, heal=heal2)
    w2 = ridge_fit(X2[:n_tr - 1], u[1:n_tr])
    rmse_heal.append(rmse(u[n_tr + 1:], ridge_predict(X2[n_tr:-1], w2)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(noises, rmse_no_heal, "o-", label="FD-PRC (no heal)")
ax.plot(noises, rmse_heal, "s-", label="FD-PRC + healing")
ax.set_xlabel("noise σ")
ax.set_ylabel("RMSE")
ax.set_title("Graceful degradation under noise")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig2_noise_sweep.png", dpi=150)
