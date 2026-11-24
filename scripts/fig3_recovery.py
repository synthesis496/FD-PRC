import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fdprc.reservoir import FDReservoir
from fdprc.healing import HealingLayer
from fdprc.readout import ridge_fit, ridge_predict
from fdprc.data import sine_signal, collect

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

u = sine_signal(noise=0.05)
n_tr = int(0.5 * len(u))
perturb_at = n_tr + 20
perturb = np.zeros(len(u))
perturb[perturb_at] = 15.0

fd_no = FDReservoir()
X_no = collect(fd_no, u[:n_tr])
w_no = ridge_fit(X_no[:-1], u[1:n_tr])
err_no = [abs(ridge_predict(fd_no.step(u[i] + perturb[i], i), w_no) - u[i + 1])
          for i in range(n_tr, len(u) - 1)]

fd_h = FDReservoir()
heal = HealingLayer(fd_h.psi)
X_h = collect(fd_h, u[:n_tr], heal=heal)
w_h = ridge_fit(X_h[:-1], u[1:n_tr])
err_h = [abs(ridge_predict(fd_h.step(u[i] + perturb[i], i, heal=heal), w_h) - u[i + 1])
         for i in range(n_tr, len(u) - 1)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(err_no, label="FD-PRC (no heal)", lw=1.3)
ax.plot(err_h, label="FD-PRC + healing", lw=1.3)
ax.axvline(perturb_at - n_tr, color="red", ls="--", alpha=0.5, label="perturbation")
ax.set_xlabel("steps after train/test split")
ax.set_ylabel("absolute error")
ax.set_title("Recovery after perturbation")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_recovery.png", dpi=150)
