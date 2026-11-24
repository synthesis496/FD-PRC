import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fdprc.reservoir import FDReservoir

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    fd = FDReservoir(seed=42)
    u = np.sin(np.linspace(0, 20, 1000))

    for i, val in enumerate(u):
        fd.step(val, i)

    states = []
    for i in range(1000):
        fd.step(0.0, i)
        states.append(fd.psi[0])

    states = np.array(states)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(states.real, states.imag, lw=1, alpha=0.7, color="tab:purple")
    ax.set_xlabel(r"Re(\(\psi_0\))")
    ax.set_ylabel(r"Im(\(\psi_0\))")
    ax.set_title("Phase Space Trajectory: Limit Cycle Attractor")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_trajectory.png", dpi=150)


main()
