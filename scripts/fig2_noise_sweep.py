import numpy as np
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

SEEDS = [42, 7, 13, 99, 123]
NOISES = [0.0, 0.1, 0.2, 0.3, 0.5]


def run_single(nz, seed):
    u = sine_signal(noise=nz, seed=seed)
    n_tr = int(0.7 * len(u))

    fd = FDReservoir(seed=seed)
    X = collect(fd, u)
    w = ridge_fit(X[:n_tr - 1], u[1:n_tr])
    err_no = rmse(u[n_tr + 1:], ridge_predict(X[n_tr:-1], w))

    fd2 = FDReservoir(seed=seed)
    heal2 = HealingLayer(fd2.psi)
    X2 = collect(fd2, u, heal=heal2)
    w2 = ridge_fit(X2[:n_tr - 1], u[1:n_tr])
    err_heal = rmse(u[n_tr + 1:], ridge_predict(X2[n_tr:-1], w2))

    return err_no, err_heal


def main():
    results_no = []
    results_heal = []

    for nz in NOISES:
        runs = [run_single(nz, s) for s in SEEDS]
        results_no.append([r[0] for r in runs])
        results_heal.append([r[1] for r in runs])

    mean_no = np.mean(results_no, axis=1)
    std_no = np.std(results_no, axis=1)
    mean_heal = np.mean(results_heal, axis=1)
    std_heal = np.std(results_heal, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(NOISES, mean_no, "o-", label="FD-PRC (no heal)", color="tab:red")
    ax.fill_between(NOISES, mean_no - std_no, mean_no + std_no, color="tab:red", alpha=0.2)

    ax.plot(NOISES, mean_heal, "s-", label="FD-PRC + healing", color="tab:blue")
    ax.fill_between(NOISES, mean_heal - std_heal, mean_heal + std_heal, color="tab:blue", alpha=0.2)

    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Graceful Degradation (5 Seeds ± Std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_noise_sweep.png", dpi=150)


main()
