import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fdprc.reservoir import FDReservoir
from fdprc.data import sine_signal, collect
from fdprc.readout_fixed import RidgeReadout, HebbianReadout, ZeroShotReadout

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    u = sine_signal(noise=0.1)
    n_tr = int(0.7 * len(u))
    
    fd = FDReservoir()
    X = collect(fd, u)
    
    X_train, y_train = X[:n_tr - 1], u[1:n_tr]
    X_test, y_test = X[n_tr:-1], u[n_tr + 1:]
    
    models = {
        "Ridge (L2)": RidgeReadout(alpha=1e-6),
        "Hebbian (1-pass)": HebbianReadout(eta=1e-3),
        "Zero-Shot (Random)": ZeroShotReadout(seed=42),
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = np.sqrt(np.mean((y_test - pred) ** 2))

    names = list(results.keys())
    vals = [results[k] for k in names]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color=colors, alpha=0.8)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Readout Comparison: Proving Zero-Training works")
    ax.grid(alpha=0.3, axis="y")
    
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01 * max(vals), f"{v:.4f}", ha="center", va="bottom")
        
    fig.tight_layout()
    fig.savefig(OUT / "fig6_readout_compare.png", dpi=150)


main()
