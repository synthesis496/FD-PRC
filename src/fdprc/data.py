import numpy as np
from .reservoir import IDENTITY_HEAL


def sine_signal(T=2000, noise=0.0, seed=0):
    t = np.linspace(0, 20 * np.pi, T)
    y = np.sin(t) + 0.3 * np.sin(2.7 * t)
    rng = np.random.default_rng(seed)
    return y + noise * rng.standard_normal(T)


def collect(fd, u_series, heal=IDENTITY_HEAL):
    return np.array([fd.step(u, i, heal=heal) for i, u in enumerate(u_series)])
