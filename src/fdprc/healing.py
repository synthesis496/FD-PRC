import numpy as np
from .reservoir import FIB_CACHE


class HealingLayer:
    def __init__(self, initial_psi, gamma_heal=0.08, p=0.5, e_omni=1.0):
        self.reference = np.array(initial_psi, copy=True)
        self.gamma_heal = gamma_heal
        self.p = p
        self.e_omni = e_omni

    def _fib_pair(self, t_idx):
        i = int(np.minimum(t_idx + 1, len(FIB_CACHE) - 1))
        j = int(np.maximum(i - 1, 0))
        return FIB_CACHE[i], FIB_CACHE[j]

    def apply(self, psi, t_idx):
        f_cur, f_prev = self._fib_pair(t_idx)
        t_fib = f_cur / f_prev
        err = float(np.linalg.norm(psi - self.reference))
        kick = self.gamma_heal * t_fib * self.p * self.e_omni * err
        out = psi + kick * (self.reference - psi)
        return out / (np.linalg.norm(out) + 1e-12)
