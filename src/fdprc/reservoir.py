import numpy as np

FIB_CACHE = np.array(
    [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584],
    dtype=np.int64,
)


class IdentityHeal:
    def apply(self, psi, t_idx):
        return psi


IDENTITY_HEAL = IdentityHeal()


class FDReservoir:
    def __init__(self, n_qudits=8, d=4, gamma=0.15, a=-0.05, beta=0.4, mu=0.0, seed=42):
        self.n = n_qudits
        self.d = d
        self.gamma = gamma
        self.a = a
        self.beta = beta
        self.mu = mu
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal(n_qudits * d) + 1j * rng.standard_normal(n_qudits * d)
        self.psi = psi / np.linalg.norm(psi)

    def _fib_pair(self, t_idx):
        i = int(np.minimum(t_idx + 1, len(FIB_CACHE) - 1))
        j = int(np.maximum(i - 1, 0))
        return FIB_CACHE[i], FIB_CACHE[j]

    def inject(self, u, t_idx):
        f_cur, f_prev = self._fib_pair(t_idx)
        amp = 0.15 * np.log1p(f_cur)
        phase = 2 * np.pi * (f_cur / f_prev)
        rng = np.random.default_rng(int(t_idx))
        pattern = rng.standard_normal(self.psi.shape)
        self.psi = self.psi + amp * u * pattern * np.exp(1j * phase)
        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-12)

    def mirror(self):
        self.psi = 2 * self.mu - np.conj(self.psi)
        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-12)

    def dissipate(self, dt=1.0):
        self.psi = self.psi * np.exp(-self.gamma * dt)
        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-12)

    def limit_cycle(self, dt=1.0):
        self.psi = np.exp(self.a * dt + 1j * self.beta * self.psi * dt)
        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-12)

    def observables(self):
        amp = np.abs(self.psi)
        phase = np.angle(self.psi)
        power = amp ** 2
        return np.concatenate([amp, phase, power])

    def step(self, u, t_idx, dt=1.0, heal=IDENTITY_HEAL):
        self.inject(u, t_idx)
        self.limit_cycle(dt)
        self.mirror()
        self.dissipate(dt)
        self.psi = heal.apply(self.psi, t_idx)
        return self.observables()


class FDNoFib(FDReservoir):
    def inject(self, u, t_idx):
        super().inject(u, 0)


class FDNoMirror(FDReservoir):
    def mirror(self):
        return None


class FDNoLimitCycle(FDReservoir):
    def limit_cycle(self, dt=1.0):
        return None
