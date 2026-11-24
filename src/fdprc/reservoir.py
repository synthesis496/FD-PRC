import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0

FIB_CACHE = np.array(,
    dtype=np.int64,
)


class IdentityHeal:
    def apply(self, psi, t_idx):
        return psi


IDENTITY_HEAL = IdentityHeal()


class FDReservoir:
    def __init__(self, n_qudits=8, d=4, gamma=0.35, a=-0.05, beta=PHI, mu=0.0, seed=42):
        self.n = n_qudits
        self.d = d
        self.dim = n_qudits * d
        self.gamma = gamma
        self.a = a
        self.beta = beta
        self.mu = mu

        rng = np.random.default_rng(seed)
        self.w_in = (rng.standard_normal(self.dim) + 1j * rng.standard_normal(self.dim)) * 0.5

        psi = rng.standard_normal(self.dim) + 1j * rng.standard_normal(self.dim)
        self.psi = psi / (np.linalg.norm(psi) + 1e-12)

    def _fib_time(self, t_idx):
        n = float(t_idx + 1)
        t_fib = (n * PHI) % 1.0
        t_fold = (t_fib * PHI) % 1.0
        return t_fib * (1.0 + t_fold)

    def _mirror_op(self, s):
        return self.mu - np.flip(s)

    def _limit_cycle_map(self, z, t_self_similar):
        mag = np.tanh(np.abs(z))
        phase = np.angle(z) + self.beta * t_self_similar
        return mag * np.exp(1j * phase)

    def observables(self):
        amp = np.abs(self.psi)
        phase = np.angle(self.psi)
        power = amp ** 2
        return np.concatenate([amp, phase, power])

    def step(self, u, t_idx, dt=0.1, heal=IDENTITY_HEAL):
        t_sim = self._fib_time(t_idx)
        drive = u * self.w_in * np.exp(1j * 2.0 * np.pi * t_sim)

        m_op = self._mirror_op(self.psi)
        d_psi = -1j * dt * (t_sim * m_op * self.psi) + drive

        psi_bound = self._limit_cycle_map(self.psi + d_psi, t_sim)
        psi_cand = self.mu - psi_bound

        leak = np.exp(-self.gamma * dt)
        self.psi = (1.0 - leak) * self.psi + leak * psi_cand

        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-12)
        self.psi = heal.apply(self.psi, t_idx)
        return self.observables()


class FDNoFib(FDReservoir):
    def _fib_time(self, t_idx):
        return 0.5


class FDNoMirror(FDReservoir):
    def _mirror_op(self, s):
        return np.zeros_like(s)


class FDNoLimitCycle(FDReservoir):
    def _limit_cycle_map(self, z, t_self_similar):
        return z
