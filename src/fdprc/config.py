from dataclasses import dataclass, field

from .reservoir import FDReservoir
from .healing import HealingLayer
from .readout import ridge_fit


@dataclass(frozen=True)
class ReservoirConfig:
    n_qudits: int = 8
    d: int = 4
    gamma: float = 0.15
    a: float = -0.05
    beta: float = 0.4
    mu: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class HealingConfig:
    gamma_heal: float = 0.08
    p: float = 0.5
    e_omni: float = 1.0


@dataclass(frozen=True)
class ReadoutConfig:
    alpha: float = 1e-6


@dataclass(frozen=True)
class DataConfig:
    T: int = 2000
    noise: float = 0.05
    seed: int = 0
    train_frac: float = 0.7


@dataclass(frozen=True)
class ExperimentConfig:
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    healing: HealingConfig = field(default_factory=HealingConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    data: DataConfig = field(default_factory=DataConfig)


DEFAULT = ExperimentConfig()


def build_reservoir(cfg: ExperimentConfig = DEFAULT) -> FDReservoir:
    r = cfg.reservoir
    return FDReservoir(
        n_qudits=r.n_qudits,
        d=r.d,
        gamma=r.gamma,
        a=r.a,
        beta=r.beta,
        mu=r.mu,
        seed=r.seed,
    )


def build_healing(reservoir: FDReservoir, cfg: ExperimentConfig = DEFAULT) -> HealingLayer:
    h = cfg.healing
    return HealingLayer(
        initial_psi=reservoir.psi,
        gamma_heal=h.gamma_heal,
        p=h.p,
        e_omni=h.e_omni,
    )


def build_readout(cfg: ExperimentConfig = DEFAULT):
    alpha = cfg.readout.alpha

    def fit(X, y):
        return ridge_fit(X, y, alpha=alpha)

    return fit
