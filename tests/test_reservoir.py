import numpy as np
import pytest

from fdprc.healing import HealingLayer
from fdprc.reservoir import (
    FIB_CACHE,
    FDNoFib,
    FDNoLimitCycle,
    FDNoMirror,
    FDReservoir,
)


def test_fibonacci_cache_is_valid_and_converges_to_phi():
    """Cache should be positive, increasing, and approach the golden ratio."""
    assert FIB_CACHE.ndim == 1
    assert len(FIB_CACHE) > 10
    assert np.all(FIB_CACHE > 0)
    assert np.all(np.diff(FIB_CACHE) > 0)

    ratios = FIB_CACHE[1:].astype(float) / FIB_CACHE[:-1].astype(float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    assert np.isclose(ratios[-1], phi, atol=1e-3)


@pytest.mark.parametrize(
    "reservoir_type",
    [FDReservoir, FDNoFib, FDNoMirror, FDNoLimitCycle],
)
def test_step_returns_finite_nonconstant_observables(reservoir_type):
    """Reservoir variants should produce finite, changing observables."""
    reservoir = reservoir_type(seed=42)
    outputs = np.asarray(
        [reservoir.step(float(value), index) for index, value in enumerate(
            np.linspace(-1.0, 1.0, 50)
        )]
    )

    assert outputs.shape == (50, 3 * reservoir.dim)
    assert np.isfinite(outputs).all()
    assert not np.allclose(outputs, 0.0)
    assert np.max(np.ptp(outputs, axis=0)) > 1e-12


def test_reservoir_state_remains_normalized_after_steps():
    """The internal complex state is normalized after every step."""
    reservoir = FDReservoir(seed=42)

    for index, value in enumerate(np.linspace(-1.0, 1.0, 100)):
        reservoir.step(float(value), index)
        assert np.isfinite(reservoir.psi).all()
        assert np.isclose(np.linalg.norm(reservoir.psi), 1.0, atol=1e-9)


def test_same_seed_and_inputs_produce_same_outputs():
    """Initialization should be reproducible when the seed is fixed."""
    first = FDReservoir(seed=123)
    second = FDReservoir(seed=123)
    inputs = np.linspace(-0.5, 0.5, 20)

    first_outputs = np.asarray(
        [first.step(float(value), i) for i, value in enumerate(inputs)]
    )
    second_outputs = np.asarray(
        [second.step(float(value), i) for i, value in enumerate(inputs)]
    )

    assert np.allclose(first_outputs, second_outputs)


def test_fibonacci_and_mirror_ablation_hooks_are_distinct():
    """Ablation variants should disable their respective operations."""
    full = FDReservoir(seed=42)
    no_fib = FDNoFib(seed=42)
    no_mirror = FDNoMirror(seed=42)

    assert full._fib_time(0) != no_fib._fib_time(0)
    assert np.allclose(no_mirror._mirror_op(full.psi), np.zeros_like(full.psi))


def test_healing_output_is_finite_and_normalized():
    """Healing should return a finite normalized state."""
    reservoir = FDReservoir(seed=42)
    healing = HealingLayer(reservoir.psi.copy())

    perturbed = reservoir.psi + (0.1 + 0.2j)
    healed = healing.apply(perturbed, t_idx=5)

    assert np.isfinite(healed).all()
    assert np.isclose(np.linalg.norm(healed), 1.0, atol=1e-9)
