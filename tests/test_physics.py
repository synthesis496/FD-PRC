import numpy as np
from fdprc.reservoir import FDReservoir, FIB_CACHE
from fdprc.healing import HealingLayer


def test_limit_cycle_bounded_orbit():
    fd = FDReservoir(seed=42)
    norms = [np.linalg.norm(fd.step(0.0, i)) for i in range(5000)]
    assert max(norms) < 1.5
    assert min(norms) > 0.5


def test_dissipation_forgets():
    fd = FDReservoir(seed=42)
    fd.inject(1.0, 0)
    psi_inj = fd.psi.copy()
    for _ in range(500):
        fd.dissipate(1.0)
    sim = np.abs(np.vdot(psi_inj, fd.psi))
    assert sim < 0.1


def test_mirror_involution():
    fd = FDReservoir(seed=42)
    psi_init = fd.psi.copy()
    fd.mirror()
    fd.mirror()
    assert np.allclose(np.abs(psi_init), np.abs(fd.psi))


def test_fibonacci_convergence():
    ratios = FIB_CACHE[1:] / FIB_CACHE[:-1]
    phi = (1 + np.sqrt(5)) / 2
    assert np.isclose(ratios[-1], phi, atol=1e-3)


def test_healing_converges():
    fd = FDReservoir(seed=42)
    heal = HealingLayer(fd.psi.copy())
    reference = fd.psi.copy()
    fd.inject(5.0, 0)
    err_init = np.linalg.norm(fd.psi - reference)
    for i in range(50):
        fd.psi = heal.apply(fd.psi, i)
    err_final = np.linalg.norm(fd.psi - reference)
    assert err_final < err_init
