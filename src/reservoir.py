import numpy as np

PHI = (1 + np.sqrt(5)) / 2
SQRT5 = np.sqrt(5)

def generate_fibonacci_time(n_steps, t0=1.0):
    n = np.arange(n_steps)
    fib_n = np.round((PHI**n - (-PHI)**-n) / SQRT5).astype(int)
    t_linear = np.linspace(0, fib_n[-1] * t0, fib_n[-1] * 10)
    t_fib = t_linear + np.mod(t_linear * PHI, 1.0) * t0
    return t_fib

def mirror_operator(S, C=0.0):
    return C - np.flip(S)

def fold_operator(t):
    return np.mod(t * PHI, 1.0)

def limit_cycle_map(z, a=-0.05, beta=PHI):
    return np.exp((a + 1j * beta) * z)

def reservoir_step(psi, t_fib, lambda_val=1.0):
    dt = 0.01
    t_self_similar = t_fib * (1.0 + fold_operator(t_fib))
    M_op = mirror_operator(psi)
    d_psi = -1j * dt * (t_self_similar * M_op * psi)
    psi_next = limit_cycle_map(psi + d_psi)
    structural_absence = np.sum(psi_next + mirror_operator(psi_next) - lambda_val * psi_next)
    return psi_next, structural_absence
