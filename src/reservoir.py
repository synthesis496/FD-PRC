import numpy as np

PHI = (1 + np.sqrt(5)) / 2

def generate_fibonacci_time(n_steps, t0=1.0):
    n = np.arange(n_steps)
    fib_n = np.round((PHI**n - (-PHI)**-n) / np.sqrt(5)).astype(float)
    t_linear = np.mod(fib_n, n_steps)
    return t_linear + np.mod(t_linear * PHI, 1.0)

def mirror_operator(S, C=0.0):
    return C - np.flip(S)

def fold_operator(t):
    return np.mod(t * PHI, 1.0)

def limit_cycle_map(z, t_fib, beta=PHI):
    return (z / (np.abs(z) + 1e-9)) * np.exp(1j * beta * t_fib)

def parity_project(S, C=0.0):
    return C - S

def reservoir_step(psi, t_fib, lambda_val=1.0, C=0.0):
    dt = 0.01
    t_self_similar = t_fib * (1.0 + fold_operator(t_fib))
    M_op = mirror_operator(psi, C)
    d_psi = -1j * dt * (t_self_similar * M_op * psi)
    psi_bound = limit_cycle_map(psi + d_psi, t_fib)
    psi_next = parity_project(psi_bound, C)
    structural_absence = np.sum(psi_next + mirror_operator(psi_next, C) - C)
    return psi_next, structural_absence
