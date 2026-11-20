import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.reservoir import generate_fibonacci_time, reservoir_step
from src.readout import train_readout, predict

def run_benchmark(n_steps=500, dim=20):
    t_fib = generate_fibonacci_time(n_steps)
    inputs = np.sin(np.arange(n_steps) * 0.1)
    psi = np.random.rand(dim) + 1j * np.random.rand(dim)
    
    states = []
    targets = []
    
    for i in range(n_steps - 1):
        psi, _ = reservoir_step(psi, t_fib[i])
        states.append(psi)
        targets.append(inputs[i + 1])
        
    split = int(len(states) * 0.8)
    X_train = np.array(states[:split])
    X_test = np.array(states[split:])
    Y_train = np.array(targets[:split])
    Y_test = np.array(targets[split:])
    
    W_out = train_readout(X_train, Y_train)
    predictions = np.array([predict(W_out, x) for x in X_test])
    mse = np.mean(np.abs(predictions - Y_test) ** 2)
    
    return mse

if __name__ == "__main__":
    error = run_benchmark()
    print(f"FD-PRC Benchmark MSE: {error:.6f}")
