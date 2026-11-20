import numpy as np

def train_readout(states, targets, ridge_alpha=1e-5):
    X = np.array(states)
    Y = np.array(targets)
    X_T = X.T
    I = np.eye(X.shape[1])
    return np.linalg.inv(X_T @ X + ridge_alpha * I) @ X_T @ Y

def predict(W_out, state):
    return np.dot(W_out, state)
