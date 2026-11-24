import numpy as np


class RidgeReadout:
    def __init__(self, alpha=1e-6):
        self.alpha = alpha
        self.w = None

    def fit(self, X, y):
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, X):
        return X @ self.w


class HebbianReadout:
    def __init__(self, eta=1e-3):
        self.eta = eta
        self.w = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for x, target in zip(X, y):
            pred = x @ self.w
            self.w += self.eta * x * (target - pred)
        return self

    def predict(self, X):
        return X @ self.w


class ZeroShotReadout:
    def __init__(self, seed=42):
        self.seed = seed
        self.w = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.w = rng.standard_normal(X.shape[1])
        scale = np.std(y) / (np.std(X @ self.w) + 1e-12)
        self.w *= scale
        return self

    def predict(self, X):
        return X @ self.w
