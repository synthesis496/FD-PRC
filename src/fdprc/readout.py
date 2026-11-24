import numpy as np


def ridge_fit(X, y, alpha=1e-6):
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def ridge_predict(X, w):
    return X @ w


def rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def r2(y, yh):
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0))


def recovery_time(err_series, baseline_err, tol=0.1):
    thr = baseline_err * (1 + tol)
    mask = np.append(np.asarray(err_series) <= thr, True)
    return int(np.argmax(mask))
