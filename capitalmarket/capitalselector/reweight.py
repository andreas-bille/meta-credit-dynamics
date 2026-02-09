import numpy as np

def simplex_normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s == 0:
        return np.ones_like(w) / len(w)
    return w / s

def exp_reweight(w: np.ndarray, score: np.ndarray, eta: float) -> np.ndarray:
    """Exponentiated gradient update on simplex.

    score is a per-channel vector. If a scalar is provided, it is broadcast.
    """
    g = np.zeros_like(w)
    g[:] = score
    return simplex_normalize(w * np.exp(eta * g))
