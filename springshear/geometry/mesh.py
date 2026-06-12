import numpy as np


def build_mesh(L: float, dx: float) -> np.ndarray:
    n = int(np.round(L / dx))
    x = np.arange(n + 1, dtype=float) * dx
    x[-1] = L
    return x.round(6)
