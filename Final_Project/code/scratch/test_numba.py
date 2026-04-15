from numba import njit
import numpy as np

@njit
def test():
    Q_real = float(5 * 1500)
    n = int(min(np.round(1.0 * Q_real), 500))
    remaining = max(0.0, Q_real - n)
    q_next_idx = int(np.round(remaining / 1500))
    return q_next_idx

print(test())
