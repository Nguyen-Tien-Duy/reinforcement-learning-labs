import numpy as np

def test_dp():
    T = 7200
    Q_max_real = 1500000
    Q_bins = 1000
    Q_step = max(1, Q_max_real // Q_bins)
    
    V = np.full((T, Q_bins + 1), np.inf, dtype=np.float32)
    print(f"Memory for V: {V.nbytes / 1024 / 1024:.2f} MB")
    
    b_star = 4
    action_ratios = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    exec_cap = 500
    
    # testing one forward pass
    q_curr_real = 60000
    n = int(min(np.round(action_ratios[b_star] * q_curr_real), exec_cap))
    print(f"At Q=60000, action=4 executed: {n}")
    
test_dp()
