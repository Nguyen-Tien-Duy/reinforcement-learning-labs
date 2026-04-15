import numpy as np

def simulate_memory():
    T = 7200
    Q_max_real = 1500000
    print(f"Memory for full Q_max: {T * Q_max_real * 4 / 1024 / 1024 / 1024:.2f} GB")
    
    # Using discretization
    Q_step = 1000
    Q_bins = int(Q_max_real / Q_step)
    print(f"Memory for discretized Q (Q_step={Q_step}, Q_bins={Q_bins}): {T * Q_bins * 4 / 1024 / 1024:.2f} MB")

simulate_memory()
