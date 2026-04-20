import numpy as np
import time

size = 5000
a = np.random.rand(size, size).astype(np.float32)
b = np.random.rand(size, size).astype(np.float32)

start = time.time()
c = np.dot(a, b)
end = time.time()

print(f"NumPy thời gian: {end - start:.4f} giây")