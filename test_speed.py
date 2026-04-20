import numpy as np
import torch

# Kiểm tra NumPy
np.show_config() 

# Kiểm tra PyTorch
print(torch.__config__.show())

print(torch.backends.cpu.get_cpu_capability())