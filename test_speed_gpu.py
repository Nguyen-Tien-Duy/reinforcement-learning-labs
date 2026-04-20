import torch
import time

# Tạo 2 ma trận lớn 10000x10000
size = 10000
a = torch.randn(size, size)
b = torch.randn(size, size)

start = time.time()
# Nhân ma trận
c = torch.mm(a, b)
end = time.time()

print(f"Thời gian tính toán: {end - start:.4f} giây")