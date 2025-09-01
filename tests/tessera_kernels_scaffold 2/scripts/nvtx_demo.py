import time, torch
from tessera_kernels import nvtx_range

with nvtx_range("demo.sleep"):
    time.sleep(0.01)

if torch.cuda.is_available():
    x = torch.randn(1, device="cuda")
    with nvtx_range("demo.cuda"):
        y = x * 2
        torch.cuda.synchronize()
