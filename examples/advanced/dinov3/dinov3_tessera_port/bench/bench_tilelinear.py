import os, time, torch
from models.dinov3_tessera.ops.flash_attention_tessera import TileLinear, DummySchedule
import torch.nn as nn

def bench_once(B=32, N=196, K=1024, M=4096, iters=50, warmup=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(B, N, K, device=device)
    lin = nn.Linear(K, M, bias=True).to(device)
    tl = TileLinear(K, M, bias=True, activation=None, schedule=DummySchedule(block_m=128, block_k=64)).to(device)
    with torch.no_grad():
        tl.weight.copy_(lin.weight)
        tl.bias.copy_(lin.bias)

    # nn.Linear
    t0 = time.time()
    for _ in range(warmup): _ = lin(x)
    torch.cuda.synchronize() if device=="cuda" else None
    t1 = time.time()
    for _ in range(iters): _ = lin(x)
    torch.cuda.synchronize() if device=="cuda" else None
    t2 = time.time()

    # TileLinear (custom kernel if enabled)
    os.environ["TESSERA_USE_CUSTOM_KERNELS"] = "1"
    t3 = time.time()
    for _ in range(warmup): _ = tl(x)
    torch.cuda.synchronize() if device=="cuda" else None
    t4 = time.time()
    for _ in range(iters): _ = tl(x)
    torch.cuda.synchronize() if device=="cuda" else None
    t5 = time.time()

    print(f"nn.Linear:   {(t2 - t1)/iters*1000:.2f} ms/iter")
    print(f"TileLinear:  {(t5 - t4)/iters*1000:.2f} ms/iter")

if __name__ == "__main__":
    bench_once()
