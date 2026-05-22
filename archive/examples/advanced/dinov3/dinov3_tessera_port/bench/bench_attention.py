import os, time, torch
from models.dinov3_tessera.ops.flash_attention_tessera import tessera_flash_attn, DummySchedule

def run(mode, B=2, H=8, N=196, D=64, steps=50, warmup=10):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    # mode: "fused", "naive", "ref"
    os.environ["TESSERA_USE_CUSTOM_KERNELS"] = "1"
    os.environ["TESSERA_REFERENCE_KERNELS"] = "1" if mode == "ref" else "0"
    os.environ["TESSERA_NAIVE_ATTENTION"] = "1" if mode == "naive" else "0"

    schedule = DummySchedule(block_n=64, block_k=64)

    # warmup
    for _ in range(warmup):
        _ = tessera_flash_attn(q, k, v, schedule=schedule)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        _ = tessera_flash_attn(q, k, v, schedule=schedule)
    if device == "cuda": torch.cuda.synchronize()
    t1 = time.time()
    print(f"{mode}: {(t1 - t0)/steps*1000:.2f} ms/iter")

if __name__ == "__main__":
    for mode in ["ref", "naive", "fused"]:
        run(mode)
