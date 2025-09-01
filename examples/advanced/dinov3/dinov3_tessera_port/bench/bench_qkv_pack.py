import os, time, torch
from models.dinov3_tessera.ops.flash_attention_tessera import fused_qkv_pack, fused_qkv_bias_gelu

def run(B=8, N=196, K=1024, D=64, steps=50, warmup=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.randn(B, N, K, device=device)
    Wq = torch.randn( D, K, device=device); bq = torch.randn(D, device=device)
    Wk = torch.randn( D, K, device=device); bk = torch.randn(D, device=device)
    Wv = torch.randn( D, K, device=device); bv = torch.randn(D, device=device)
    Wcat = torch.cat([Wq, Wk, Wv], dim=0).contiguous()
    bcat = torch.cat([bq, bk, bv], dim=0).contiguous()

    os.environ["TESSERA_USE_CUSTOM_KERNELS"] = "1"

    # warmup pack
    for _ in range(warmup): fused_qkv_pack(x, Wcat, bcat, True)
    torch.cuda.synchronize() if device=="cuda" else None
    t0 = time.time()
    for _ in range(steps): fused_qkv_pack(x, Wcat, bcat, True)
    torch.cuda.synchronize() if device=="cuda" else None
    t1 = time.time()

    # warmup 3x linear
    for _ in range(warmup): fused_qkv_bias_gelu(x, Wq, bq, Wk, bk, Wv, bv)
    torch.cuda.synchronize() if device=="cuda" else None
    t2 = time.time()
    for _ in range(steps): fused_qkv_bias_gelu(x, Wq, bq, Wk, bk, Wv, bv)
    torch.cuda.synchronize() if device=="cuda" else None
    t3 = time.time()

    print(f"QKV pack: {(t1-t0)/steps*1000:.2f} ms/iter")
    print(f"3x Linear: {(t3-t2)/steps*1000:.2f} ms/iter")

if __name__ == "__main__":
    run()
