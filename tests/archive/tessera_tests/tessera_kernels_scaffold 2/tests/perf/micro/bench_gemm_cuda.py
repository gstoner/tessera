import time, json, argparse, torch
from pathlib import Path
from tessera_kernels import gemm_fp16, HAS_EXT

def run_once(M,N,K, iters=200, warmup=50):
    assert torch.cuda.is_available() and HAS_EXT
    A = torch.randn(M,K, device="cuda", dtype=torch.float16)
    B = torch.randn(K,N, device="cuda", dtype=torch.float16)
    for _ in range(warmup): gemm_fp16(A,B)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        gemm_fp16(A,B)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    p50 = sorted(ts)[len(ts)//2]
    p95 = sorted(ts)[int(len(ts)*0.95)]
    flops = 2*M*N*K
    return {"p50_s":p50,"p95_s":p95,"tflops_p50": flops/p50/1e12, "tflops_p95": flops/p95/1e12}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", type=str, default="runs/micro_gemm_cuda.jsonl")
    args = ap.parse_args()
    res = run_once(args.M,args.N,args.K,args.iters,args.warmup)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"a") as f:
        f.write(json.dumps(res)+"\n")
    print(res)
