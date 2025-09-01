
import time, json, argparse, torch
from pathlib import Path
from tessera_kernels.tiled import flashattn_fwd_tiled

def run(B=1,H=16,S=1024,D=128, iters=50, warmup=10, seed=1234):
    dtype = torch.float16 if torch.cuda.get_device_capability(0)[0]>=8 else torch.float32
    Q = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    K = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    V = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    for _ in range(warmup): flashattn_fwd_tiled(Q,K,V,None,None,is_causal=True, seed=seed)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        flashattn_fwd_tiled(Q,K,V,None,None,is_causal=True, seed=seed)
        torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    ts.sort()
    return {"p50_s": ts[len(ts)//2], "p95_s": ts[int(len(ts)*0.95)]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="runs/micro_flashattn_tiled.jsonl")
    args = ap.parse_args()
    res = run(args.B,args.H,args.S,args.D,args.iters,args.warmup,args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"a") as f: f.write(json.dumps(res)+"\n")
    print(res)
