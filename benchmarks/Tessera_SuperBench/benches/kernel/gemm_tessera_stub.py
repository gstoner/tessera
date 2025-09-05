
#!/usr/bin/env python3
import argparse, json, time, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=512)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--k", type=int, default=512)
ap.add_argument("--dtype", type=str, default="f32", choices=["f32","f16","bf16"])
ap.add_argument("--repeat", type=int, default=3)
args = ap.parse_args()

dtype = np.float32 if args.dtype=="f32" else np.float16
A = (np.arange(args.m*args.k, dtype=np.float32)%13/13).reshape(args.m,args.k).astype(dtype)
B = (np.arange(args.k*args.n, dtype=np.float32)%17/17).reshape(args.k,args.n).astype(dtype)

best = 0.0
last_ms = 0.0
max_abs = 0.0

for r in range(args.repeat):
    t0 = time.time()
    # Placeholder for Tessera-targeted path would go here
    C = A @ B
    t1 = time.time()
    ms = (t1-t0)*1000.0
    last_ms = ms
    flops = 2.0*args.m*args.n*args.k / max((t1-t0), 1e-9)
    best = max(best, flops)

    # Correctness vs float64 baseline
    C_ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(dtype)
    max_abs = float(np.max(np.abs(C - C_ref)))

row = {
  "throughput_flops": best,
  "latency_ms": last_ms,
  "max_abs_err": max_abs
}
print(json.dumps(row))
