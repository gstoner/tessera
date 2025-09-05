
#!/usr/bin/env python3
import argparse, json, time, numpy as np

def flash_attn(Q,K,V, scale=None, eps=1e-9):
    # Naive attention; not memory-efficient but fine for correctness
    # Q: [B,H,T,D], K: [B,H,T,D], V: [B,H,T,D]
    B,H,T,D = Q.shape
    scale = scale or (1.0/np.sqrt(D))
    out = np.zeros_like(Q)
    for b in range(B):
        for h in range(H):
            scores = (Q[b,h] @ K[b,h].transpose(1,0)) * scale  # [T,T]
            # subtract max for stability
            scores = scores - np.max(scores, axis=1, keepdims=True)
            P = np.exp(scores)
            P = P / (np.sum(P, axis=1, keepdims=True) + eps)
            out[b,h] = P @ V[b,h]
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--repeat", type=int, default=1)
args = ap.parse_args()

B,H,T,D = args.batch, args.heads, args.seq, args.d
Q = np.random.randn(B,H,T,D).astype(np.float32)*0.01
K = np.random.randn(B,H,T,D).astype(np.float32)*0.01
V = np.random.randn(B,H,T,D).astype(np.float32)*0.01

best_tps = 0.0
last_ms = 0.0
max_abs = 0.0

for r in range(args.repeat):
    t0 = time.time()
    O = flash_attn(Q,K,V)
    t1 = time.time()
    last_ms = (t1-t0)*1000.0
    toks = B*H*T
    tps = toks / max((t1-t0), 1e-9)
    best_tps = max(best_tps, tps)
    # correctness vs float64 naive
    O_ref = flash_attn(Q.astype(np.float64),K.astype(np.float64),V.astype(np.float64)).astype(np.float32)
    max_abs = float(np.max(np.abs(O - O_ref)))

row = {
  "throughput_tokens_per_s": best_tps,
  "latency_ms": last_ms,
  "max_abs_err": max_abs
}
print(json.dumps(row))
