
#!/usr/bin/env python3
import argparse, json, time, numpy as np

def conv2d_nhwc(x, w, stride=1, padding=0):
    N,H,W,C = x.shape
    KH,KW,C,OC = w.shape
    H2 = (H + 2*padding - KH)//stride + 1
    W2 = (W + 2*padding - KW)//stride + 1
    y = np.zeros((N,H2,W2,OC), dtype=x.dtype)
    if padding>0:
        xp = np.zeros((N,H+2*padding,W+2*padding,C), dtype=x.dtype)
        xp[:,padding:padding+H,padding:padding+W,:] = x
        x = xp
    for n in range(N):
        for i in range(H2):
            for j in range(W2):
                rs = i*stride
                cs = j*stride
                patch = x[n, rs:rs+KH, cs:cs+KW, :] # KH×KW×C
                # (KH*KW*C) dot (KH*KW*C, OC)
                y[n,i,j,:] = np.tensordot(patch, w, axes=([0,1,2],[0,1,2]))
    return y

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--h", type=int, default=64)
ap.add_argument("--w", type=int, default=64)
ap.add_argument("--c", type=int, default=32)
ap.add_argument("--oc", type=int, default=64)
ap.add_argument("--kh", type=int, default=3)
ap.add_argument("--kw", type=int, default=3)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--pad", type=int, default=1)
ap.add_argument("--repeat", type=int, default=2)
args = ap.parse_args()

x = (np.random.rand(args.n,args.h,args.w,args.c).astype(np.float32)-0.5)*0.1
w = (np.random.rand(args.kh,args.kw,args.c,args.oc).astype(np.float32)-0.5)*0.1

best = 0.0
last_ms = 0.0
max_abs = 0.0

for r in range(args.repeat):
    t0 = time.time()
    y = conv2d_nhwc(x, w, stride=args.stride, padding=args.pad)
    t1 = time.time()
    last_ms = (t1-t0)*1000.0
    # FLOPs estimation: per output pixel per OC: KH*KW*C*2
    H2 = y.shape[1]; W2 = y.shape[2]; OC = y.shape[3]
    flops = args.n * H2 * W2 * OC * args.kh * args.kw * args.c * 2.0 / max((t1-t0), 1e-9)
    best = max(best, flops)
    # Correctness vs im2col-ref (float64)
    y_ref = conv2d_nhwc(x.astype(np.float64), w.astype(np.float64), stride=args.stride, padding=args.pad).astype(np.float32)
    max_abs = float(np.max(np.abs(y - y_ref)))

row = {
  "throughput_flops": best,
  "latency_ms": last_ms,
  "max_abs_err": max_abs
}
print(json.dumps(row))
