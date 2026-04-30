#!/usr/bin/env python3
import argparse, time, math
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def main():
    parser = argparse.ArgumentParser(description="P3D Turbulent Cube Surrogate Benchmark")
    parser.add_argument("--n", type=int, default=1, help="batch")
    parser.add_argument("--c", type=int, default=16, help="channels")
    parser.add_argument("--d", type=int, default=256, help="depth")
    parser.add_argument("--h", type=int, default=256, help="height")
    parser.add_argument("--w", type=int, default=256, help="width")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if torch is None:
        print("Torch not found; stub benchmark.")
        return

    x = torch.randn(args.n, args.c, args.d, args.h, args.w, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(args.c, args.c, 3,3,3, dtype=torch.float16, device=x.device)
    bias = torch.randn(args.c, dtype=torch.float32, device=x.device)

    def p3d_block(x):
        y = torch.nn.functional.conv3d(x, weight, bias=bias, padding=1, stride=1, dilation=1)
        yflat = y.flatten(2)
        attn = torch.softmax((yflat.transpose(1,2) @ yflat) / math.sqrt(max(1, y.shape[1])), dim=-1)
        y2 = (attn @ yflat.transpose(1,2)).transpose(1,2).reshape_as(y)
        return y2

    for _ in range(5):
        p3d_block(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        y = p3d_block(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters
    vox = args.n * args.d * args.h * args.w
    print(f"Avg time/iter: {dt*1000:.2f} ms | Voxels: {vox/1e9:.3f} Gvox | Throughput: {vox/dt/1e9:.2f} Gvox/s")

if __name__ == "__main__":
    main()
