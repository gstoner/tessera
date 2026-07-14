"""Record real sm_120 D2 matmul verdicts into the shared autotune corpus.

Run on a live NVIDIA host; the recorder never fabricates data off-device.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def _shape(text: str, rank: int) -> tuple[int, ...]:
    try:
        dims = tuple(int(part) for part in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}") from exc
    if len(dims) != rank or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(
            f"expected {rank} positive dimensions joined by x; got {text!r}")
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matmul-shapes", nargs="+", default=(
            "64x64x64", "256x256x256", "512x512x512",
            "1024x1024x1024", "2048x2048x2048", "128x256x64"),
        metavar="MxNxK")
    parser.add_argument(
        "--fused-shapes", nargs="+", default=("64x64x64", "256x256x256",
                                                "128x512x256"),
        metavar="MxNxK")
    parser.add_argument(
        "--attention-shapes", nargs="+",
        default=("128x128x64x64", "64x512x64x64"), metavar="MxNkxDxDv")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--device-warmup", type=int, default=10)
    parser.add_argument("--matmul-dtypes", nargs="+",
                        choices=("float16", "bfloat16"),
                        default=("float16", "bfloat16"))
    args = parser.parse_args()

    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda  # registers candidates
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit.candidate import OP_MATMUL
    from tessera.compiler.emit.candidate import OP_FUSED_REGION, OP_ATTENTION
    from tessera.compiler.fusion import MatmulRegion, FusedRegion, AttentionRegion

    if rt._nvidia_device_name() != "sm_120":
        print("sm_120 NVIDIA runtime unavailable; corpus unchanged")
        return 0
    cache = at.MeasureCache()
    at.load_corpus(cache=cache)
    rng = np.random.default_rng(20260714)
    for dtype in args.matmul_dtypes:
        for shape_text in args.matmul_shapes:
            m, n, k = _shape(shape_text, 3)
            a = (rng.standard_normal((m, k)) * .2).astype(np.float32)
            b = (rng.standard_normal((k, n)) * .2).astype(np.float32)
            region = MatmulRegion(dtype=dtype)
            winner = at.measured_arbitrate(
                region, OP_MATMUL, "nvidia", a, b, dims=(m, n, k),
                dtype=dtype, cache=cache, reps=args.reps, warmup=args.warmup)
            if winner is None:
                raise RuntimeError(
                    f"no verified NVIDIA {dtype} candidate for {m}x{n}x{k}")
            print(f"matmul {dtype} {m}x{n}x{k}: {winner.name}")
            device_winner = at.measured_arbitrate(
                region, OP_MATMUL, "nvidia", a, b, dims=(m, n, k),
                dtype=dtype, cache=cache, reps=args.device_reps,
                warmup=args.device_warmup, timing="device")
            if device_winner is None:
                raise RuntimeError(
                    f"no device-timed NVIDIA {dtype} candidate for {m}x{n}x{k}")
            print(f"matmul-device {dtype} {m}x{n}x{k}: {device_winner.name}")
    # Representative production regions: each row compares every F4-passing
    # NVIDIA candidate, so generic and tensor-core lanes are evidence-ranked.
    for shape_text in args.fused_shapes:
        m, n, k = _shape(shape_text, 3)
        a = (rng.standard_normal((m, k)) * .2).astype(np.float32)
        b = (rng.standard_normal((k, n)) * .2).astype(np.float32)
        bias = (rng.standard_normal((n,)) * .2).astype(np.float32)
        winner = at.measured_arbitrate(
            FusedRegion(epilogue=("bias", "gelu")), OP_FUSED_REGION, "nvidia",
            a, b, bias, dims=(m, n, k), dtype="f16", cache=cache,
            reps=args.reps, warmup=args.warmup,
        )
        if winner is None:
            raise RuntimeError(
                f"no verified NVIDIA fused candidate for {m}x{n}x{k}")
        print(f"fused {m}x{n}x{k}: {winner.name}")
    for shape_text in args.attention_shapes:
        m, nk, d, dv = _shape(shape_text, 4)
        q = (rng.standard_normal((m, d)) * .2).astype(np.float32)
        k = (rng.standard_normal((nk, d)) * .2).astype(np.float32)
        v = (rng.standard_normal((nk, dv)) * .2).astype(np.float32)
        winner = at.measured_arbitrate(
            AttentionRegion(scale=d ** -.5, causal=True), OP_ATTENTION, "nvidia",
            q, k, v, dims=(m, nk, d, dv), dtype="f16", cache=cache,
            reps=args.reps, warmup=args.warmup,
        )
        if winner is None:
            raise RuntimeError(
                f"no verified NVIDIA attention candidate for {m}x{nk}x{d}x{dv}")
        print(f"attention {m}x{nk}x{d}x{dv}: {winner.name}")
    print(f"wrote {at.save_corpus(cache=cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
