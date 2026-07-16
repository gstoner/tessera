"""Record real sm_120 D2 matmul verdicts into the shared autotune corpus.

Run on a live NVIDIA host; the recorder never fabricates data off-device.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import subprocess
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
            "1024x1024x1024", "2048x2048x2048", "128x256x64",
            "127x259x63"),
        metavar="MxNxK")
    parser.add_argument(
        "--fused-shapes", nargs="+", default=("64x64x64", "256x256x256",
        "128x512x256", "127x259x63"),
        metavar="MxNxK")
    parser.add_argument(
        "--attention-shapes", nargs="+",
        default=("128x128x64x64", "64x512x64x64"), metavar="MxNkxDxDv")
    parser.add_argument("--gated-shapes", nargs="+",
                        default=("64x256x256", "128x512x512"),
                        metavar="MxHxK")
    parser.add_argument("--conv-shapes", nargs="+",
                        default=("1x32x32x32x3x3x64", "1x64x64x64x3x3x64"),
                        metavar="BxIHxIWxCIxKHxKWxCO")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--device-warmup", type=int, default=10)
    parser.add_argument("--matmul-dtypes", nargs="+",
                        choices=("float16", "bfloat16"),
                        default=("float16", "bfloat16"))
    parser.add_argument("--composed-dtypes", nargs="+",
                        choices=("f32", "fp8_e4m3", "fp8_e5m2"),
                        default=("f32", "fp8_e4m3", "fp8_e5m2"))
    parser.add_argument("--warm-start", action="store_true",
                        help="load existing corpus instead of measuring from a fresh cache")
    args = parser.parse_args()

    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda  # registers candidates
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key
    from tessera.compiler.emit.candidate import OP_MATMUL
    from tessera.compiler.emit.candidate import (
        OP_ATTENTION, OP_FUSED_REGION, OP_GATED_MATMUL)
    from tessera.compiler.fusion import (
        AttentionRegion, FusedRegion, GatedMatmulRegion, MatmulRegion)

    if rt._nvidia_device_name() != "sm_120":
        print("sm_120 NVIDIA runtime unavailable; corpus unchanged")
        return 0
    cache = at.MeasureCache()
    observed_shapes: dict[tuple[object, ...], list[int]] = {}
    if args.warm_start:
        at.load_corpus(cache=cache)
    nvcc_version = subprocess.run(
        ["/usr/local/cuda/bin/nvcc", "--version"], check=True,
        capture_output=True, text=True).stdout
    compiler_fingerprint = "sha256:" + hashlib.sha256(
        nvcc_version.encode()).hexdigest()
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
            bucket = bucket_key((m, n, k), SpecPolicy.BUCKET)
            for timing in (at.TIMING_END_TO_END, at.TIMING_DEVICE):
                observed_shapes[("nvidia:sm_120", "nvidia", OP_MATMUL,
                                 bucket, dtype, timing)] = [m, n, k]
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
    # Device-only transformer composition rows. These keep H2D/D2H out of the
    # TF32/FP8 crossover evidence and cover fused, attention, and gated paths.
    for storage in args.composed_dtypes:
        for shape_text in args.fused_shapes:
            m, n, k = _shape(shape_text, 3)
            a = (rng.standard_normal((m, k)) * .1).astype(np.float32)
            b = (rng.standard_normal((k, n)) * .1).astype(np.float32)
            bias = (rng.standard_normal(n) * .05).astype(np.float32)
            region = FusedRegion(
                epilogue=("bias", "gelu"), storage_dtype=storage)
            winner = at.measured_arbitrate(
                region, OP_FUSED_REGION, "nvidia", a, b, bias,
                dims=(m, n, k), dtype=storage, cache=cache,
                reps=args.device_reps, warmup=args.device_warmup,
                timing="device")
            if winner is None:
                raise RuntimeError(f"no {storage} fused device candidate")
            print(f"fused-device {storage} {shape_text}: {winner.name}")
        for shape_text in args.attention_shapes:
            m, nk, d, dv = _shape(shape_text, 4)
            q, kk, v = ((rng.standard_normal(s) * .1).astype(np.float32)
                        for s in ((m, d), (nk, d), (nk, dv)))
            region = AttentionRegion(
                scale=d ** -.5, causal=True, storage_dtype=storage)
            winner = at.measured_arbitrate(
                region, OP_ATTENTION, "nvidia", q, kk, v,
                dims=(m, nk, d, dv), dtype=storage, cache=cache,
                reps=args.device_reps, warmup=args.device_warmup,
                timing="device")
            if winner is None:
                raise RuntimeError(f"no {storage} attention device candidate")
            print(f"attention-device {storage} {shape_text}: {winner.name}")
        for shape_text in args.gated_shapes:
            m, h, k = _shape(shape_text, 3)
            a, wg, wu = ((rng.standard_normal(s) * .1).astype(np.float32)
                         for s in ((m, k), (k, h), (k, h)))
            region = GatedMatmulRegion(
                gate_act="silu", storage_dtype=storage)
            winner = at.measured_arbitrate(
                region, OP_GATED_MATMUL, "nvidia", a, wg, wu,
                dims=(m, h, k), dtype=storage, cache=cache,
                reps=args.device_reps, warmup=args.device_warmup,
                timing="device")
            if winner is None:
                raise RuntimeError(f"no {storage} gated device candidate")
            print(f"gated-device {storage} {shape_text}: {winner.name}")
    # Convolution is a route family rather than a Candidate op today. Persist
    # the same device-event evidence in D2 so promotion can consume it later.
    for shape_text in args.conv_shapes:
        B, IH, IW, CI, KH, KW, CO = _shape(shape_text, 7)
        x = (rng.standard_normal((B, IH, IW, CI)) * .1).astype(np.float32)
        w = (rng.standard_normal((KH, KW, CI, CO)) * .1).astype(np.float32)
        timings = {}
        for route in ("direct", "shared", "im2col_tf32"):
            _, timings[route] = nvidia_cuda.run_conv2d_resident_candidate(
                x, w, route=route, padding=(KH // 2, KW // 2),
                reps=args.device_reps, warmup=args.device_warmup)
        winner = min(timings, key=timings.__getitem__)
        cache.put(("nvidia:sm_120", "nvidia", "conv2d",
                   (B, IH, IW, CI, KH, KW, CO), "f32", "device"),
                  at.MeasureRecord(winner, timings[winner], timings))
        print(f"conv2d-device {shape_text}: {winner}")
    evidence = {
        "compiler_fingerprint": compiler_fingerprint,
        "compile_state": "warm_after_correctness_gate",
        "warmup": args.warmup,
        "reps": args.reps,
        "device_warmup": args.device_warmup,
        "device_reps": args.device_reps,
        "measure_cache": "warm_started" if args.warm_start else "fresh",
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    }
    for key, record in list(cache._store.items()):
        workload_shape = observed_shapes.get(key)
        cache._store[key] = dataclasses.replace(
            record, evidence={**record.evidence, **evidence,
                              **({"workload_shape": workload_shape}
                                 if workload_shape else {})})
    print(f"wrote {at.save_corpus(cache=cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
