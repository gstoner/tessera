"""Single-command-buffer decoder-layer benchmark — audit Action 6 closeout.

Measures the wall-clock speedup of running a Llama-style decoder
attention block (rmsnorm + Q/K/V projections + rope + flash_attn +
output projection = 8 encoded ops) on ONE command buffer versus N
command buffers (one per op).

Same JSON schema as ``benchmarks/benchmark_gemm.py`` so the roofline
tooling ingests the result unchanged. Run shape spec: ``BxSxD`` where
B is batch, S is sequence length, D is the head dimension.

Per-op baseline opens a FRESH encode session for each op (so the
exact same kernels are exercised — only the session-batching
behavior differs). The single-cb variant runs the whole chain in
one session.

Usage:
    python benchmarks/apple_gpu/benchmark_decoder_layer_one_cb.py \\
        --shapes 1x8x16 1x32x64 1x64x128 --reps 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be BxSxD, got {spec}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _time(fn, reps: int) -> tuple[float, float]:
    # Warm up + ignored first call.
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if reps > 1 else 0.0
    return median, stdev


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+",
                        default=["1x8x16", "1x32x64", "1x64x128"])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [],
                 "skipped_apple_gpu": "non-darwin host"},
                indent=2, sort_keys=True))
        print("decoder-layer one-cb benchmark: skipping (non-darwin)",
              file=sys.stderr)
        return 0

    from tessera.apple_gpu_batched import (
        batched_session, bmm_enc, device_tensor, flash_attn_enc,
        rmsnorm_enc, rope_enc, session_available,
    )

    if not session_available():
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [],
                 "skipped_apple_gpu": "encode session unavailable"},
                indent=2, sort_keys=True))
        print("decoder-layer one-cb benchmark: skipping "
              "(no encode session)", file=sys.stderr)
        return 0

    version = "dev"
    try:
        import importlib.metadata
        version = importlib.metadata.version("tessera")
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for spec in args.shapes:
        B, S, D = _parse_shape(spec)
        scale = 1.0 / np.sqrt(D)
        eps = 1e-5
        rng = np.random.default_rng(0xDEC0DEBE)
        X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1
        gamma = rng.standard_normal((D,), dtype=np.float32)
        Wq = rng.standard_normal((D, D), dtype=np.float32) * 0.05
        Wk = rng.standard_normal((D, D), dtype=np.float32) * 0.05
        Wv = rng.standard_normal((D, D), dtype=np.float32) * 0.05
        Wo = rng.standard_normal((D, D), dtype=np.float32) * 0.05
        Theta = (np.arange(B * S * D, dtype=np.float32) * 0.001
                 ).reshape(B * S, D)

        x_dev = device_tensor(X)
        g_dev = device_tensor(gamma)
        wq_dev = device_tensor(Wq.reshape(1, D, D))
        wk_dev = device_tensor(Wk.reshape(1, D, D))
        wv_dev = device_tensor(Wv.reshape(1, D, D))
        wo_dev = device_tensor(Wo.reshape(1, D, D))
        theta_dev = device_tensor(Theta)

        try:
            def per_op_cold():
                # The naive newcomer pattern: upload EVERY tensor
                # (weights AND activation) on each iteration. Then
                # run the chain with one cb per op. Real-world LLM
                # decode that doesn't know about resident weights
                # OR single-cb would look like this. Free everything
                # at end of iter.
                _x = device_tensor(X)
                _g = device_tensor(gamma)
                _wq = device_tensor(Wq.reshape(1, D, D))
                _wk = device_tensor(Wk.reshape(1, D, D))
                _wv = device_tensor(Wv.reshape(1, D, D))
                _wo = device_tensor(Wo.reshape(1, D, D))
                _theta = device_tensor(Theta)
                try:
                    with batched_session() as s:
                        n = rmsnorm_enc(s, _x, _g,
                                         rows=B * S, cols=D, eps=eps)
                    with batched_session() as s:
                        q = bmm_enc(s, n, _wq,
                                     batch=1, M=B * S, N=D, K=D)
                    with batched_session() as s:
                        k = bmm_enc(s, n, _wk,
                                     batch=1, M=B * S, N=D, K=D)
                    with batched_session() as s:
                        v = bmm_enc(s, n, _wv,
                                     batch=1, M=B * S, N=D, K=D)
                    with batched_session() as s:
                        q_r = rope_enc(s, q, _theta, M=B * S, K=D)
                    with batched_session() as s:
                        k_r = rope_enc(s, k, _theta, M=B * S, K=D)
                    with batched_session() as s:
                        a = flash_attn_enc(s, q_r, k_r, v,
                                            B=B, Sq=S, Sk=S, D=D,
                                            scale=scale)
                    with batched_session() as s:
                        final = bmm_enc(s, a, _wo,
                                         batch=1, M=B * S, N=D, K=D)
                    result = final.download(np.float32, (1, B * S, D))
                    for t in (n, q, k, v, q_r, k_r, a, final):
                        t.free()
                    return result
                finally:
                    for d in (_x, _g, _wq, _wk, _wv, _wo, _theta):
                        d.free()

            def per_op():
                # 8 separate sessions (one cb each). Same ops, same
                # device-resident weights — the only difference is
                # session batching.
                with batched_session() as s:
                    n = rmsnorm_enc(s, x_dev, g_dev,
                                     rows=B * S, cols=D, eps=eps)
                with batched_session() as s:
                    q = bmm_enc(s, n, wq_dev,
                                 batch=1, M=B * S, N=D, K=D)
                with batched_session() as s:
                    k = bmm_enc(s, n, wk_dev,
                                 batch=1, M=B * S, N=D, K=D)
                with batched_session() as s:
                    v = bmm_enc(s, n, wv_dev,
                                 batch=1, M=B * S, N=D, K=D)
                with batched_session() as s:
                    q_r = rope_enc(s, q, theta_dev, M=B * S, K=D)
                with batched_session() as s:
                    k_r = rope_enc(s, k, theta_dev, M=B * S, K=D)
                with batched_session() as s:
                    a = flash_attn_enc(s, q_r, k_r, v,
                                        B=B, Sq=S, Sk=S, D=D,
                                        scale=scale)
                with batched_session() as s:
                    final = bmm_enc(s, a, wo_dev,
                                     batch=1, M=B * S, N=D, K=D)
                # Force a download to ensure GPU completion before
                # timing stops.
                result = final.download(np.float32, (1, B * S, D))
                for t in (n, q, k, v, q_r, k_r, a, final):
                    t.free()
                return result

            def batched():
                with batched_session() as s:
                    n = rmsnorm_enc(s, x_dev, g_dev,
                                     rows=B * S, cols=D, eps=eps)
                    q = bmm_enc(s, n, wq_dev,
                                 batch=1, M=B * S, N=D, K=D)
                    k = bmm_enc(s, n, wk_dev,
                                 batch=1, M=B * S, N=D, K=D)
                    v = bmm_enc(s, n, wv_dev,
                                 batch=1, M=B * S, N=D, K=D)
                    q_r = rope_enc(s, q, theta_dev, M=B * S, K=D)
                    k_r = rope_enc(s, k, theta_dev, M=B * S, K=D)
                    a = flash_attn_enc(s, q_r, k_r, v,
                                        B=B, Sq=S, Sk=S, D=D,
                                        scale=scale)
                    out = bmm_enc(s, a, wo_dev,
                                   batch=1, M=B * S, N=D, K=D)
                result = out.download(np.float32, (1, B * S, D))
                for t in (n, q, k, v, q_r, k_r, a, out):
                    t.free()
                return result

            # Steady-state mode (Project-3, 2026-06-01) — uses
            # ResidentWeights to upload weights ONCE before the timed
            # loop AND @auto_batch for natural-looking code. This is
            # what a real LLM decode loop looks like in practice.
            from tessera.apple_gpu_resident import ResidentWeights
            import tessera.apple_gpu_ops as agpu

            warm_cache = ResidentWeights()
            warm_cache.weight("gamma", gamma)
            warm_cache.weight("Wq", Wq.reshape(1, D, D))
            warm_cache.weight("Wk", Wk.reshape(1, D, D))
            warm_cache.weight("Wv", Wv.reshape(1, D, D))
            warm_cache.weight("Wo", Wo.reshape(1, D, D))
            warm_cache.weight("Theta", Theta)

            @agpu.auto_batch
            def warmed_attention(x_act):
                """Same chain as `batched`, but using the tessera
                apple_gpu_ops surface (auto_batch decorator) AND with
                weights already device-resident via ResidentWeights.
                Per-step cost is: 1 activation upload + 8 ops in
                1 cb + 1 download."""
                n = agpu.rmsnorm(x_act, warm_cache["gamma"],
                                  rows=B * S, cols=D, eps=eps)
                q = agpu.bmm(n, warm_cache["Wq"],
                              batch=1, M=B * S, N=D, K=D)
                k = agpu.bmm(n, warm_cache["Wk"],
                              batch=1, M=B * S, N=D, K=D)
                v = agpu.bmm(n, warm_cache["Wv"],
                              batch=1, M=B * S, N=D, K=D)
                q_r = agpu.rope(q, warm_cache["Theta"],
                                 M=B * S, K=D)
                k_r = agpu.rope(k, warm_cache["Theta"],
                                 M=B * S, K=D)
                a = agpu.flash_attn(q_r, k_r, v,
                                     B=B, Sq=S, Sk=S, D=D,
                                     scale=scale)
                return agpu.bmm(a, warm_cache["Wo"],
                                 batch=1, M=B * S, N=D, K=D)

            def warmed_one_cb():
                # Re-upload only the activation each iteration.
                x_dev_warm = warm_cache.activation("x", X)
                out = warmed_attention(x_dev_warm)
                # Force download.
                result = out.download(np.float32, (1, B * S, D))
                out.free()
                return result

            for mode, fn in (("per_op_cold", per_op_cold),
                             ("per_op_baseline", per_op),
                             ("one_cb", batched),
                             ("warmed_one_cb", warmed_one_cb)):
                ms, stdev_ms = _time(fn, args.reps)
                rows.append({
                    "backend": "apple_gpu",
                    "op": "llama_attention_block",
                    "shape": spec,
                    "dtype": "f32",
                    "mode": mode,
                    "chain_len": 8,
                    "syncs": (8 if mode in ("per_op_baseline",
                                             "per_op_cold") else 1),
                    # warmed_one_cb uploads weights ONCE before the
                    # timed loop; per-iter cost is just the activation
                    # upload + chain + result download. per_op_cold
                    # re-uploads weights every iteration (the naive
                    # newcomer pattern).
                    "warmed": mode == "warmed_one_cb",
                    "cold_uploads": mode == "per_op_cold",
                    "reps": args.reps,
                    "latency_ms": ms,
                    "stdev_ms": stdev_ms,
                    "tflops": 0.0,
                    "memory_bw_gb_s": 0.0,
                    "device": "apple_silicon_metal",
                    "tessera_version": version,
                })

            # Compute speedup rows for the dashboard.
            def _mode_ms(name):
                return next(r["latency_ms"] for r in rows
                            if r["shape"] == spec and r["mode"] == name)

            cold_ms = _mode_ms("per_op_cold")
            per_op_ms = _mode_ms("per_op_baseline")
            one_cb_ms = _mode_ms("one_cb")
            warmed_ms = _mode_ms("warmed_one_cb")

            # Speedup story (each step strictly larger numerator):
            #   cold → per_op_baseline = ResidentWeights win
            #         (eliminates per-iter weight upload)
            #   per_op_baseline → one_cb = single-cb win
            #         (eliminates per-op cb commit overhead)
            #   per_op_baseline → warmed_one_cb = combined win
            #         (and equals one_cb when weights are device-
            #          resident — pure architectural framing)
            #   cold → warmed_one_cb = total stack win
            #         (the headline number for real LLM inference)
            rows.append({
                "backend": "apple_gpu",
                "op": "llama_attention_block_speedup",
                "shape": spec,
                "dtype": "f32",
                "mode": "speedup",
                "speedup_x_resident_weights": (
                    cold_ms / per_op_ms if per_op_ms > 0 else 0.0),
                "speedup_x_one_cb_vs_per_op": (
                    per_op_ms / one_cb_ms if one_cb_ms > 0 else 0.0),
                "speedup_x_one_cb_vs_cold": (
                    cold_ms / one_cb_ms if one_cb_ms > 0 else 0.0),
                "speedup_x_warmed_vs_cold": (
                    cold_ms / warmed_ms if warmed_ms > 0 else 0.0),
                "speedup_x_warmed_vs_per_op": (
                    per_op_ms / warmed_ms if warmed_ms > 0 else 0.0),
                "cold_ms": cold_ms,
                "per_op_ms": per_op_ms,
                "one_cb_ms": one_cb_ms,
                "warmed_one_cb_ms": warmed_ms,
                "device": "apple_silicon_metal",
                "tessera_version": version,
            })
            warm_cache.free()
        finally:
            for d in (x_dev, g_dev, wq_dev, wk_dev, wv_dev, wo_dev,
                      theta_dev):
                d.free()

    output = json.dumps({"runs": rows}, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
