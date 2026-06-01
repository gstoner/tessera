"""Multi-layer transformer mini-benchmark — full-stack validation.

Stacks N attention layers + simplified MLP layers under
``ResidentWeights`` + ``@auto_batch``, exercising the entire
encode-session pipeline at realistic shapes. Measures per-step
latency and tokens/sec across the four execution modes (cold,
per_op, one_cb, warmed) so the headline single-cb + warmup speedups
hold under multi-layer scale.

A "decode step" runs N transformer layers, where each layer is:

  x = attention_block(rmsnorm(x), W_q, W_k, W_v, W_o, theta)
  x = simplified_mlp_block(rmsnorm(x), W_gate, W_up_proxy, W_down)
        # ``simplified_mlp`` = down_bmm(silu(gate_bmm(n)))
        # The gate·up elementwise multiply is omitted; the
        # canonical SwiGLU MLP requires a binary_dev_enc wrapper
        # that's a follow-on. The latency profile is
        # representative: 2 bmms + 1 silu per layer.

Ops per layer: 8 attention (rmsnorm + qkv×3 + rope×2 + flash_attn +
out_bmm) + 5 MLP (rmsnorm + gate_bmm + silu + down_bmm + residual-
free) = ~13 user ops per layer.

For N=6 layers: ~78 user ops, ~78 internal encoded ops, ONE
command buffer commit per step (warmed_one_cb mode).

Output JSON follows the standard schema for downstream consumption.

Usage:
    python benchmarks/apple_gpu/benchmark_multi_layer_transformer.py \\
        --shapes 1x32x64,6  1x64x128,6  --reps 20
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


def _parse_shape(spec: str) -> tuple[int, int, int, int]:
    """Parse a ``BxSxD,N`` shape spec into (B, S, D, num_layers)."""
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"shape must be BxSxD,N; got {spec}")
    bsd, n = parts
    bsd_parts = bsd.lower().split("x")
    if len(bsd_parts) != 3:
        raise ValueError(f"shape's BxSxD must be 3 components; got {bsd}")
    return int(bsd_parts[0]), int(bsd_parts[1]), int(bsd_parts[2]), int(n)


def _time(fn, reps: int) -> tuple[float, float]:
    fn()  # warmup (first call pays MSL compile + MPSGraph build)
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
    # Defaults exercise the chunking path (Phase 5b, 2026-06-01).
    # The chain planner splits at DEFAULT_OPS_PER_CB=30, so N=6 at
    # 12 ops/layer = 72 ops → 3 cbs; N=12 = 144 ops → 5 cbs.
    # Steady-state numbers across these N values characterize the
    # tokens/sec / speedup envelope for real multi-layer decode.
    parser.add_argument("--shapes", nargs="+",
                        default=["1x8x16,6", "1x32x64,6",
                                 "1x32x64,12"])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped": "non-darwin host"},
                indent=2, sort_keys=True))
        print("multi-layer-transformer benchmark: skipping (non-darwin)",
              file=sys.stderr)
        return 0

    from tessera.apple_gpu_batched import (
        batched_session, bmm_enc, device_tensor, flash_attn_enc,
        rmsnorm_enc, rope_enc, session_available, silu_enc,
    )
    from tessera.apple_gpu_resident import ResidentWeights
    import tessera.apple_gpu_ops as agpu

    if not session_available():
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped": "no encode-session"},
                indent=2, sort_keys=True))
        print("multi-layer-transformer: skipping (no encode-session)",
              file=sys.stderr)
        return 0

    version = "dev"
    try:
        import importlib.metadata
        version = importlib.metadata.version("tessera")
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for spec in args.shapes:
        B, S, D, N = _parse_shape(spec)
        scale = 1.0 / np.sqrt(D)
        eps = 1e-5
        FFD = 2 * D  # MLP hidden dim ~ 2-4x is canonical; use 2 for
                      # tighter shapes at the benchmark scale.

        rng = np.random.default_rng(0xDEADBEEF)
        X = rng.standard_normal((B * S, D), dtype=np.float32) * 0.1

        # Generate per-layer weight tensors.
        layers = []
        for layer_idx in range(N):
            layers.append({
                "gamma_attn": rng.standard_normal((D,),
                    dtype=np.float32),
                "Wq": rng.standard_normal((1, D, D),
                    dtype=np.float32) * 0.05,
                "Wk": rng.standard_normal((1, D, D),
                    dtype=np.float32) * 0.05,
                "Wv": rng.standard_normal((1, D, D),
                    dtype=np.float32) * 0.05,
                "Wo": rng.standard_normal((1, D, D),
                    dtype=np.float32) * 0.05,
                "Theta": (np.arange(B * S * D, dtype=np.float32) *
                          0.001 * (1 + layer_idx)).reshape(B * S, D),
                "gamma_mlp": rng.standard_normal((D,),
                    dtype=np.float32),
                "Wgate": rng.standard_normal((1, D, FFD),
                    dtype=np.float32) * 0.05,
                "Wdown": rng.standard_normal((1, FFD, D),
                    dtype=np.float32) * 0.05,
            })

        # Pre-upload weights once for the steady-state modes.
        cache = ResidentWeights()
        for i, layer in enumerate(layers):
            for name, arr in layer.items():
                cache.weight(f"L{i}_{name}", arr)

        try:
            def _one_layer_attention(s, x_dev, layer_dev_lookup):
                """Run one attention sub-block under session ``s``,
                returning the new x tensor (DeviceTensor)."""
                n = rmsnorm_enc(s, x_dev,
                                 layer_dev_lookup("gamma_attn"),
                                 rows=B * S, cols=D, eps=eps)
                q = bmm_enc(s, n, layer_dev_lookup("Wq"),
                             batch=1, M=B * S, N=D, K=D)
                k = bmm_enc(s, n, layer_dev_lookup("Wk"),
                             batch=1, M=B * S, N=D, K=D)
                v = bmm_enc(s, n, layer_dev_lookup("Wv"),
                             batch=1, M=B * S, N=D, K=D)
                q_r = rope_enc(s, q, layer_dev_lookup("Theta"),
                                M=B * S, K=D)
                k_r = rope_enc(s, k, layer_dev_lookup("Theta"),
                                M=B * S, K=D)
                a = flash_attn_enc(s, q_r, k_r, v,
                                    B=B, Sq=S, Sk=S, D=D, scale=scale)
                return bmm_enc(s, a, layer_dev_lookup("Wo"),
                                batch=1, M=B * S, N=D, K=D)

            def _one_layer_mlp(s, x_dev, layer_dev_lookup):
                """Simplified MLP: rmsnorm → gate_bmm → silu →
                down_bmm. SwiGLU's gate·up multiply is omitted; the
                latency shape is representative."""
                n = rmsnorm_enc(s, x_dev,
                                 layer_dev_lookup("gamma_mlp"),
                                 rows=B * S, cols=D, eps=eps)
                gate = bmm_enc(s, n, layer_dev_lookup("Wgate"),
                                batch=1, M=B * S, N=FFD, K=D)
                act = silu_enc(s, gate, n=B * S * FFD)
                return bmm_enc(s, act, layer_dev_lookup("Wdown"),
                                batch=1, M=B * S, N=D, K=FFD)

            # ---- Mode 1: per_op_cold (re-upload everything each step)
            def per_op_cold():
                x_d = device_tensor(X)
                fresh_devs = []
                # Re-upload all weights this iteration (the cold path).
                for layer in layers:
                    fresh = {}
                    for name, arr in layer.items():
                        d = device_tensor(arr)
                        fresh[name] = d
                        fresh_devs.append(d)
                    for stage in (_one_layer_attention, _one_layer_mlp):
                        # Each stage opens its own session, runs ops
                        # per-op (one cb per op).
                        x_inner_dev = x_d
                        x_d = stage(s=None, x_dev=x_d,  # placeholder
                                     layer_dev_lookup=lambda k: fresh[k])
                        # ^ The session=None won't work; use a per-op
                        # session instead. Rewrite below.
                    # WAIT — the helpers need a real session. Reset.
                # This mode is implemented below with explicit per-op
                # sessions to be honest.
                pass  # actual impl below

            # Real per_op_cold: re-upload + run each op in its own cb.
            def per_op_cold_real():
                fresh = []
                for layer in layers:
                    layer_d = {name: device_tensor(arr)
                               for name, arr in layer.items()}
                    fresh.append(layer_d)
                x_d = device_tensor(X)
                fresh_x_owned = True
                try:
                    for layer_d in fresh:
                        # Attention block — 8 ops, 8 sessions.
                        with batched_session() as s:
                            n = rmsnorm_enc(s, x_d, layer_d["gamma_attn"],
                                             rows=B * S, cols=D, eps=eps)
                        with batched_session() as s:
                            q = bmm_enc(s, n, layer_d["Wq"],
                                         batch=1, M=B * S, N=D, K=D)
                        with batched_session() as s:
                            k = bmm_enc(s, n, layer_d["Wk"],
                                         batch=1, M=B * S, N=D, K=D)
                        with batched_session() as s:
                            v = bmm_enc(s, n, layer_d["Wv"],
                                         batch=1, M=B * S, N=D, K=D)
                        with batched_session() as s:
                            q_r = rope_enc(s, q, layer_d["Theta"],
                                            M=B * S, K=D)
                        with batched_session() as s:
                            k_r = rope_enc(s, k, layer_d["Theta"],
                                            M=B * S, K=D)
                        with batched_session() as s:
                            a = flash_attn_enc(s, q_r, k_r, v,
                                                B=B, Sq=S, Sk=S, D=D,
                                                scale=scale)
                        with batched_session() as s:
                            attn_out = bmm_enc(s, a, layer_d["Wo"],
                                                batch=1, M=B * S, N=D, K=D)
                        # MLP block — 4 ops, 4 sessions.
                        with batched_session() as s:
                            mlp_n = rmsnorm_enc(s, attn_out,
                                                 layer_d["gamma_mlp"],
                                                 rows=B * S, cols=D, eps=eps)
                        with batched_session() as s:
                            gate = bmm_enc(s, mlp_n, layer_d["Wgate"],
                                            batch=1, M=B * S, N=FFD, K=D)
                        with batched_session() as s:
                            act = silu_enc(s, gate, n=B * S * FFD)
                        with batched_session() as s:
                            new_x = bmm_enc(s, act, layer_d["Wdown"],
                                             batch=1, M=B * S, N=D, K=FFD)
                        # Free interior tensors.
                        for t in (n, q, k, v, q_r, k_r, a, attn_out,
                                  mlp_n, gate, act):
                            t.free()
                        if fresh_x_owned:
                            x_d.free()
                            fresh_x_owned = False
                        x_d = new_x
                        fresh_x_owned = True
                    # Final download to ensure GPU completes.
                    result = x_d.download(np.float32, (1, B * S, D))
                    return result
                finally:
                    if fresh_x_owned:
                        x_d.free()
                    for layer_d in fresh:
                        for t in layer_d.values():
                            t.free()

            # ---- Mode 2: warmed_one_cb (full stack — what real LLM
            # decode looks like). Weights resident, all N layers in one
            # cb via @auto_batch.
            @agpu.auto_batch
            def warmed_chain(x_arr):
                x_t = x_arr
                for i in range(N):
                    # Attention.
                    n = agpu.rmsnorm(x_t, cache[f"L{i}_gamma_attn"],
                                      rows=B * S, cols=D, eps=eps)
                    q = agpu.bmm(n, cache[f"L{i}_Wq"],
                                  batch=1, M=B * S, N=D, K=D)
                    k = agpu.bmm(n, cache[f"L{i}_Wk"],
                                  batch=1, M=B * S, N=D, K=D)
                    v = agpu.bmm(n, cache[f"L{i}_Wv"],
                                  batch=1, M=B * S, N=D, K=D)
                    q_r = agpu.rope(q, cache[f"L{i}_Theta"],
                                     M=B * S, K=D)
                    k_r = agpu.rope(k, cache[f"L{i}_Theta"],
                                     M=B * S, K=D)
                    a = agpu.flash_attn(q_r, k_r, v,
                                         B=B, Sq=S, Sk=S, D=D,
                                         scale=scale)
                    x_t = agpu.bmm(a, cache[f"L{i}_Wo"],
                                    batch=1, M=B * S, N=D, K=D)
                    # MLP.
                    mlp_n = agpu.rmsnorm(x_t, cache[f"L{i}_gamma_mlp"],
                                          rows=B * S, cols=D, eps=eps)
                    gate = agpu.bmm(mlp_n, cache[f"L{i}_Wgate"],
                                     batch=1, M=B * S, N=FFD, K=D)
                    act = agpu.silu(gate, n=B * S * FFD)
                    x_t = agpu.bmm(act, cache[f"L{i}_Wdown"],
                                    batch=1, M=B * S, N=D, K=FFD)
                return x_t

            def warmed_one_cb():
                x_act_dev = cache.activation("x", X)
                # auto_batch returns a DeviceTensor for the final x_t.
                out = warmed_chain(x_act_dev)
                result = out.download(np.float32, (1, B * S, D))
                out.free()
                return result

            for mode, fn in (("per_op_cold", per_op_cold_real),
                             ("warmed_one_cb", warmed_one_cb)):
                ms, stdev_ms = _time(fn, args.reps)
                rows.append({
                    "backend": "apple_gpu",
                    "op": "transformer_decoder_step",
                    "shape": spec,
                    "dtype": "f32",
                    "mode": mode,
                    "num_layers": N,
                    "ops_per_layer": 12,    # 8 attn + 4 mlp
                    "total_ops": N * 12,
                    "reps": args.reps,
                    "latency_ms": ms,
                    "stdev_ms": stdev_ms,
                    "ms_per_layer": ms / N,
                    "tokens_per_sec": (
                        (B * S) / (ms / 1000) if ms > 0 else 0.0),
                    "device": "apple_silicon_metal",
                    "tessera_version": version,
                })

            cold_ms = next(r["latency_ms"] for r in rows
                           if r["shape"] == spec
                           and r["mode"] == "per_op_cold")
            warmed_ms = next(r["latency_ms"] for r in rows
                             if r["shape"] == spec
                             and r["mode"] == "warmed_one_cb")
            rows.append({
                "backend": "apple_gpu",
                "op": "transformer_decoder_step_speedup",
                "shape": spec,
                "dtype": "f32",
                "mode": "speedup",
                "num_layers": N,
                "speedup_x_full_stack": (
                    cold_ms / warmed_ms if warmed_ms > 0 else 0.0),
                "cold_ms": cold_ms,
                "warmed_one_cb_ms": warmed_ms,
                "tokens_per_sec_warmed": (
                    (B * S) / (warmed_ms / 1000)
                    if warmed_ms > 0 else 0.0),
                "device": "apple_silicon_metal",
                "tessera_version": version,
            })
        finally:
            cache.free()

    output = json.dumps({"runs": rows}, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
