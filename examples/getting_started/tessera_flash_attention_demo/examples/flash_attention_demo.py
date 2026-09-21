#!/usr/bin/env python3
"""Canonical Tessera Flash Attention example with a PyTorch oracle.

The Tessera call intentionally uses the CPU reference target so this example is
portable. Its timing is reported separately from PyTorch and is not a device
performance comparison. Use an architecture-owned benchmark for native GPU
performance claims.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import tessera as ts
from utils.attention_ref import sdpa_reference


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@ts.jit
def flash_attn(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=False)


@ts.jit
def causal_flash_attn(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


def _timed_ms(fn, *, iterations: int, synchronize=None):
    total = 0.0
    result = None
    for _ in range(iterations):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        result = fn()
        if synchronize is not None:
            synchronize()
        total += (time.perf_counter() - start) * 1000.0
    return result, total / iterations


def _write_ir_artifacts(compiled_fn) -> dict[str, object]:
    artifacts = {
        "graph": compiled_fn.ir_text(),
        "schedule": compiled_fn.schedule_ir,
        "tile": compiled_fn.tile_ir,
        "target": compiled_fn.target_ir,
    }
    missing = [stage for stage, text in artifacts.items() if not text]
    if missing:
        raise RuntimeError(
            "flash-attention example expected non-empty artifacts for: "
            + ", ".join(missing)
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for stage, text in artifacts.items():
        (ARTIFACTS / f"{stage}_ir.mlir").write_text(text, encoding="utf-8")

    summary = {
        "execution_kind": compiled_fn.execution_kind,
        "is_reference_execution": compiled_fn.is_reference_execution,
        "is_native_execution": compiled_fn.is_native_execution,
        "stages": {stage: len(text) for stage, text in artifacts.items()},
    }
    (ARTIFACTS / "compilation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("f32",), default="f32")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dump-ir", action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device=cuda requested, but PyTorch cannot see CUDA")

    shape = (args.batch, args.heads, args.seq, args.dim)
    rng = np.random.default_rng(7)
    q = rng.standard_normal(shape, dtype=np.float32)
    k = rng.standard_normal(shape, dtype=np.float32)
    v = rng.standard_normal(shape, dtype=np.float32)

    q_ref = torch.from_numpy(q).to(device)
    k_ref = torch.from_numpy(k).to(device)
    v_ref = torch.from_numpy(v).to(device)
    synchronize = torch.cuda.synchronize if device == "cuda" else None
    iterations = max(1, args.iters)

    reference, reference_ms = _timed_ms(
        lambda: sdpa_reference(q_ref, k_ref, v_ref, causal=args.causal),
        iterations=iterations,
        synchronize=synchronize,
    )

    compiled_fn = causal_flash_attn if args.causal else flash_attn
    actual, tessera_ms = _timed_ms(
        lambda: compiled_fn(q, k, v),
        iterations=iterations,
    )
    expected = reference.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    print("Tessera Flash Attention")
    print(f"shape: {shape}; causal: {args.causal}")
    print(f"tessera execution: {compiled_fn.execution_kind}")
    print(f"PyTorch {device} wall time: {reference_ms:.3f} ms")
    print(f"Tessera reference_cpu wall time: {tessera_ms:.3f} ms")
    print("oracle: PASS")
    print(compiled_fn.explain())

    if args.dump_ir:
        summary = _write_ir_artifacts(compiled_fn)
        print(json.dumps(summary, indent=2))
        print(f"artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
