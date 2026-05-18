"""Canonical program: ``kv_cache append → prune → read``.

The KV-cache decode state block.  Exercises the
:class:`tessera.cache.KVCacheHandle` API across all three mutation
surfaces (append, prune, read) and emits a :class:`CompileReport`
that records IR identity, target decision, and (where applicable)
proof of native execution.

The KV cache today has no fused MSL kernel per se — the handle's
methods do their own bookkeeping in Python and call into the
runtime only for the FA-4 read path that downstream attention
layers exercise.  This canonical program documents that honestly:
``target_decision`` names the handle backend, and
``fallback_reason`` is ``None`` on Darwin (the handle is a real
native data structure) and ``NON_DARWIN_HOST`` elsewhere.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import tessera
from tessera.cache import KVCacheHandle
from tessera.compiler import jit_bridge as bridge
from tessera.compiler.compile_report import (
    CompileReport,
    FRONTEND_TESSERA_JIT,
    VALUE_KIND_TENSOR,
    hash_ir_text,
)
from tessera.compiler.fallback import (
    FallbackReason,
    TesseraNativeRequiredError,
    classify_host,
)


PROGRAM_ID = "kv_cache_append_prune_read"


def _make_inputs(
    *, num_heads: int = 4, head_dim: int = 16, max_seq: int = 32,
    new_tokens: int = 20, prune_to: int = 12, seed: int = 0,
):
    """Deterministic ``(K_in, V_in)`` tuple — `new_tokens` exceeds
    `prune_to`, so the program actually exercises the prune path."""
    rng = np.random.RandomState(seed)
    K_in = rng.randn(new_tokens, num_heads, head_dim).astype(np.float32)
    V_in = rng.randn(new_tokens, num_heads, head_dim).astype(np.float32)
    return K_in, V_in, num_heads, head_dim, max_seq, new_tokens, prune_to


def _numpy_reference(
    K_in: np.ndarray, V_in: np.ndarray, prune_to: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The append→prune→read pipeline in pure numpy."""
    # After append all N tokens are present; prune to the last
    # `prune_to` tokens (keeps the most recent, drops the oldest).
    return K_in[-prune_to:], V_in[-prune_to:]


def _ir_text(num_heads: int, head_dim: int, max_seq: int,
             new_tokens: int, prune_to: int) -> str:
    return (
        "graph_ir {\n"
        f"  %cache = tessera.kv_cache.create num_heads={num_heads} "
        f"head_dim={head_dim} max_seq={max_seq} dtype=f32\n"
        f"  %K_in  = tessera.placeholder shape=({new_tokens}, "
        f"{num_heads}, {head_dim}) dtype=f32\n"
        f"  %V_in  = tessera.placeholder shape=({new_tokens}, "
        f"{num_heads}, {head_dim}) dtype=f32\n"
        "  tessera.kv_cache.append(%cache, %K_in, %V_in)\n"
        f"  tessera.kv_cache.prune(%cache, max_entries={prune_to})\n"
        "  %K_out, %V_out = tessera.kv_cache.read(%cache)\n"
        "  return %K_out, %V_out\n"
        "}\n"
    )


def run(
    *, num_heads: int = 4, head_dim: int = 16, max_seq: int = 32,
    new_tokens: int = 20, prune_to: int = 12, seed: int = 0,
    native_required: bool = False,
) -> CompileReport:
    K_in, V_in, nh, hd, ms, nt, pt = _make_inputs(
        num_heads=num_heads, head_dim=head_dim, max_seq=max_seq,
        new_tokens=new_tokens, prune_to=prune_to, seed=seed,
    )
    is_darwin = sys.platform == "darwin"
    host_fail = classify_host(is_darwin=is_darwin, runtime_available=True)
    if native_required and host_fail is not None:
        raise TesseraNativeRequiredError(
            host_fail, target="apple_gpu", op_name=PROGRAM_ID,
        )
    target = "apple_gpu" if is_darwin else "cpu"
    target_decision = {
        target: (
            f"KVCacheHandle backend: paged numpy storage "
            f"(max_seq={ms}, num_heads={nh}, head_dim={hd}); "
            "fused FA-4 path consumes this state at read time"
        )
    }
    fallback_reason: FallbackReason | None = host_fail

    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        cache = KVCacheHandle(num_heads=nh, head_dim=hd, max_seq=ms, dtype="fp32")
        cache.append(K_in, V_in)
        cache.prune(pt)
        K_out, V_out = cache.read(0, cache.current_seq)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev_tracing)

    K_ref, V_ref = _numpy_reference(K_in, V_in, pt)
    max_abs_err = float(
        max(np.abs(K_out - K_ref).max(), np.abs(V_out - V_ref).max())
    )

    return CompileReport(
        program_id=PROGRAM_ID,
        source=f"{__name__}.run",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR,
        target=target,
        tessera_version=getattr(tessera, "__version__", ""),
        ir_hashes={"graph_ir": hash_ir_text(_ir_text(nh, hd, ms, nt, pt))},
        target_decision=target_decision,
        fallback_reason=fallback_reason,
        proof_routes=routes,
        timing_ms={"end_to_end": elapsed_ms},
        correctness={"max_abs_err": max_abs_err, "tolerance": 1e-6},
    )


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
