"""Helpers for connecting benchmark rows to the current Tessera compiler path."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompilerRun:
    output: Any
    latency_ms: float
    uses_compiled_path: bool
    graph_ir: str | None
    schedule_ir: str | None
    tile_ir: str | None
    target_ir: str | None
    lowering: str


def _import_tessera():
    try:
        import tessera  # type: ignore
    except Exception:
        return None
    return tessera


def compiler_matmul_relu(a: np.ndarray, b: np.ndarray, tile: tuple[int, int, int]) -> CompilerRun | None:
    """Run a tiny current-compiler path: Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU.

    This intentionally uses the supported Phase 1 CPU-lowered subset
    (`matmul -> relu`) so benchmark smoke runs can prove that benchmark code is
    still attached to the active compiler surface.
    """

    tessera = _import_tessera()
    if tessera is None:
        return None

    @tessera.jit(cpu_tile=tile)
    def bench_kernel(x, w):
        y = tessera.ops.matmul(x, w)
        return tessera.ops.relu(y)

    start = time.perf_counter()
    out = bench_kernel(a, b)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return CompilerRun(
        output=out,
        latency_ms=latency_ms,
        uses_compiled_path=bench_kernel.uses_compiled_path,
        graph_ir=bench_kernel.ir_text(),
        schedule_ir=bench_kernel.schedule_ir,
        tile_ir=bench_kernel.tile_ir,
        target_ir=bench_kernel.target_ir,
        lowering=bench_kernel.explain_lowering(),
    )


def compiler_flash_attention_ir() -> dict[str, Any]:
    """Emit current Graph IR for flash attention if Tessera is importable.

    Flash attention is not in the narrow CPU executable subset yet, so this
    helper reports artifact support without pretending to benchmark a compiled
    kernel.
    """

    tessera = _import_tessera()
    if tessera is None:
        return {"available": False, "reason": "tessera import failed"}

    @tessera.jit
    def flash_attention_kernel(q, k, v):
        return tessera.ops.flash_attn(q, k, v)

    return {
        "available": True,
        "uses_compiled_path": flash_attention_kernel.uses_compiled_path,
        "graph_ir": flash_attention_kernel.ir_text(),
        "schedule_ir": flash_attention_kernel.schedule_ir,
        "tile_ir": flash_attention_kernel.tile_ir,
        "target_ir": flash_attention_kernel.target_ir,
        "lowering": flash_attention_kernel.explain_lowering(),
    }
