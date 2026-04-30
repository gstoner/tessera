"""Helpers that connect benchmarks to the current Tessera compiler surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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

    @property
    def artifact_hash(self) -> str:
        parts = [self.graph_ir or "", self.schedule_ir or "", self.tile_ir or "", self.target_ir or ""]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _import_tessera():
    try:
        import tessera  # type: ignore
    except Exception:
        return None
    return tessera


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compiler_matmul_relu(a: np.ndarray, b: np.ndarray, tile: tuple[int, int, int]) -> CompilerRun | None:
    """Run the supported Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU path."""

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


def _artifact_info(fn, op_name: str) -> dict[str, Any]:
    try:
        graph_ir = fn.ir_text()
    except Exception as exc:
        return {"available": False, "reason": str(exc), "op": op_name}
    schedule_ir = getattr(fn, "schedule_ir", None)
    tile_ir = getattr(fn, "tile_ir", None)
    target_ir = getattr(fn, "target_ir", None)
    parts = [graph_ir or "", schedule_ir or "", tile_ir or "", target_ir or ""]
    return {
        "available": True,
        "op": op_name,
        "uses_compiled_path": bool(getattr(fn, "uses_compiled_path", False)),
        "graph_ir": graph_ir,
        "schedule_ir": schedule_ir,
        "tile_ir": tile_ir,
        "target_ir": target_ir,
        "artifact_hash": _hash_text("\n".join(parts)),
        "lowering": fn.explain_lowering() if hasattr(fn, "explain_lowering") else "",
    }


def compiler_flash_attention_ir() -> dict[str, Any]:
    tessera = _import_tessera()
    if tessera is None:
        return {"available": False, "reason": "tessera import failed", "op": "flash_attn"}

    @tessera.jit
    def flash_attention_kernel(q, k, v):
        return tessera.ops.flash_attn(q, k, v)

    return _artifact_info(flash_attention_kernel, "flash_attn")


def compiler_conv2d_ir() -> dict[str, Any]:
    tessera = _import_tessera()
    if tessera is None:
        return {"available": False, "reason": "tessera import failed", "op": "conv2d"}

    @tessera.jit
    def conv2d_kernel(x, w):
        return tessera.ops.conv2d(x, w, stride=1, padding=1)

    return _artifact_info(conv2d_kernel, "conv2d")


def compiler_spectral_ir(op: str) -> dict[str, Any]:
    tessera = _import_tessera()
    if tessera is None:
        return {"available": False, "reason": "tessera import failed", "op": op}

    if op in {"fft1d", "fft2d", "spectrum"}:
        @tessera.jit
        def spectral_kernel(x):
            return tessera.ops.fft(x)
    elif op == "dct2":
        @tessera.jit
        def spectral_kernel(x):
            return tessera.ops.dct(x, type=2)
    else:
        @tessera.jit
        def spectral_kernel(x, w):
            return tessera.ops.spectral_conv(x, w)

    return _artifact_info(spectral_kernel, op)
