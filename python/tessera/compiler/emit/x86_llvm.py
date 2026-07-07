"""Workstream C1 — x86 (Zen 5) codegen plugin. Mirrors ``emit/apple_msl.py``.

Three registered seams against the target-agnostic synthesizer (``fusion_core``),
exactly the shape ``WORKSTREAM_C_HANDOFF.md`` prescribes:

* :class:`X86CEmitter` (``register_emitter``) — a ``FusedRegion`` → portable C
  source (matmul + prologue/epilogue/residual/reduction), f32.
* :func:`_x86_compile_fn` (``register_compiler``) — ``cc``/``clang -O3
  -march=native -shared`` → a ``.so`` path (real ahead-of-time compile, not the
  Apple compile-on-launch deferral).
* :class:`X86CRunner` (``register_runner``, ``default=False``) — ``ctypes``
  dlopen + launch → ``(out, "x86_native")`` when the kernel ran, else the numpy
  reference tagged ``"reference"`` (Decision #21: never mislabel a fallback).

The op-name → C-snippet tables are maintained HERE because the shared
``EpilogueOp.emit(target)`` deliberately raises for non-Metal targets (no silent
wrong-language emit). Each C snippet matches its ``EPILOGUE_OPS``/``REDUCTION_OPS``
numpy reference so the F4 oracle (``verify_synthesized_region``) gates this
backend for real on the Zen 5 box; on a host without a C compiler the runner
skip-cleans to the reference (honest, host-free-safe).

Scope: the f32 ``FusedRegion`` hot path (the fusable-DAG middle ground). Other
region kinds / dtypes decline via :class:`EmitError` (emit) or a ``"reference"``
tag (run) — never a mislabeled kernel. Widen ``can_emit`` as more kinds land.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from typing import Any

from tessera.compiler.emit._fused_scalar_body import row_compute_body
from tessera.compiler.emit.kernel_cache import build, register_compiler
from tessera.compiler.emit.kernel_emitter import (
    EmitError,
    KernelEmitter,
    KernelRunner,
    KernelSource,
    SpecPolicy,
    bucket_key,
    register_emitter,
    register_runner,
)
from tessera.compiler.fusion_core import FusedRegion

_TARGET = "x86"
_LANG = "c"
_ENTRY = "tessera_x86_fused"
_REAL_TAG = "x86_native"


def _synthesize_fused_c(region: FusedRegion) -> str:
    """Emit the C source for a ``FusedRegion`` (f32) — a plain host function that
    loops over rows and embeds the shared per-row body. Signature is dims-invariant
    (M/N/K are runtime args), so one kernel serves every shape — the arbiter/cache
    key it shape-anonymously. The per-row math is shared with the ROCm HIP lane
    (`_fused_scalar_body.row_compute_body`) so both stay locked to one reference."""
    return (
        "#include <math.h>\n"
        f"int {_ENTRY}(const float* A, const float* B, const float* bias,\n"
        "               const float* residual, float* out,\n"
        "               int M, int N, int K) {\n"
        "    for (int m = 0; m < M; ++m) {\n"
        "        float* row = out + (long)m * N;\n"
        f"{row_compute_body(region)}"
        "    }\n"
        "    return 1;\n"
        "}\n"
    )


# ── Seam 1: emitter ───────────────────────────────────────────────────────────

class X86CEmitter(KernelEmitter):
    target = _TARGET
    lang = _LANG

    def can_emit(self, region: Any) -> bool:
        return isinstance(region, FusedRegion)

    def emit(self, region: Any, *, spec: SpecPolicy = SpecPolicy.BUCKET,
             dtype: str = "f32", dims: tuple[int, ...] | None = None) -> KernelSource:
        if not isinstance(region, FusedRegion):
            raise EmitError(
                f"X86CEmitter cannot emit a region of type {type(region).__name__} "
                "(only FusedRegion so far)")
        if spec is SpecPolicy.DYNAMIC:
            raise EmitError("X86CEmitter does not yet support SpecPolicy.DYNAMIC "
                            "(bucket/static only)")
        if dtype != "f32":
            raise EmitError(f"X86CEmitter only supports f32 so far, got {dtype!r}")
        source = _synthesize_fused_c(region)
        key = bucket_key(dims, spec, dim_names=getattr(region, "dim_names", None))
        return KernelSource(source=source, entry=_ENTRY, lang=self.lang,
                            spec=spec, shape_key=key)


# ── Seam 2: compile_fn (source → .so) ─────────────────────────────────────────

def _cc() -> str:
    """The C compiler to use: ``$TESSERA_X86_CC`` override, else clang, else cc.
    (Zen 5 has no AMX — ``-march=native`` enables AVX-512 without hardcoding a
    flag that could fail on the NR2 Pro's non-AVX-512 host.)"""
    return (os.environ.get("TESSERA_X86_CC")
            or shutil.which("clang") or shutil.which("cc")
            or shutil.which("gcc") or "cc")


def _x86_compile_fn(source: KernelSource) -> str:
    """Compile the emitted C to a shared object and return its path. Raises on a
    toolchain/compile failure; ``build`` wraps it in ``CompileError`` (never a
    silent no-op)."""
    d = tempfile.mkdtemp(prefix="tessera_x86_")
    src = os.path.join(d, "kernel.c")
    so = os.path.join(d, "kernel.so")
    with open(src, "w") as f:
        f.write(source.source)
    subprocess.run(
        [_cc(), "-O3", "-march=native", "-fPIC", "-shared", src, "-o", so, "-lm"],
        check=True, capture_output=True, text=True)
    return so


# ── Seam 3: runner (execute → (out, tag)) ─────────────────────────────────────

_LIB_CACHE: dict[str, Any] = {}


def _load_entry(artifact: str):
    """dlopen ``artifact`` (cached) and return its bound entry symbol with the
    fixed C ABI: ``int(A, B, bias, residual, out, M, N, K)``."""
    lib = _LIB_CACHE.get(artifact)
    if lib is None:
        lib = ctypes.CDLL(artifact)
        _LIB_CACHE[artifact] = lib
    fn = getattr(lib, _ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
    return fn


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.c_void_p) if arr is not None else None


class X86CRunner(KernelRunner):
    target = _TARGET

    def run_fused_region(self, region: Any, A: Any, B: Any, bias: Any = None,
                         *args: Any, residual: Any = None,
                         **kwargs: Any) -> tuple[Any, str]:
        import numpy as np
        # Required-buffer guard BEFORE any launch: the emitted C dereferences
        # bias[n] / residual[...] whenever the region declares them, so a missing
        # buffer would pass a NULL pointer and segfault PAST Python's ``except``
        # (an uncatchable SIGSEGV). Route such an ill-formed call through the numpy
        # reference instead, which raises a clean, catchable ValueError naming the
        # missing operand — never launch the kernel with a null it will deref.
        if (region.has_bias and bias is None) or \
                (region.has_residual and residual is None):
            return region.reference(A, B, bias, residual), "reference"
        try:
            Af = np.ascontiguousarray(A, np.float32)
            Bf = np.ascontiguousarray(B, np.float32)
            M, K = Af.shape
            _, N = Bf.shape
            # Shape-anonymous build: the C kernel takes M/N/K at runtime, so one
            # compiled artifact serves every shape (dims=None → no shape key).
            compiled = build(region, _TARGET, dtype="f32", dims=None)
            fn = _load_entry(compiled.artifact)
            bias_arr = (np.ascontiguousarray(bias, np.float32)
                        if bias is not None else None)
            res_arr = (np.ascontiguousarray(residual, np.float32)
                       if residual is not None else None)
            out = np.zeros((M, N), np.float32)
            rc = fn(_ptr(Af), _ptr(Bf), _ptr(bias_arr), _ptr(res_arr),
                    _ptr(out), M, N, K)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
        return region.reference(A, B, bias, residual), "reference"

    # x86 has no fused GPU-style kernel for these yet — decline honestly (the
    # numpy reference, tagged so the oracle trusts rather than gates it).
    def run_fused_attention(self, region: Any, Q: Any, K: Any, V: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(Q, K, V), "reference"

    def run_gated_matmul_region(self, region: Any, A: Any, Wg: Any, Wu: Any,
                                *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(A, Wg, Wu), "reference"

    def run_pointwise_graph(self, region: Any, arrays: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(*arrays), "reference"


# ── registration (import side effect, exactly like apple_msl) ─────────────────
register_emitter(X86CEmitter())
register_compiler(_TARGET, _x86_compile_fn)
register_runner(X86CRunner(), default=False)
