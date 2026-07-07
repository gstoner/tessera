"""Workstream C — ROCm gfx1151 plugin: wire the shipped hardware-verified
gfx1151 kernels into the target-agnostic F4 oracle.

Unlike the x86 plugin (which *emits* + compiles C), ROCm's lead kernels are the
already-shipped, hardware-verified gfx1151 lanes (WMMA GEMM, compiled FA-2
flash-attention). So this module registers a :class:`KernelRunner` **only** — no
emitter, no ``compile_fn`` — which lets ROCm's real kernels be gated by the same
universal F4 correctness oracle as the synthesized backends (the cross-backend
differential-equivalence superpower, Theory §7.5) *without* claiming a generic
ROCm emit lane (that is C3). For region kinds ROCm has no single fused GPU kernel
for yet (matmul+epilogue / gated / pointwise), it declines to the numpy reference
— honest, never a mislabeled kernel (Decision #21).

Precision: the flash-attn lane is f16 storage / f32 accumulate, so the runner
declares an f16 accuracy budget (:attr:`accuracy_atol`); the oracle widens its
tolerance to it so f16 rounding is not misread as a miscompile, while an O(1)
miscompile is still caught (Decision #28, the accuracy-budgeted arbiter).

Runs only where a live gfx1151 + the compiled flash lane are present (probed via
``runtime._rocm_compiled_flash_attn_available``); on any other host it declines
to the reference, so authoring/tests stay host-free.
"""
from __future__ import annotations

from typing import Any

from tessera.compiler.emit.kernel_emitter import KernelRunner, register_runner

_TARGET = "rocm"
_REAL_TAG = "rocm_hip"
#: f16 storage budget for the WMMA / flash lanes vs the f32 reference. Loose
#: enough for f16 rounding (measured max ~2.5e-3 on the oracle probes), tight
#: enough that an O(1) miscompile (transpose / wrong softmax / wrong scale) is
#: still caught.
_F16_ATOL = 5e-3


class RocmHipRunner(KernelRunner):
    target = _TARGET
    accuracy_atol = _F16_ATOL

    def run_fused_attention(self, region: Any, Q: Any, K: Any, V: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        import numpy as np
        try:
            from tessera import runtime as rt
            if not rt._rocm_compiled_flash_attn_available():
                return region.reference(Q, K, V), "reference"
            Qn, Kn = region._natural(Q, K)          # orient per transpose flags (f32)
            Vn = np.asarray(V, np.float32)
            M, D = Qn.shape
            Nk, Dk = Kn.shape
            if D % 16 != 0 or Dk != D:              # WMMA needs head_dim % 16 == 0
                return region.reference(Q, K, V), "reference"
            q = np.ascontiguousarray(Qn.reshape(1, 1, M, D), np.float16)
            kk = np.ascontiguousarray(Kn.reshape(1, 1, Nk, D), np.float16)
            v = np.ascontiguousarray(Vn.reshape(1, 1, Nk, D), np.float16)
            out = np.asarray(rt._rocm_flash_attn(q, kk, v, scale=region.scale,
                                                 causal=region.causal))
            return out.reshape(M, D).astype(np.float32), _REAL_TAG
        except Exception:
            return region.reference(Q, K, V), "reference"

    # No single fused GPU kernel for these yet (that is the C3 emit lane) —
    # decline honestly to the numpy reference.
    def run_fused_region(self, region: Any, A: Any, B: Any, bias: Any = None,
                         *a: Any, residual: Any = None,
                         **k: Any) -> tuple[Any, str]:
        return region.reference(A, B, bias, residual), "reference"

    def run_gated_matmul_region(self, region: Any, A: Any, Wg: Any, Wu: Any,
                                *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(A, Wg, Wu), "reference"

    def run_pointwise_graph(self, region: Any, arrays: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(*arrays), "reference"


# Runner only (no emitter/compile_fn — ROCm's kernels are shipped, not
# synthesized here). default=False so Apple stays the active default runner.
register_runner(RocmHipRunner(), default=False)
