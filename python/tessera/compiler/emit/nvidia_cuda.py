"""Workstream C2 — NVIDIA (sm_120) codegen plugin: generic synth → CUDA.

The NVIDIA counterpart to ``emit/rocm_hip.py`` / ``emit/x86_llvm.py`` — the three
registered seams the target-agnostic synthesizer (``fusion_core``) calls into, so
NVIDIA gains the generic **compiled** middle-ground lane it lacks today (the
shipped ``libtessera_nvidia_gemm.so`` is a *pure* mma.sync GEMM with no fused
epilogue, dispatched by the jit ``nvidia_mma`` executor — it cannot serve a
``FusedRegion``, which always carries at least one fused feature):

* :class:`NvidiaCudaEmitter` (``register_emitter``) — a ``FusedRegion`` → CUDA
  source (a ``__global__`` one-thread-per-row kernel + a host-pointer C-ABI
  wrapper doing H2D / launch / D2H), reusing the *same* scalar body as the x86 C
  and ROCm HIP lanes (`_fused_scalar_body.row_compute_body`) so all three stay
  locked to the one ``fusion_core`` numpy reference.
* :func:`_nvidia_cuda_compile_fn` (``register_compiler``) — ``nvcc
  -arch=sm_120a -O3 --shared`` → a ``.so`` the runtime dlopens (real
  ahead-of-time compile, not the Apple compile-on-launch deferral).
* :class:`NvidiaCudaRunner` (``register_runner``, ``default=False``) —
  ``ctypes`` dlopen + launch → ``(out, "nvidia_cuda")`` when the kernel ran, else
  the numpy reference tagged ``"reference"`` (Decision #21: never mislabel a
  fallback).

Lead-safety (Decision #28): this generic CUDA kernel is a correctness-first
candidate for the fusable middle ground (epilogues / pointwise chains) — the
crown-jewel ``wgmma`` / ``mma.sync`` GEMM and (future) fused attention stay
first-class; the D1 arbiter picks the generic lane only where it measures faster
and in budget. Runs only where a live NVIDIA GPU + ``nvcc`` are present;
everywhere else it declines to the numpy reference so authoring/tests stay
host-free.

Scope: the f32 ``FusedRegion`` hot path. Other region kinds / dtypes decline via
:class:`EmitError` (emit) or a ``"reference"`` tag (run) — never a mislabeled
kernel. The Tier-2 ``ptx_emit.py`` (``mma.sync``/``wgmma``) emit lane and its
``ptxas``→CUBIN→launch bridge are C2's on-box remainder (see
``COMPILER_REFACTOR_PLAN.md`` §9.1(2)).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from typing import Any

from tessera.compiler.emit._fused_scalar_body import row_compute_body
from tessera.compiler.emit.candidate import (
    OP_FUSED_REGION,
    Candidate,
    Tier,
    register_candidate,
)
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

_TARGET = "nvidia"
_LANG = "cuda"
_ENTRY = "tessera_nvidia_fused"
_REAL_TAG = "nvidia_cuda"


# ── CUDA source synthesis (generic FusedRegion lane) ──────────────────────────

def _synthesize_fused_cuda(region: FusedRegion) -> str:
    """CUDA source for a ``FusedRegion`` (f32): a one-thread-per-row ``__global__``
    kernel embedding the shared scalar body, plus a host-pointer C-ABI wrapper that
    does H2D / launch / D2H (same shape as the shipped ``libtessera_nvidia_gemm.so``
    symbols). Dims are runtime args, so one kernel serves every shape."""
    return (
        "#include <cuda_runtime.h>\n"
        "#include <math.h>\n"
        f"__global__ void {_ENTRY}_kernel(const float* A, const float* B,\n"
        "        const float* bias, const float* residual, float* out,\n"
        "        int M, int N, int K) {\n"
        "    int m = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (m >= M) return;\n"
        "    float* row = out + (long)m * N;\n"
        f"{row_compute_body(region)}"
        "}\n"
        f'extern "C" int {_ENTRY}(const float* hA, const float* hB,\n'
        "        const float* hbias, const float* hresidual, float* hout,\n"
        "        int M, int N, int K) {\n"
        "    size_t szA=(size_t)M*K*sizeof(float), szB=(size_t)K*N*sizeof(float),\n"
        "           szO=(size_t)M*N*sizeof(float);\n"
        "    float *dA=0,*dB=0,*dbias=0,*dres=0,*dO=0;\n"
        "    if (cudaMalloc(&dA,szA)!=cudaSuccess) return 2;\n"
        "    if (cudaMalloc(&dB,szB)!=cudaSuccess) { cudaFree(dA); return 2; }\n"
        "    if (cudaMalloc(&dO,szO)!=cudaSuccess) { cudaFree(dA); cudaFree(dB); return 2; }\n"
        "    cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);\n"
        "    cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice);\n"
        "    if (hbias) { cudaMalloc(&dbias,(size_t)N*sizeof(float));\n"
        "        cudaMemcpy(dbias,hbias,(size_t)N*sizeof(float),cudaMemcpyHostToDevice); }\n"
        "    if (hresidual) { cudaMalloc(&dres,szO);\n"
        "        cudaMemcpy(dres,hresidual,szO,cudaMemcpyHostToDevice); }\n"
        "    int t=64, b=(M+t-1)/t;\n"
        f"    {_ENTRY}_kernel<<<dim3(b), dim3(t)>>>(\n"
        "        dA,dB,dbias,dres,dO,M,N,K);\n"
        "    int ok = (cudaDeviceSynchronize()==cudaSuccess) ? 1 : 3;\n"
        "    if (ok==1) cudaMemcpy(hout,dO,szO,cudaMemcpyDeviceToHost);\n"
        "    cudaFree(dA); cudaFree(dB); cudaFree(dO);\n"
        "    if (dbias) cudaFree(dbias);\n"
        "    if (dres) cudaFree(dres);\n"
        "    return ok;\n"
        "}\n"
    )


class NvidiaCudaEmitter(KernelEmitter):
    target = _TARGET
    lang = _LANG

    def can_emit(self, region: Any) -> bool:
        return isinstance(region, FusedRegion)

    def emit(self, region: Any, *, spec: SpecPolicy = SpecPolicy.BUCKET,
             dtype: str = "f32", dims: tuple[int, ...] | None = None) -> KernelSource:
        if not isinstance(region, FusedRegion):
            raise EmitError(
                f"NvidiaCudaEmitter cannot emit a region of type "
                f"{type(region).__name__} (only FusedRegion; the shipped mma.sync "
                "GEMM lane serves single matmuls via the jit nvidia_mma executor)")
        if spec is SpecPolicy.DYNAMIC:
            raise EmitError("NvidiaCudaEmitter does not yet support SpecPolicy.DYNAMIC "
                            "(bucket/static only)")
        if dtype != "f32":
            raise EmitError(f"NvidiaCudaEmitter only supports f32 so far, got {dtype!r}")
        source = _synthesize_fused_cuda(region)
        key = bucket_key(dims, spec, dim_names=getattr(region, "dim_names", None))
        return KernelSource(source=source, entry=_ENTRY, lang=self.lang,
                            spec=spec, shape_key=key)


# ── compile_fn (CUDA → .so) ───────────────────────────────────────────────────

def _nvidia_arch() -> str:
    """sm target: ``$TESSERA_NVIDIA_ARCH`` override, else sm_120a (the NR2 Pro
    Blackwell default per COMPILER_REFACTOR_PLAN §7.4). A plain scalar kernel
    compiles for any arch, so this only pins the SASS ISA."""
    return os.environ.get("TESSERA_NVIDIA_ARCH") or "sm_120a"


def _nvcc() -> str:
    """The CUDA compiler: ``$TESSERA_NVCC`` override, else nvcc on PATH, else the
    default toolkit location."""
    return (os.environ.get("TESSERA_NVCC")
            or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")


def _nvidia_cuda_compile_fn(source: KernelSource) -> str:
    """Compile the emitted CUDA to a shared object with nvcc and return its path.
    Raises on a missing toolchain/compile failure; ``build`` wraps it in
    ``CompileError`` (never a silent no-op)."""
    d = tempfile.mkdtemp(prefix="tessera_nvidia_")
    src = os.path.join(d, "kernel.cu")
    so = os.path.join(d, "kernel.so")
    with open(src, "w") as f:
        f.write(source.source)
    subprocess.run(
        [_nvcc(), f"-arch={_nvidia_arch()}", "-O3", "--shared",
         "-Xcompiler", "-fPIC", src, "-o", so],
        check=True, capture_output=True, text=True)
    return so


# ── runner (execute → (out, tag)) ─────────────────────────────────────────────

_LIB_CACHE: dict[str, Any] = {}


def _load_entry(artifact: str):
    """dlopen ``artifact`` (cached) and return its bound entry symbol with the fixed
    C ABI: ``int(A, B, bias, residual, out, M, N, K)``."""
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


class NvidiaCudaRunner(KernelRunner):
    target = _TARGET

    def run_fused_region(self, region: Any, A: Any, B: Any, bias: Any = None,
                         *args: Any, residual: Any = None,
                         **kwargs: Any) -> tuple[Any, str]:
        import numpy as np
        # Required-buffer guard BEFORE launch: the emitted CUDA dereferences
        # bias[n] / residual[...] whenever the region declares them, so a missing
        # buffer would pass a null the kernel derefs (an uncatchable SIGSEGV past
        # Python's ``except``). Route ill-formed calls through the reference (a
        # clean, catchable ValueError) instead of launching with a null.
        if (region.has_bias and bias is None) or \
                (region.has_residual and residual is None):
            return region.reference(A, B, bias, residual), "reference"
        try:
            Af = np.ascontiguousarray(A, np.float32)
            Bf = np.ascontiguousarray(B, np.float32)
            M, K = Af.shape
            _, N = Bf.shape
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

    # No single fused CUDA kernel for these yet — decline honestly (the numpy
    # reference, tagged so the oracle trusts rather than gates it).
    def run_fused_attention(self, region: Any, Q: Any, K: Any, V: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(Q, K, V), "reference"

    def run_gated_matmul_region(self, region: Any, A: Any, Wg: Any, Wu: Any,
                                *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(A, Wg, Wu), "reference"

    def run_pointwise_graph(self, region: Any, arrays: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(*arrays), "reference"


# ── D1 candidate (C2 Tier-1) ──────────────────────────────────────────────────
#
# Only the generic synthesized lane is a candidate today: NVIDIA has no *fused*
# hand-tuned kernel to register as a Tier-3 FusedRegion candidate (the shipped
# mma.sync kernel is a pure GEMM). The Tier-2 emitted lane (ptx_emit mma.sync/
# wgmma) and any Tier-3 fused kernel land with C2's on-box launch bridge.

_SHARED_RUNNER = NvidiaCudaRunner()


class NvidiaGenericCudaCandidate(Candidate):
    """Tier-1: the generic one-thread-per-row CUDA lane (arch-agnostic synth). Serves
    any ``FusedRegion`` — the floor-raising middle ground that is correctness-first,
    not a matrix-core GEMM. Declines (to the reference) off an NVIDIA GPU / without
    ``nvcc``, so it drops out of the arbiter's enumeration there."""

    name = "nvidia_generic_cuda"
    tier = Tier.SYNTHESIZED
    target = _TARGET
    op = OP_FUSED_REGION

    def run(self, region: Any, A: Any, B: Any, bias: Any = None,
            residual: Any = None, *a: Any, **k: Any) -> tuple[Any, str]:
        # residual is positional-or-keyword (matching the A,B,bias,residual
        # reference ABI) so the arbiter's positional inputs thread it instead of
        # dropping it into *a — else a residual fusion hits the missing-buffer
        # guard and raises (PR #290 review).
        return _SHARED_RUNNER.run_fused_region(region, A, B, bias,
                                               residual=residual)


# ── registration (import side effect, exactly like rocm_hip / x86_llvm) ────────
register_emitter(NvidiaCudaEmitter())
register_compiler(_TARGET, _nvidia_cuda_compile_fn)
register_runner(NvidiaCudaRunner(), default=False)

register_candidate(NvidiaGenericCudaCandidate())
