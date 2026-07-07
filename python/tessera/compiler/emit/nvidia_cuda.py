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
    OP_ATTENTION,
    OP_FUSED_REGION,
    OP_MATMUL,
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
from tessera.compiler.fusion_core import AttentionRegion, FusedRegion, MatmulRegion

_TARGET = "nvidia"
_LANG = "cuda"
_ENTRY = "tessera_nvidia_fused"
_ATTN_ENTRY = "tessera_nvidia_attn"
_REAL_TAG = "nvidia_cuda"
#: Max head dim (Dv) the one-thread-per-query flash kernel holds in its per-thread
#: online-softmax accumulator; larger Dv declines to the reference.
_ATTN_DV_CAP = 256


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


def _synthesize_attention_cuda() -> str:
    """CUDA source for ``O = softmax(scale * Q @ K^T) @ V`` (f32) — a **flash**
    kernel: one thread per query row streams the KV sequence with an online
    (numerically-stable) softmax, so no O(Nk) score buffer is needed. The
    per-thread output accumulator is capped at ``_ATTN_DV_CAP``; larger head dims
    are rejected (rc 2) and the runner declines to the reference. Q(M,D) row-major,
    K(Nk,D) row-major, V(Nk,Dv) row-major, O(M,Dv) row-major — the natural
    orientation the runner feeds after applying the region's transpose flags."""
    return (
        "#include <cuda_runtime.h>\n"
        "#include <math.h>\n"
        f"#define DV_CAP {_ATTN_DV_CAP}\n"
        f"__global__ void {_ATTN_ENTRY}_kernel(const float* Q, const float* K,\n"
        "        const float* V, float* O, int M, int Nk, int D, int Dv,\n"
        "        float scale, int causal) {\n"
        "    int m = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (m >= M) return;\n"
        "    float acc[DV_CAP];\n"
        "    for (int dv=0; dv<Dv; ++dv) acc[dv]=0.0f;\n"
        "    float mi = -INFINITY, li = 0.0f;\n"
        "    for (int n=0; n<Nk; ++n) {\n"
        "        if (causal && n > m) continue;\n"
        "        float s = 0.0f;\n"
        "        for (int d=0; d<D; ++d) s += Q[(long)m*D+d]*K[(long)n*D+d];\n"
        "        s *= scale;\n"
        "        float mnew = fmaxf(mi, s);\n"
        "        float corr = expf(mi - mnew);\n"
        "        float p = expf(s - mnew);\n"
        "        li = li*corr + p;\n"
        "        for (int dv=0; dv<Dv; ++dv) acc[dv] = acc[dv]*corr + p*V[(long)n*Dv+dv];\n"
        "        mi = mnew;\n"
        "    }\n"
        "    float inv = (li > 0.0f) ? 1.0f/li : 0.0f;\n"
        "    for (int dv=0; dv<Dv; ++dv) O[(long)m*Dv+dv] = acc[dv]*inv;\n"
        "}\n"
        f'extern "C" int {_ATTN_ENTRY}(const float* hQ, const float* hK,\n'
        "        const float* hV, float* hO, int M, int Nk, int D, int Dv,\n"
        "        float scale, int causal) {\n"
        "    if (Dv > DV_CAP) return 2;\n"
        "    size_t szQ=(size_t)M*D*4, szK=(size_t)Nk*D*4, szV=(size_t)Nk*Dv*4,\n"
        "           szO=(size_t)M*Dv*4;\n"
        "    float *dQ=0,*dK=0,*dV=0,*dO=0;\n"
        "    if (cudaMalloc(&dQ,szQ)!=cudaSuccess) return 3;\n"
        "    if (cudaMalloc(&dK,szK)!=cudaSuccess){cudaFree(dQ);return 3;}\n"
        "    if (cudaMalloc(&dV,szV)!=cudaSuccess){cudaFree(dQ);cudaFree(dK);return 3;}\n"
        "    if (cudaMalloc(&dO,szO)!=cudaSuccess){cudaFree(dQ);cudaFree(dK);cudaFree(dV);return 3;}\n"
        "    cudaMemcpy(dQ,hQ,szQ,cudaMemcpyHostToDevice);\n"
        "    cudaMemcpy(dK,hK,szK,cudaMemcpyHostToDevice);\n"
        "    cudaMemcpy(dV,hV,szV,cudaMemcpyHostToDevice);\n"
        "    int t=128, b=(M+t-1)/t;\n"
        f"    {_ATTN_ENTRY}_kernel<<<dim3(b), dim3(t)>>>(\n"
        "        dQ,dK,dV,dO,M,Nk,D,Dv,scale,causal);\n"
        "    int ok = (cudaDeviceSynchronize()==cudaSuccess) ? 1 : 3;\n"
        "    if (ok==1) cudaMemcpy(hO,dO,szO,cudaMemcpyDeviceToHost);\n"
        "    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);\n"
        "    return ok;\n"
        "}\n"
    )


class NvidiaCudaEmitter(KernelEmitter):
    target = _TARGET
    lang = _LANG

    def can_emit(self, region: Any) -> bool:
        return isinstance(region, (FusedRegion, AttentionRegion))

    def emit(self, region: Any, *, spec: SpecPolicy = SpecPolicy.BUCKET,
             dtype: str = "f32", dims: tuple[int, ...] | None = None) -> KernelSource:
        if not isinstance(region, (FusedRegion, AttentionRegion)):
            raise EmitError(
                f"NvidiaCudaEmitter cannot emit a region of type "
                f"{type(region).__name__} (FusedRegion / AttentionRegion; the "
                "shipped mma.sync GEMM lane serves single matmuls via the jit "
                "nvidia_mma executor)")
        if spec is SpecPolicy.DYNAMIC:
            raise EmitError("NvidiaCudaEmitter does not yet support SpecPolicy.DYNAMIC "
                            "(bucket/static only)")
        if dtype != "f32":
            raise EmitError(f"NvidiaCudaEmitter only supports f32 so far, got {dtype!r}")
        if isinstance(region, AttentionRegion):
            source, entry = _synthesize_attention_cuda(), _ATTN_ENTRY
        else:
            source, entry = _synthesize_fused_cuda(region), _ENTRY
        key = bucket_key(dims, spec, dim_names=getattr(region, "dim_names", None))
        return KernelSource(source=source, entry=entry, lang=self.lang,
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


def _load_attn_entry(artifact: str):
    """dlopen ``artifact`` (cached) and bind the flash-attention entry: ``int(Q, K,
    V, O, M, Nk, D, Dv, scale, causal)``."""
    lib = _LIB_CACHE.get(artifact)
    if lib is None:
        lib = ctypes.CDLL(artifact)
        _LIB_CACHE[artifact] = lib
    fn = getattr(lib, _ATTN_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
                   + [ctypes.c_float, ctypes.c_int])
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

    def run_fused_attention(self, region: Any, Q: Any, K: Any, V: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        # C4: the synthesized flash-attention lane — O = softmax(scale*Q@K^T)@V,
        # one query per thread, online softmax. Orient Q/K per the region's
        # transpose flags (f32), then build + launch; decline to the reference off
        # an NVIDIA GPU / for a head dim past the accumulator cap.
        import numpy as np
        try:
            Qn, Kn = region._natural(Q, K)          # f32, natural Q(M,D)/K(Nk,D)
            Qn = np.ascontiguousarray(Qn, np.float32)
            Kn = np.ascontiguousarray(Kn, np.float32)
            Vn = np.ascontiguousarray(V, np.float32)
            M, D = Qn.shape
            Nk, Dk = Kn.shape
            Nkv, Dv = Vn.shape
            if Dk != D or Nkv != Nk or Dv > _ATTN_DV_CAP:
                return region.reference(Q, K, V), "reference"
            compiled = build(region, _TARGET, dtype="f32", dims=None)
            fn = _load_attn_entry(compiled.artifact)
            out = np.zeros((M, Dv), np.float32)
            rc = fn(_ptr(Qn), _ptr(Kn), _ptr(Vn), _ptr(out),
                    M, Nk, D, Dv, ctypes.c_float(float(region.scale)),
                    1 if region.causal else 0)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
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


class NvidiaFlashAttnCandidate(Candidate):
    """Tier-1 (C4): the synthesized flash-attention CUDA lane
    (``O = softmax(scale·Q·Kᵀ)·V``, one query per thread, online softmax). Serves
    any ``AttentionRegion``; declines (to the reference) off an NVIDIA GPU / for a
    head dim past the accumulator cap. NVIDIA has no *shipped* attention kernel, so
    this correctness-first synth is the only attention candidate — an mma.sync
    tensor-core flash version is the perf follow-on."""

    name = "nvidia_flash_attn"
    tier = Tier.SYNTHESIZED
    target = _TARGET
    op = OP_ATTENTION

    def run(self, region: Any, Q: Any, K: Any, V: Any,
            *a: Any, **k: Any) -> tuple[Any, str]:
        return _SHARED_RUNNER.run_fused_attention(region, Q, K, V)


# ── D1 matmul candidates (B1) — bare GEMM, Tier-2 emitted vs Tier-3 shipped ────
#
# The arbiter enumerates these per (target="nvidia", op=matmul) and F4-gates each.
# Tier-priority (Decision #28) prefers the hand-tuned shipped lane by default; D2's
# measured loop lets the emitted lane win where it is faster + in accuracy budget.
# Both are 16-bit storage (bf16/f16) → f32 accumulate, so they declare the f16
# budget the oracle honors. Off an NVIDIA GPU / without the built libs they decline
# to the reference and drop out of the enumeration.

_GEMM_F16_ATOL = 5e-3          # 16-bit storage vs the f32 reference (Decision #28)
_GEMM_DTYPES = ("bfloat16", "float16")


def _aligned_2d(A: Any, B: Any) -> bool:
    """A (M,K) @ B (K,N) with the emitted kernel's tile alignment (M%16,N%8,K%16)."""
    import numpy as np
    Aa, Ba = np.asarray(A), np.asarray(B)
    if Aa.ndim != 2 or Ba.ndim != 2 or Aa.shape[1] != Ba.shape[0]:
        return False
    M, K = Aa.shape
    _, N = Ba.shape
    return M % 16 == 0 and N % 8 == 0 and K % 16 == 0


class NvidiaMmaGemmShippedCandidate(Candidate):
    """Tier-3 (hand-tuned): the shipped ``libtessera_nvidia_gemm`` mma.sync GEMM —
    the crown-jewel lane, arbiter default until D2 measures otherwise. Serves any
    (unaligned OK) bf16/f16 matmul; declines off an NVIDIA GPU."""

    name = "nvidia_mma_gemm_shipped"
    tier = Tier.HAND_TUNED
    target = _TARGET
    op = OP_MATMUL
    accuracy_atol = _GEMM_F16_ATOL

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return rt._nvidia_mma_runtime_available()
        except Exception:
            return False

    def applies_to(self, region: Any) -> bool:
        return isinstance(region, MatmulRegion) and region.dtype in _GEMM_DTYPES

    def run(self, region: Any, A: Any, B: Any, *a: Any, **k: Any) -> tuple[Any, str]:
        try:
            from tessera import runtime as rt
            return rt._nvidia_mma_gemm_2d(A, B, region.dtype), "nvidia_mma_shipped"
        except Exception:
            return region.reference(A, B), "reference"


class NvidiaMmaGemmEmittedCandidate(Candidate):
    """Tier-2 (emitted): the compiler-EMITTED ``ptx_emit`` mma.sync GEMM driven
    through the launch bridge — the C2 emit lane as a first-class arbiter candidate.
    Serves ALIGNED (M%16/N%8/K%16) bf16/f16 matmuls; declines (to the reference) for
    ragged shapes or off an NVIDIA GPU / without the built bridge."""

    name = "nvidia_mma_gemm_emitted"
    tier = Tier.EMITTED
    target = _TARGET
    op = OP_MATMUL
    accuracy_atol = _GEMM_F16_ATOL

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return (rt._load_nvidia_ptx_launch() is not None
                    and rt._nvidia_mma_runtime_available())
        except Exception:
            return False

    def applies_to(self, region: Any) -> bool:
        return isinstance(region, MatmulRegion) and region.dtype in _GEMM_DTYPES

    def run(self, region: Any, A: Any, B: Any, *a: Any, **k: Any) -> tuple[Any, str]:
        if not _aligned_2d(A, B):              # emitter is aligned-only (for now)
            return region.reference(A, B), "reference"
        try:
            from tessera import runtime as rt
            return rt._nvidia_ptx_gemm_2d(A, B, region.dtype), "nvidia_ptx_gemm"
        except Exception:
            return region.reference(A, B), "reference"


# ── registration (import side effect, exactly like rocm_hip / x86_llvm) ────────
register_emitter(NvidiaCudaEmitter())
register_compiler(_TARGET, _nvidia_cuda_compile_fn)
register_runner(NvidiaCudaRunner(), default=False)

register_candidate(NvidiaGenericCudaCandidate())
register_candidate(NvidiaFlashAttnCandidate())        # C4: synthesized attention
# Bare-GEMM lanes: hand-tuned shipped (Tier 3) + compiler-emitted (Tier 2).
register_candidate(NvidiaMmaGemmShippedCandidate())
register_candidate(NvidiaMmaGemmEmittedCandidate())
