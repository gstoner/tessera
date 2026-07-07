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
import re
import shutil
import subprocess
import tempfile
from typing import Any

from tessera.compiler.emit._fused_scalar_body import row_compute_body
from tessera.compiler.emit.candidate import (
    OP_ATTENTION,
    OP_FUSED_REGION,
    OP_GATED_MATMUL,
    OP_MATMUL,
    OP_POINTWISE,
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
from tessera.compiler.fusion_core import (
    POINTWISE_OPS,
    AttentionRegion,
    FusedRegion,
    GatedMatmulRegion,
    MatmulRegion,
    PointwiseGraphRegion,
)

_TARGET = "nvidia"
_LANG = "cuda"
_ENTRY = "tessera_nvidia_fused"
_ATTN_ENTRY = "tessera_nvidia_attn"
_GATED_ENTRY = "tessera_nvidia_gated"
_PW_ENTRY = "tessera_nvidia_pointwise"
_REAL_TAG = "nvidia_cuda"
#: Max head dim (Dv) the one-thread-per-query flash kernel holds in its per-thread
#: online-softmax accumulator; larger Dv declines to the reference.
_ATTN_DV_CAP = 256
#: 16-bit storage accuracy budget vs the f32 reference (Decision #28) — shared by
#: the tensor-core fused lane and the bf16/f16 GEMM candidates.
_GEMM_F16_ATOL = 5e-3
_GEMM_DTYPES = ("bfloat16", "float16")


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


def _synthesize_gated_cuda(region: GatedMatmulRegion) -> str:
    """CUDA source for the SwiGLU gate ``O = f(A @ Wg) ⊙ (A @ Wu)`` (f32) — one
    thread per output row, sharing the A load across the two K-contractions, then
    the gate activation + elementwise multiply. A(M,K), Wg/Wu(K,H), O(M,H)."""
    from tessera.compiler.emit._fused_scalar_body import pointwise_snippet
    act = pointwise_snippet(region.gate_act, "g")     # e.g. g = g/(1+expf(-g));
    return (
        "#include <cuda_runtime.h>\n"
        "#include <math.h>\n"
        f"__global__ void {_GATED_ENTRY}_kernel(const float* A, const float* Wg,\n"
        "        const float* Wu, float* O, int M, int K, int H) {\n"
        "    int m = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (m >= M) return;\n"
        "    for (int h=0; h<H; ++h) {\n"
        "        float g=0.0f, u=0.0f;\n"
        "        for (int k=0; k<K; ++k) { float a=A[(long)m*K+k];\n"
        "            g += a*Wg[(long)k*H+h]; u += a*Wu[(long)k*H+h]; }\n"
        f"        {act}\n"
        "        O[(long)m*H+h] = g * u;\n"
        "    }\n"
        "}\n"
        f'extern "C" int {_GATED_ENTRY}(const float* hA, const float* hWg,\n'
        "        const float* hWu, float* hO, int M, int K, int H) {\n"
        "    size_t szA=(size_t)M*K*4, szW=(size_t)K*H*4, szO=(size_t)M*H*4;\n"
        "    float *dA=0,*dWg=0,*dWu=0,*dO=0;\n"
        "    if (cudaMalloc(&dA,szA)!=cudaSuccess) return 3;\n"
        "    if (cudaMalloc(&dWg,szW)!=cudaSuccess){cudaFree(dA);return 3;}\n"
        "    if (cudaMalloc(&dWu,szW)!=cudaSuccess){cudaFree(dA);cudaFree(dWg);return 3;}\n"
        "    if (cudaMalloc(&dO,szO)!=cudaSuccess){cudaFree(dA);cudaFree(dWg);cudaFree(dWu);return 3;}\n"
        "    cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);\n"
        "    cudaMemcpy(dWg,hWg,szW,cudaMemcpyHostToDevice);\n"
        "    cudaMemcpy(dWu,hWu,szW,cudaMemcpyHostToDevice);\n"
        "    int t=128, b=(M+t-1)/t;\n"
        f"    {_GATED_ENTRY}_kernel<<<dim3(b), dim3(t)>>>(dA,dWg,dWu,dO,M,K,H);\n"
        "    int ok = (cudaDeviceSynchronize()==cudaSuccess) ? 1 : 3;\n"
        "    if (ok==1) cudaMemcpy(hO,dO,szO,cudaMemcpyDeviceToHost);\n"
        "    cudaFree(dA); cudaFree(dWg); cudaFree(dWu); cudaFree(dO);\n"
        "    return ok;\n"
        "}\n"
    )


def _pw_cvar(vid: str) -> str:
    """A valid C identifier for a pointwise value-id."""
    return "v_" + re.sub(r"\W", "_", str(vid))


def _synthesize_pointwise_cuda(region: PointwiseGraphRegion) -> str:
    """CUDA source for a same-shape pointwise DAG (f32) — one thread per element.
    The DAG is emitted from the ``POINTWISE_OPS`` C-expression table (topo order),
    with device shims for the ops the table names but whose CUDA builtins differ
    from numpy: ``sign``/``clamp`` (undefined in CUDA) and ``max``/``min`` (CUDA's
    suppress NaN; numpy's ``np.maximum``/``np.minimum`` propagate it). All preserve
    NaN so a DAG on NaN-containing data agrees with the reference. One kernel per
    region (the DAG + input count are baked in)."""
    n = len(region.inputs)
    params = ", ".join(f"const float* i{j}" for j in range(n))
    loads = "".join(f"    float {_pw_cvar(v)} = i{j}[idx];\n"
                    for j, v in enumerate(region.inputs))
    body = ""
    for key, ins, out in region.ops:
        _arity, expr, _ref = POINTWISE_OPS[key]
        line = expr.format(*[_pw_cvar(i) for i in ins])
        # Route the table's bare max()/min() through NaN-propagating shims (numpy
        # semantics); word-boundary so fmaxf/fminf etc. are untouched.
        line = re.sub(r"\bmax\(", "tsr_max(", re.sub(r"\bmin\(", "tsr_min(", line))
        body += f"    float {_pw_cvar(out)} = {line};\n"
    hparams = ", ".join(f"const float* hi{j}" for j in range(n))
    # Free every already-allocated input buffer if a later cudaMalloc fails, so a
    # partial-allocation failure under memory pressure does not leak device memory
    # in the long-lived process (PR #297 review).
    alloc_lines = []
    for j in range(n):
        prior = " ".join(f"cudaFree(d{p});" for p in range(j))
        fail = f"{{ {prior} return 3; }}" if prior else "return 3;"
        alloc_lines.append(
            f"    float* d{j}=0; if (cudaMalloc(&d{j},sz)!=cudaSuccess) {fail}\n"
            f"    cudaMemcpy(d{j},hi{j},sz,cudaMemcpyHostToDevice);\n")
    allocs = "".join(alloc_lines)
    all_free = " ".join(f"cudaFree(d{j});" for j in range(n))
    dargs = ", ".join(f"d{j}" for j in range(n))
    return (
        "#include <cuda_runtime.h>\n"
        "#include <math.h>\n"
        "__device__ __forceinline__ float sign(float x){ return isnan(x) ? x : (float)((x>0.0f)-(x<0.0f)); }\n"
        "__device__ __forceinline__ float clamp(float x, float lo, float hi){ return isnan(x) ? x : fminf(fmaxf(x,lo),hi); }\n"
        "__device__ __forceinline__ float tsr_max(float a, float b){ return (isnan(a)||isnan(b)) ? NAN : fmaxf(a,b); }\n"
        "__device__ __forceinline__ float tsr_min(float a, float b){ return (isnan(a)||isnan(b)) ? NAN : fminf(a,b); }\n"
        f"__global__ void {_PW_ENTRY}_kernel({params}, float* out, long numel) {{\n"
        "    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (idx >= numel) return;\n"
        f"{loads}{body}"
        f"    out[idx] = {_pw_cvar(region.output)};\n"
        "}\n"
        f'extern "C" int {_PW_ENTRY}({hparams}, float* hout, long numel) {{\n'
        "    size_t sz=(size_t)numel*4;\n"
        f"{allocs}"
        f"    float* dout=0; if (cudaMalloc(&dout,sz)!=cudaSuccess) {{ {all_free} return 3; }}\n"
        "    int t=256; long b=(numel+t-1)/t;\n"
        f"    {_PW_ENTRY}_kernel<<<dim3((unsigned)b), dim3(t)>>>({dargs}, dout, numel);\n"
        "    int ok = (cudaDeviceSynchronize()==cudaSuccess) ? 1 : 3;\n"
        "    if (ok==1) cudaMemcpy(hout,dout,sz,cudaMemcpyDeviceToHost);\n"
        f"    {all_free} cudaFree(dout);\n"
        "    return ok;\n"
        "}\n"
    )


class NvidiaCudaEmitter(KernelEmitter):
    target = _TARGET
    lang = _LANG

    def can_emit(self, region: Any) -> bool:
        return isinstance(region, (FusedRegion, AttentionRegion,
                                   GatedMatmulRegion, PointwiseGraphRegion))

    def emit(self, region: Any, *, spec: SpecPolicy = SpecPolicy.BUCKET,
             dtype: str = "f32", dims: tuple[int, ...] | None = None) -> KernelSource:
        if not self.can_emit(region):
            raise EmitError(
                f"NvidiaCudaEmitter cannot emit a region of type "
                f"{type(region).__name__} (FusedRegion / AttentionRegion / "
                "GatedMatmulRegion / PointwiseGraphRegion; the shipped mma.sync "
                "GEMM lane serves single matmuls via the jit nvidia_mma executor)")
        if spec is SpecPolicy.DYNAMIC:
            raise EmitError("NvidiaCudaEmitter does not yet support SpecPolicy.DYNAMIC "
                            "(bucket/static only)")
        if dtype != "f32":
            raise EmitError(f"NvidiaCudaEmitter only supports f32 so far, got {dtype!r}")
        if isinstance(region, AttentionRegion):
            source, entry = _synthesize_attention_cuda(), _ATTN_ENTRY
        elif isinstance(region, GatedMatmulRegion):
            source, entry = _synthesize_gated_cuda(region), _GATED_ENTRY
        elif isinstance(region, PointwiseGraphRegion):
            source, entry = _synthesize_pointwise_cuda(region), _PW_ENTRY
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


def _load_lib(artifact: str):
    """dlopen ``artifact`` (cached) and return the raw handle — callers bind the
    entry symbol + argtypes for their own ABI."""
    lib = _LIB_CACHE.get(artifact)
    if lib is None:
        lib = ctypes.CDLL(artifact)
        _LIB_CACHE[artifact] = lib
    return lib


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
        # C5: the SwiGLU gate lane O = gate_act(A@Wg) * (A@Wu), one row per thread.
        import numpy as np
        try:
            Af = np.ascontiguousarray(A, np.float32)
            Wgf = np.ascontiguousarray(Wg, np.float32)
            Wuf = np.ascontiguousarray(Wu, np.float32)
            M, K = Af.shape
            Kg, H = Wgf.shape
            if Kg != K or Wuf.shape != (K, H):
                return region.reference(A, Wg, Wu), "reference"
            compiled = build(region, _TARGET, dtype="f32", dims=None)
            fn = getattr(_load_lib(compiled.artifact), _GATED_ENTRY)
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
            out = np.zeros((M, H), np.float32)
            rc = fn(_ptr(Af), _ptr(Wgf), _ptr(Wuf), _ptr(out), M, K, H)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
        return region.reference(A, Wg, Wu), "reference"

    def run_pointwise_graph(self, region: Any, arrays: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        # C5: the same-shape pointwise-DAG lane, one thread per element.
        import numpy as np
        try:
            ins = [np.ascontiguousarray(x, np.float32) for x in arrays]
            if len(ins) != len(region.inputs) or not ins:
                return region.reference(*arrays), "reference"
            shape = ins[0].shape
            if any(x.shape != shape for x in ins):
                return region.reference(*arrays), "reference"
            numel = int(np.prod(shape)) if shape else 1
            compiled = build(region, _TARGET, dtype="f32", dims=None)
            fn = getattr(_load_lib(compiled.artifact), _PW_ENTRY)
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p] * len(ins) + [ctypes.c_void_p, ctypes.c_long]
            out = np.zeros(shape, np.float32)
            rc = fn(*[_ptr(x) for x in ins], _ptr(out), numel)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
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


class NvidiaGatedCandidate(Candidate):
    """Tier-1 (C5): the synthesized SwiGLU-gate CUDA lane
    (``O = gate_act(A·Wg) ⊙ (A·Wu)``). Serves any ``GatedMatmulRegion``."""

    name = "nvidia_gated"
    tier = Tier.SYNTHESIZED
    target = _TARGET
    op = OP_GATED_MATMUL

    def run(self, region: Any, A: Any, Wg: Any, Wu: Any,
            *a: Any, **k: Any) -> tuple[Any, str]:
        return _SHARED_RUNNER.run_gated_matmul_region(region, A, Wg, Wu)


class NvidiaPointwiseCandidate(Candidate):
    """Tier-1 (C5): the synthesized same-shape pointwise-DAG CUDA lane (one thread
    per element). Serves any ``PointwiseGraphRegion``."""

    name = "nvidia_pointwise"
    tier = Tier.SYNTHESIZED
    target = _TARGET
    op = OP_POINTWISE

    def run(self, region: Any, arrays: Any, *a: Any, **k: Any) -> tuple[Any, str]:
        return _SHARED_RUNNER.run_pointwise_graph(region, arrays)


# ── tensor-core FUSED lane — mma.sync GEMM + bias/activation epilogue (Tier-2) ─
#
# The perf follow-on to the Tier-1 scalar generic lane for the fusable middle
# ground: a warp-tiled `mma.sync.m16n8k16` GEMM (bf16 operands, f32 accumulate)
# with a fused bias + single-activation epilogue at the store — the MLP hot path.
# bf16 storage → declares the f16 accuracy budget the oracle honors (Decision #28);
# arch alignment/boundary handling mirrors the shipped `libtessera_nvidia_gemm`
# tiled kernel. Serves the FusedRegion subset it can fuse (bias?, one activation,
# no reduction/residual/prologue); everything else stays on the scalar lane.

_MMA_FUSED_ENTRY = "tessera_nvidia_mma_fused"
#: Activations the mma.sync fused epilogue applies after the (optional) bias add.
_MMA_FUSED_ACTS = ("relu", "gelu", "silu", "sigmoid", "tanh")
_mma_fused_fn_cache: dict[tuple[bool, str | None], Any] = {}


def _mma_fused_epilogue(region: Any) -> tuple[bool, str | None] | None:
    """Map ``region`` to the mma.sync fused kernel's ``(has_bias, activation)``
    epilogue, or ``None`` when it is not representable. Representable iff a
    ``FusedRegion`` with no reduction/residual/prologue and an epilogue that is an
    ordered subsequence of ``[bias?, <one of _MMA_FUSED_ACTS>?]`` (bias-add then a
    single pointwise activation before the store)."""
    if not isinstance(region, FusedRegion):
        return None
    if region.reduction is not None or region.residual or region.prologue:
        return None
    epi = list(region.epilogue)
    has_bias = False
    if epi and epi[0] == "bias":
        has_bias, epi = True, epi[1:]
    if not epi:
        return has_bias, None
    if len(epi) == 1 and epi[0] in _MMA_FUSED_ACTS:
        return has_bias, epi[0]
    return None                               # bias-after-act / unfusable op


def _synthesize_mma_fused_cuda(has_bias: bool, act: str | None) -> str:
    """CUDA source for a warp-tiled ``mma.sync.m16n8k16`` f16 GEMM (row-major A/B,
    f32 accumulate) with a fused ``bias? + activation?`` epilogue. One warp per
    16x8 output tile, K-looped, boundary-checked (same fragment layout as the
    shipped tiled kernel). f16 (10-bit mantissa) keeps the rounding vs the f32
    reference inside the 5e-3 budget where bf16 (7-bit) would not."""
    from tessera.compiler.emit._fused_scalar_body import pointwise_snippet
    bias_param = "const float* bias, " if has_bias else ""

    def epi(var: str, col: str) -> str:
        s = ""
        if has_bias:
            s += f"    if (nt+{col} < N) {var} += bias[nt+{col}];\n"
        if act:
            s += f"    {pointwise_snippet(act, var)}\n"
        return s

    epilogue = (epi("d0", "2*tig") + epi("d1", "2*tig+1")
                + epi("d2", "2*tig") + epi("d3", "2*tig+1"))
    return (
        "#include <cuda_runtime.h>\n"
        "#include <math.h>\n"
        f"extern \"C\" __global__ void {_MMA_FUSED_ENTRY}_kernel(\n"
        "    const unsigned short* A, const unsigned short* B, const float* bias,\n"
        "    float* D, int M, int N, int K) {\n"
        "  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;\n"
        "  float d0=0,d1=0,d2=0,d3=0;\n"
        "  for (int k0=0;k0<K;k0+=16){\n"
        "    auto la=[&](int r,int c)->unsigned{int rr=mt+r,cc=k0+c;\n"
        "      unsigned lo=(rr<M&&cc<K)?A[rr*K+cc]:0u, hi=(rr<M&&cc+1<K)?A[rr*K+cc+1]:0u; return (hi<<16)|lo;};\n"
        "    auto lb=[&](int r,int c)->unsigned{int rr=k0+r,cc=nt+c;\n"
        "      unsigned lo=(rr<K&&cc<N)?B[rr*N+cc]:0u, hi=(rr+1<K&&cc<N)?B[(rr+1)*N+cc]:0u; return (hi<<16)|lo;};\n"
        "    unsigned a0=la(gid,2*tig),a1=la(gid+8,2*tig),a2=la(gid,2*tig+8),a3=la(gid+8,2*tig+8);\n"
        "    unsigned b0=lb(2*tig,gid),b1=lb(2*tig+8,gid);\n"
        "    asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \"\n"
        "      \"{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\\n\"\n"
        "      :\"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3):\"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1));\n"
        "  }\n"
        f"{epilogue}"
        "  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};\n"
        "  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);\n"
        "}\n"
        f'extern "C" int {_MMA_FUSED_ENTRY}(const unsigned short* hA,\n'
        "    const unsigned short* hB, const float* hbias, float* hD,\n"
        "    int M, int N, int K) {\n"
        "  size_t szA=(size_t)M*K*2, szB=(size_t)K*N*2, szO=(size_t)M*N*4, szBias=(size_t)N*4;\n"
        "  unsigned short *dA=0,*dB=0; float *dbias=0,*dD=0;\n"
        "  if (cudaMalloc(&dA,szA)!=cudaSuccess) return 3;\n"
        "  if (cudaMalloc(&dB,szB)!=cudaSuccess){cudaFree(dA);return 3;}\n"
        "  if (cudaMalloc(&dD,szO)!=cudaSuccess){cudaFree(dA);cudaFree(dB);return 3;}\n"
        "  cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);\n"
        "  cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice);\n"
        "  if (hbias){ if (cudaMalloc(&dbias,szBias)!=cudaSuccess){cudaFree(dA);cudaFree(dB);cudaFree(dD);return 3;}\n"
        "    cudaMemcpy(dbias,hbias,szBias,cudaMemcpyHostToDevice); }\n"
        "  dim3 grid((M+15)/16,(N+7)/8), block(32);\n"
        f"  {_MMA_FUSED_ENTRY}_kernel<<<grid,block>>>(dA,dB,dbias,dD,M,N,K);\n"
        "  int ok = (cudaDeviceSynchronize()==cudaSuccess) ? 1 : 3;\n"
        "  if (ok==1) cudaMemcpy(hD,dD,szO,cudaMemcpyDeviceToHost);\n"
        "  cudaFree(dA); cudaFree(dB); cudaFree(dD); if (dbias) cudaFree(dbias);\n"
        "  return ok;\n"
        "}\n"
    )


def _mma_fused_fn(has_bias: bool, act: str | None):
    """Compile (once per epilogue signature) the mma.sync fused kernel and return
    its bound entry symbol: ``int(A bf16, B bf16, bias f32|NULL, D f32, M,N,K)``."""
    sig = (has_bias, act)
    fn = _mma_fused_fn_cache.get(sig)
    if fn is not None:
        return fn
    from tessera.compiler.emit.kernel_emitter import KernelSource
    src = KernelSource(source=_synthesize_mma_fused_cuda(has_bias, act),
                       entry=_MMA_FUSED_ENTRY, lang=_LANG)
    artifact = _nvidia_cuda_compile_fn(src)
    lib = _load_lib(artifact)
    fn = getattr(lib, _MMA_FUSED_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    _mma_fused_fn_cache[sig] = fn
    return fn


class NvidiaMmaFusedCandidate(Candidate):
    """Tier-2 (emitted): the warp-tiled ``mma.sync`` GEMM + bias/activation epilogue
    — the tensor-core perf lane for the fusable matmul-epilogue middle ground. bf16
    operands / f32 accumulate, so it declares the f16 accuracy budget; the arbiter
    prefers it over the Tier-1 scalar lane where it applies + measures faster + in
    budget. Declines (to the reference) off an NVIDIA GPU, for a region it cannot
    fuse, or when a required bias buffer is missing."""

    name = "nvidia_mma_fused"
    tier = Tier.EMITTED
    target = _TARGET
    op = OP_FUSED_REGION
    accuracy_atol = _GEMM_F16_ATOL

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return rt._nvidia_mma_runtime_available()
        except Exception:
            return False

    def applies_to(self, region: Any) -> bool:
        return _mma_fused_epilogue(region) is not None

    def run(self, region: Any, A: Any, B: Any, bias: Any = None,
            *a: Any, residual: Any = None, **k: Any) -> tuple[Any, str]:
        import numpy as np
        epi = _mma_fused_epilogue(region)
        if epi is None:
            return region.reference(A, B, bias), "reference"
        has_bias, act = epi
        if has_bias and bias is None:              # NULL-buffer guard (as scalar lane)
            return region.reference(A, B, bias), "reference"
        # Validate operands BEFORE the launch: the C ABI copies K*N*2 bytes from B
        # and N*4 from bias off A's K / B's N, so a mismatched contraction or a
        # short bias would overread. Route invalid inputs through the reference,
        # which raises (like FusedRegion.reference / the GEMM helpers) — never
        # launch on a bad buffer (PR #301 review).
        Aa, Ba = np.asarray(A), np.asarray(B)
        if (Aa.ndim != 2 or Ba.ndim != 2 or Aa.shape[1] != Ba.shape[0]
                or (has_bias and np.asarray(bias).shape != (Ba.shape[1],))):
            return region.reference(A, B, bias), "reference"
        try:
            Ab = np.ascontiguousarray(Aa, np.float16)  # f16 storage, f32 accumulate
            Bb = np.ascontiguousarray(Ba, np.float16)
            M, K = Ab.shape
            _, N = Bb.shape
            bias_arr = (np.ascontiguousarray(bias, np.float32)
                        if has_bias else None)
            out = np.zeros((M, N), np.float32)
            fn = _mma_fused_fn(has_bias, act)
            rc = fn(_ptr(Ab), _ptr(Bb), _ptr(bias_arr), _ptr(out), M, N, K)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
        return region.reference(A, B, bias), "reference"


# ── tensor-core ATTENTION lane — mma.sync flash, smem-staged softmax (Tier-2) ─
#
# The perf follow-on to the Tier-1 scalar flash lane. Two mma.sync matmuls (Q·Kᵀ
# then P·V) with the row softmax staged through shared memory — this sidesteps the
# accumulator→operand fragment shuffle (write scores to smem, softmax in smem, load
# P back as the second mma's operand). One warp per 16-query tile; f16 operands /
# f32 accumulate → f16 accuracy budget. Nk is capped by the smem the score tile
# needs (16·Nk·4 bytes, opt-in up to ~96 KB); larger Nk delegates to the scalar
# lane, which streams any length.

_MMA_ATTN_ENTRY = "tessera_nvidia_mma_attn"
#: Max KV length whose 16xNk f32 score tile fits the opt-in dynamic smem (~96 KB).
_MMA_ATTN_NK_CAP = 1536
#: f16 is only safe for the attention lane when BOTH hold: operand magnitude
#: ``amax = max|Q|,|K|,|V|`` is small (bounds the f16 V-rounding output error) AND
#: the softmax sharpness proxy ``scale·D·amax²`` is bounded (softmax(scale·Q·Kᵀ)
#: sharpens with score magnitude, amplifying f16 Q/K rounding). Larger-magnitude /
#: sharper f32 attention is delegated to the exact scalar lane rather than silently
#: degraded (PR #302 review; both bounds empirically validated to keep the abs
#: error under the 5e-3 budget across scale/D/magnitude).
_MMA_ATTN_ABS_CAP = 5.0
_MMA_ATTN_SHARPNESS_CAP = 500.0
_mma_attn_fn_cache: list[Any] = []


def _synthesize_mma_attn_cuda() -> str:
    """CUDA source for ``O = softmax(scale·Q·Kᵀ)·V`` via two ``mma.sync.m16n8k16``
    f16 matmuls with the row softmax staged in shared memory (16xNk f32 scores).
    Natural Q(M,D)/K(Nk,D)/V(Nk,Dv), row-major; boundary-checked; causal masks
    keys ``n > m``. One warp per 16-query block."""
    e = _MMA_ATTN_ENTRY
    MMA = ('asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "\n'
           '        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\\n"\n'
           '        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),'
           '"r"(a3),"r"(b0),"r"(b1));')
    return (
        "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <math.h>\n"
        f"__global__ void {e}_kernel(const unsigned short* Q, const unsigned short* K,\n"
        "    const unsigned short* V, float* O, int M, int Nk, int D, int Dv,\n"
        "    float scale, int causal) {\n"
        "  extern __shared__ float S[];\n"
        "  int mt=blockIdx.x*16, lane=threadIdx.x, gid=lane>>2, tig=lane&3;\n"
        "  for (int nt=0; nt<Nk; nt+=8) {\n"
        "    float d0=0,d1=0,d2=0,d3=0;\n"
        "    for (int k0=0;k0<D;k0+=16){\n"
        "      auto lq=[&](int r,int c)->unsigned{int rr=mt+r,cc=k0+c;\n"
        "        unsigned lo=(rr<M&&cc<D)?Q[rr*D+cc]:0u, hi=(rr<M&&cc+1<D)?Q[rr*D+cc+1]:0u; return (hi<<16)|lo;};\n"
        "      auto lk=[&](int r,int c)->unsigned{int rr=k0+r,cc=nt+c;\n"
        "        unsigned lo=(cc<Nk&&rr<D)?K[cc*D+rr]:0u, hi=(cc<Nk&&rr+1<D)?K[cc*D+rr+1]:0u; return (hi<<16)|lo;};\n"
        "      unsigned a0=lq(gid,2*tig),a1=lq(gid+8,2*tig),a2=lq(gid,2*tig+8),a3=lq(gid+8,2*tig+8);\n"
        "      unsigned b0=lk(2*tig,gid),b1=lk(2*tig+8,gid);\n"
        f"      {MMA}\n"
        "    }\n"
        "    auto sst=[&](int r,int c,float v){int cc=nt+c;\n"
        "      if (r<16 && cc<Nk){ float sv=v*scale; if(causal && cc>mt+r) sv=-INFINITY; S[r*Nk+cc]=sv; }};\n"
        "    sst(gid,2*tig,d0);sst(gid,2*tig+1,d1);sst(gid+8,2*tig,d2);sst(gid+8,2*tig+1,d3);\n"
        "  }\n"
        "  __syncwarp();\n"
        "  for (int row=lane; row<16; row+=32){\n"
        "    if (mt+row>=M) continue;\n"
        "    float mx=-INFINITY; for(int n=0;n<Nk;n++) mx=fmaxf(mx,S[row*Nk+n]);\n"
        "    float sm=0; for(int n=0;n<Nk;n++){ float ex=__expf(S[row*Nk+n]-mx); S[row*Nk+n]=ex; sm+=ex; }\n"
        "    float inv=(sm>0)?1.0f/sm:0.0f; for(int n=0;n<Nk;n++) S[row*Nk+n]*=inv;\n"
        "  }\n"
        "  __syncwarp();\n"
        "  for (int nt=0; nt<Dv; nt+=8) {\n"
        "    float d0=0,d1=0,d2=0,d3=0;\n"
        "    for (int k0=0;k0<Nk;k0+=16){\n"
        "      auto lp=[&](int r,int c)->unsigned{int cc=k0+c;\n"
        "        unsigned short lo=(r<16&&cc<Nk)?__half_as_ushort(__float2half(S[r*Nk+cc])):(unsigned short)0;\n"
        "        unsigned short hi=(r<16&&cc+1<Nk)?__half_as_ushort(__float2half(S[r*Nk+cc+1])):(unsigned short)0;\n"
        "        return ((unsigned)hi<<16)|lo;};\n"
        "      auto lv=[&](int r,int c)->unsigned{int rr=k0+r,cc=nt+c;\n"
        "        unsigned lo=(rr<Nk&&cc<Dv)?V[rr*Dv+cc]:0u, hi=(rr+1<Nk&&cc<Dv)?V[(rr+1)*Dv+cc]:0u; return (hi<<16)|lo;};\n"
        "      unsigned a0=lp(gid,2*tig),a1=lp(gid+8,2*tig),a2=lp(gid,2*tig+8),a3=lp(gid+8,2*tig+8);\n"
        "      unsigned b0=lv(2*tig,gid),b1=lv(2*tig+8,gid);\n"
        f"      {MMA}\n"
        "    }\n"
        "    auto ost=[&](int r,int c,float v){int rr=mt+r,cc=nt+c; if(rr<M&&cc<Dv) O[rr*Dv+cc]=v;};\n"
        "    ost(gid,2*tig,d0);ost(gid,2*tig+1,d1);ost(gid+8,2*tig,d2);ost(gid+8,2*tig+1,d3);\n"
        "  }\n"
        "}\n"
        f'extern "C" int {e}(const unsigned short* hQ, const unsigned short* hK,\n'
        "    const unsigned short* hV, float* hO, int M, int Nk, int D, int Dv,\n"
        "    float scale, int causal) {\n"
        "  size_t szQ=(size_t)M*D*2, szK=(size_t)Nk*D*2, szV=(size_t)Nk*Dv*2, szO=(size_t)M*Dv*4;\n"
        "  int smem=16*Nk*4;\n"
        "  unsigned short *dQ=0,*dK=0,*dV=0; float* dO=0;\n"
        "  if (cudaMalloc(&dQ,szQ)!=cudaSuccess) return 3;\n"
        "  if (cudaMalloc(&dK,szK)!=cudaSuccess){cudaFree(dQ);return 3;}\n"
        "  if (cudaMalloc(&dV,szV)!=cudaSuccess){cudaFree(dQ);cudaFree(dK);return 3;}\n"
        "  if (cudaMalloc(&dO,szO)!=cudaSuccess){cudaFree(dQ);cudaFree(dK);cudaFree(dV);return 3;}\n"
        "  cudaMemcpy(dQ,hQ,szQ,cudaMemcpyHostToDevice);\n"
        "  cudaMemcpy(dK,hK,szK,cudaMemcpyHostToDevice);\n"
        "  cudaMemcpy(dV,hV,szV,cudaMemcpyHostToDevice);\n"
        f"  cudaFuncSetAttribute({e}_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n"
        f"  {e}_kernel<<<dim3((M+15)/16),dim3(32),smem>>>(dQ,dK,dV,dO,M,Nk,D,Dv,scale,causal);\n"
        "  int ok = (cudaDeviceSynchronize()==cudaSuccess) ? 1 : 3;\n"
        "  if (ok==1) cudaMemcpy(hO,dO,szO,cudaMemcpyDeviceToHost);\n"
        "  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);\n"
        "  return ok;\n"
        "}\n"
    )


def _mma_attn_fn():
    """Compile (once) the mma.sync attention kernel and return its bound entry:
    ``int(Q f16, K f16, V f16, O f32, M,Nk,D,Dv, scale, causal)``."""
    if _mma_attn_fn_cache:
        return _mma_attn_fn_cache[0]
    from tessera.compiler.emit.kernel_emitter import KernelSource
    src = KernelSource(source=_synthesize_mma_attn_cuda(), entry=_MMA_ATTN_ENTRY,
                       lang=_LANG)
    fn = getattr(_load_lib(_nvidia_cuda_compile_fn(src)), _MMA_ATTN_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
                   + [ctypes.c_float, ctypes.c_int])
    _mma_attn_fn_cache.append(fn)
    return fn


class NvidiaMmaAttnCandidate(Candidate):
    """Tier-2 (emitted): the tensor-core flash-attention lane — two ``mma.sync``
    matmuls with a smem-staged row softmax, f16 operands / f32 accumulate. Serves
    any ``AttentionRegion``; for a KV length past the smem cap it delegates to the
    scalar flash lane (which streams any length), so it never declines a shape the
    scalar lane can run. f16 accuracy budget."""

    name = "nvidia_mma_attn"
    tier = Tier.EMITTED
    target = _TARGET
    op = OP_ATTENTION
    accuracy_atol = 5e-3                        # f16 storage budget (Decision #28)

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return rt._nvidia_mma_runtime_available()
        except Exception:
            return False

    def run(self, region: Any, Q: Any, K: Any, V: Any,
            *a: Any, **k: Any) -> tuple[Any, str]:
        import numpy as np
        try:
            Qn, Kn = region._natural(Q, K)         # natural Q(M,D)/K(Nk,D), f32
            Vf = np.asarray(V, np.float32)
            M, D = Qn.shape
            Nk, Dk = Kn.shape
            Nkv, Dv = Vf.shape
            if Dk != D or Nkv != Nk:
                return region.reference(Q, K, V), "reference"
            # Delegate to the EXACT scalar lane when the score tile won't fit smem
            # OR when f16 would exceed the 5e-3 budget: softmax(scale·Q·Kᵀ) sharpens
            # with score magnitude, so f16 rounding of large-magnitude / large-scale
            # f32 attention diverges — which the fixed F4 probe doesn't exercise.
            # This keeps the lane from silently degrading default f32 semantics
            # (PR #302 review); ``scale·D·amax²`` is the validated sharpness proxy.
            amax = max(float(np.max(np.abs(Qn))), float(np.max(np.abs(Kn))),
                       float(np.max(np.abs(Vf))))
            if (Nk > _MMA_ATTN_NK_CAP or amax > _MMA_ATTN_ABS_CAP
                    or float(region.scale) * D * amax * amax > _MMA_ATTN_SHARPNESS_CAP):
                return _SHARED_RUNNER.run_fused_attention(region, Q, K, V)
            Qh = np.ascontiguousarray(Qn, np.float16)
            Kh = np.ascontiguousarray(Kn, np.float16)
            Vh = np.ascontiguousarray(Vf, np.float16)
            out = np.zeros((M, Dv), np.float32)
            rc = _mma_attn_fn()(_ptr(Qh), _ptr(Kh), _ptr(Vh), _ptr(out),
                                M, Nk, D, Dv, ctypes.c_float(float(region.scale)),
                                1 if region.causal else 0)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
        return region.reference(Q, K, V), "reference"


# ── D1 matmul candidates (B1) — bare GEMM, Tier-2 emitted vs Tier-3 shipped ────
#
# The arbiter enumerates these per (target="nvidia", op=matmul) and F4-gates each.
# Tier-priority (Decision #28) prefers the hand-tuned shipped lane by default; D2's
# measured loop lets the emitted lane win where it is faster + in accuracy budget.
# Both are 16-bit storage (bf16/f16) → f32 accumulate, so they declare the f16
# budget the oracle honors. Off an NVIDIA GPU / without the built libs they decline
# to the reference and drop out of the enumeration.


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
register_candidate(NvidiaMmaFusedCandidate())         # tensor-core fused GEMM+epi
register_candidate(NvidiaFlashAttnCandidate())        # C4: synthesized attention
register_candidate(NvidiaMmaAttnCandidate())          # tensor-core flash attention
register_candidate(NvidiaGatedCandidate())            # C5: SwiGLU gate
register_candidate(NvidiaPointwiseCandidate())        # C5: pointwise DAG
# Bare-GEMM lanes: hand-tuned shipped (Tier 3) + compiler-emitted (Tier 2).
register_candidate(NvidiaMmaGemmShippedCandidate())
register_candidate(NvidiaMmaGemmEmittedCandidate())
