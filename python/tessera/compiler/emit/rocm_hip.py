"""Workstream C3 — ROCm gfx1151 codegen plugin: generic synth → HIP.

Two lanes under one `target = "rocm"` plugin, both F4-gated on real silicon:

* **Generic compiled lane (C3)** — a full three seams for the fusable
  middle ground (`FusedRegion`: matmul + prologue/epilogue/residual/reduction):
  - :class:`RocmHipEmitter` (`register_emitter`) — region → HIP source (a
    ``__global__`` one-thread-per-row kernel + a host-pointer C-ABI wrapper),
    reusing the *same* scalar body as the x86 C lane
    (`_fused_scalar_body.row_compute_body`) so both stay locked to the
    `fusion_core` numpy reference.
  - :func:`_rocm_hip_compile_fn` (`register_compiler`) — `hipcc
    --offload-arch=<gfx> -O3 -shared` → a `.so` the runtime dlopens.
  - :meth:`RocmHipRunner.run_fused_region` — H2D / launch / D2H via the shipped
    lib's host-pointer ABI → `(out, "rocm_hip")`, else the reference.
* **Shipped hand-tuned lane (Tier 3)** — :meth:`RocmHipRunner.run_fused_attention`
  runs the shipped compiled FA-2 flash-attn kernel (not generically emitted); the
  same universal oracle gates it. This is the cross-backend differential-
  equivalence superpower on the lead's real kernels.

Lead-safety: the generic HIP kernel is a correctness-first candidate for the
middle ground — crown-jewel WMMA/MFMA GEMM stays first-class (the D1 arbiter
picks the generic lane only where it measures faster and in budget). Runs only
where a live gfx1151 + `hipcc` are present; everywhere else it declines to the
numpy reference so authoring/tests stay host-free.

Precision: the flash lane is f16 storage, so the runner declares an f16
`accuracy_atol` budget the oracle honors (Decision #28); the generic f32 HIP
kernel is comfortably within it.
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
    Candidate,
    Tier,
    register_candidate,
)
from tessera.compiler.emit.kernel_cache import build, register_compiler
from tessera.compiler.emit.kernel_emitter import (
    EmitError,
    KernelEmitter,
    KernelSource,
    KernelRunner,
    SpecPolicy,
    bucket_key,
    register_emitter,
    register_runner,
)
from tessera.compiler.fusion_core import FusedRegion

_TARGET = "rocm"
_LANG = "hip"
_ENTRY = "tessera_rocm_fused"
_REAL_TAG = "rocm_hip"
#: Real-execution tag for the fused WMMA matrix-core lane (distinct from the
#: generic scalar HIP lane's "rocm_hip") so the arbiter/fallback log can tell
#: which candidate actually ran.
_WMMA_TAG = "rocm_wmma"
#: f16 storage budget for the shipped flash lane vs the f32 reference. Loose
#: enough for f16 rounding (~2.5e-3 on the oracle probes), tight enough that an
#: O(1) miscompile is still caught. The generic f32 HIP kernel is well within it.
_F16_ATOL = 5e-3


# ── HIP source synthesis (generic FusedRegion lane) ───────────────────────────

def _synthesize_fused_hip(region: FusedRegion) -> str:
    """HIP source for a ``FusedRegion`` (f32): a one-thread-per-row kernel embedding
    the shared scalar body, plus a host-pointer C-ABI wrapper that does H2D /
    launch / D2H (same shape as the shipped ``libtessera_rocm_gemm.so`` symbols).
    Dims are runtime args, so one kernel serves every shape."""
    return (
        "#include <hip/hip_runtime.h>\n"
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
        "    if (hipMalloc(&dA,szA)!=hipSuccess) return 2;\n"
        "    if (hipMalloc(&dB,szB)!=hipSuccess) { hipFree(dA); return 2; }\n"
        "    if (hipMalloc(&dO,szO)!=hipSuccess) { hipFree(dA); hipFree(dB); return 2; }\n"
        "    hipMemcpy(dA,hA,szA,hipMemcpyHostToDevice);\n"
        "    hipMemcpy(dB,hB,szB,hipMemcpyHostToDevice);\n"
        "    if (hbias) { hipMalloc(&dbias,(size_t)N*sizeof(float));\n"
        "        hipMemcpy(dbias,hbias,(size_t)N*sizeof(float),hipMemcpyHostToDevice); }\n"
        "    if (hresidual) { hipMalloc(&dres,szO);\n"
        "        hipMemcpy(dres,hresidual,szO,hipMemcpyHostToDevice); }\n"
        "    int t=64, b=(M+t-1)/t;\n"
        f"    hipLaunchKernelGGL({_ENTRY}_kernel, dim3(b), dim3(t), 0, 0,\n"
        "        dA,dB,dbias,dres,dO,M,N,K);\n"
        "    int ok = (hipDeviceSynchronize()==hipSuccess) ? 1 : 3;\n"
        "    if (ok==1) hipMemcpy(hout,dO,szO,hipMemcpyDeviceToHost);\n"
        "    hipFree(dA); hipFree(dB); hipFree(dO);\n"
        "    if (dbias) hipFree(dbias);\n"
        "    if (dres) hipFree(dres);\n"
        "    return ok;\n"
        "}\n"
    )


class RocmHipEmitter(KernelEmitter):
    target = _TARGET
    lang = _LANG

    def can_emit(self, region: Any) -> bool:
        return isinstance(region, FusedRegion)

    def emit(self, region: Any, *, spec: SpecPolicy = SpecPolicy.BUCKET,
             dtype: str = "f32", dims: tuple[int, ...] | None = None) -> KernelSource:
        if not isinstance(region, FusedRegion):
            raise EmitError(
                f"RocmHipEmitter cannot emit a region of type "
                f"{type(region).__name__} (only FusedRegion; attention uses the "
                "shipped flash lane)")
        # DYNAMIC is supported: the generic HIP kernel already takes M/N/K as
        # runtime args with an in-kernel bounds guard, so the source is
        # dims-invariant — one compiled kernel serves every shape (Workstream G /
        # W2). The only difference from BUCKET is the shape_key below, which under
        # DYNAMIC is the symbolic identity, so the cache holds ONE entry across all
        # shapes instead of one per bucket.
        if dtype != "f32":
            raise EmitError(f"RocmHipEmitter only supports f32 so far, got {dtype!r}")
        source = _synthesize_fused_hip(region)
        key = bucket_key(dims, spec, dim_names=getattr(region, "dim_names", None))
        return KernelSource(source=source, entry=_ENTRY, lang=self.lang,
                            spec=spec, shape_key=key)


# ── compile_fn (HIP → .so) ────────────────────────────────────────────────────

def _rocm_arch() -> str:
    """gfx target: ``$TESSERA_ROCM_ARCH`` override, else the live device's chip,
    else gfx1151 (the Strix Halo default)."""
    env = os.environ.get("TESSERA_ROCM_ARCH")
    if env:
        return env
    try:
        from tessera import runtime as rt
        chip = rt._rocm_chip()
        if chip:
            return str(chip)
    except Exception:
        pass
    return "gfx1151"


def _rocm_hip_compile_fn(source: KernelSource) -> str:
    """Compile the emitted HIP to a shared object with hipcc and return its path.
    Raises on a missing toolchain/compile failure; ``build`` wraps in
    ``CompileError`` (never a silent no-op)."""
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    d = tempfile.mkdtemp(prefix="tessera_rocm_")
    src = os.path.join(d, "kernel.hip")
    so = os.path.join(d, "kernel.so")
    with open(src, "w") as f:
        f.write(source.source)
    subprocess.run(
        [hipcc, f"--offload-arch={_rocm_arch()}", "-O3", "-fPIC", "-shared",
         src, "-o", so],
        check=True, capture_output=True, text=True)
    return so


# ── runner (execute → (out, tag)) ─────────────────────────────────────────────

_LIB_CACHE: dict[str, Any] = {}


def _load_entry(artifact: str):
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


_PAGED_KV_ENTRY = "tessera_rocm_paged_kv_read_f32"
_paged_kv_artifact: str | None = None


def _synthesize_paged_kv_read_hip() -> str:
    """HIP gather for the stable PLHD pages + logical-page-table ABI."""
    return f'''#include <hip/hip_runtime.h>
__global__ void paged_read(const float*pages,const int*table,const long long*idx,float*out,int page_size,int H,int D,long long T){{long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=T*H*D;if(z>=n)return;int d=z%D,h=(z/D)%H;long long t=z/(D*H),tok=idx[t];int lp=(int)(tok/page_size),off=(int)(tok%page_size),pp=table[lp];out[z]=pages[(((long long)pp*page_size+off)*H+h)*D+d];}}
extern "C" int {_PAGED_KV_ENTRY}(const float*hp,const int*ht,const long long*hi,float*ho,int P,int LP,int page_size,int H,int D,long long T){{if(!hp||!ht||!hi||!ho||P<1||LP<1||page_size<1||H<1||D<1||T<1)return 2;size_t pb=(size_t)P*page_size*H*D*4,tb=(size_t)LP*4,ib=(size_t)T*8,ob=(size_t)T*H*D*4;float *p=0,*o=0;int*t=0;long long*i=0;if(hipMalloc(&p,pb)!=hipSuccess||hipMalloc(&t,tb)!=hipSuccess||hipMalloc(&i,ib)!=hipSuccess||hipMalloc(&o,ob)!=hipSuccess)return 3;if(hipMemcpy(p,hp,pb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(t,ht,tb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(i,hi,ib,hipMemcpyHostToDevice)!=hipSuccess)return 3;long long n=T*H*D;hipLaunchKernelGGL(paged_read,dim3((unsigned)((n+255)/256)),dim3(256),0,0,p,t,i,o,page_size,H,D,T);int ok=hipDeviceSynchronize()==hipSuccess&&hipMemcpy(ho,o,ob,hipMemcpyDeviceToHost)==hipSuccess;hipFree(p);hipFree(t);hipFree(i);hipFree(o);return ok?1:3;}}'''


def run_paged_kv_cache_read_f32(
    pages: Any, page_table: Any, token_indices: Any,
) -> Any:
    """Gather arbitrary logical tokens from stable physical f32 pages on ROCm."""
    import numpy as np
    p = np.ascontiguousarray(pages)
    table = np.ascontiguousarray(page_table, dtype=np.int32)
    idx = np.ascontiguousarray(token_indices, dtype=np.int64).reshape(-1)
    if p.dtype != np.float32 or p.ndim != 4:
        raise ValueError("ROCm paged KV pages must be rank-4 f32 [P,L,H,D]")
    if table.ndim != 1 or table.size < 1:
        raise ValueError("ROCm paged KV page_table must be non-empty rank-1")
    if np.any(table < 0) or np.any(table >= p.shape[0]):
        raise ValueError("ROCm paged KV page_table references an invalid physical page")
    if idx.size < 1 or np.any(idx < 0) or np.any(idx >= table.size * p.shape[1]):
        raise ValueError("ROCm paged KV token index exceeds logical table capacity")
    global _paged_kv_artifact
    if _paged_kv_artifact is None:
        _paged_kv_artifact = _rocm_hip_compile_fn(KernelSource(
            source=_synthesize_paged_kv_read_hip(), entry=_PAGED_KV_ENTRY,
            lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=("paged-kv-v1",)))
    fn = getattr(ctypes.CDLL(_paged_kv_artifact), _PAGED_KV_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
                   + [ctypes.c_longlong])
    P, page_size, H, D = (int(x) for x in p.shape)
    out = np.empty((idx.size, H, D), np.float32)
    rc = fn(_ptr(p), _ptr(table), _ptr(idx), _ptr(out), P, int(table.size),
            page_size, H, D, int(idx.size))
    if rc != 1:
        raise RuntimeError(f"ROCm paged KV read launch failed (rc={rc})")
    return out


class RocmHipRunner(KernelRunner):
    target = _TARGET
    accuracy_atol = _F16_ATOL

    def run_fused_region(self, region: Any, A: Any, B: Any, bias: Any = None,
                         *args: Any, residual: Any = None,
                         **kwargs: Any) -> tuple[Any, str]:
        import numpy as np
        # Required-buffer guard BEFORE launch: the emitted HIP dereferences
        # bias[n] / residual[...] whenever the region declares them, so a missing
        # buffer would pass a null the kernel derefs. Route ill-formed calls
        # through the reference (a clean, catchable ValueError) instead.
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

    # No single fused GPU kernel for these yet — decline to the numpy reference.
    def run_gated_matmul_region(self, region: Any, A: Any, Wg: Any, Wu: Any,
                                *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(A, Wg, Wu), "reference"

    def run_pointwise_graph(self, region: Any, arrays: Any,
                            *a: Any, **k: Any) -> tuple[Any, str]:
        return region.reference(*arrays), "reference"


# ── D1 candidates (C3 tail) ───────────────────────────────────────────────────
#
# The arbiter (emit/candidate.py) enumerates these per (target, op) and F4-gates
# each. Three ROCm lanes become first-class candidates:
#   • generic scalar HIP  — Tier 1 (synthesized), serves ANY FusedRegion.
#   • fused WMMA GEMM      — Tier 3 (hand-tuned, the `generate-wmma-gemm-kernel`
#                            `Generate*` pass), serves the bias/relu/gelu/silu
#                            middle ground on matrix cores. THIS is the C3 tail:
#                            the crown-jewel GEMM driven through the same loop.
#   • compiled FA-2 flash  — Tier 3 (hand-tuned), serves attention.
# Lead-safety (Decision #28): the default arbiter prefers the highest tier, so
# WMMA wins over the generic lane wherever it applies — until D2's measured loop
# proves the generic lane faster + in budget on a given shape-bucket.

_SHARED_RUNNER = RocmHipRunner()

#: Activations the WMMA `generate-wmma-gemm-kernel` epilogue fuses (bias is a
#: separate flag). The kernel applies bias FIRST, then one of these — so a region
#: is representable only when its epilogue is a bias-before-activation subsequence.
_WMMA_ACTS = ("relu", "gelu", "silu")


def _wmma_epilogue(region: Any) -> tuple[bool, str] | None:
    """Map ``region`` to the fused WMMA kernel's ``(has_bias, activation)`` epilogue,
    or ``None`` when the region is not representable on that kernel. Representable
    iff: a ``FusedRegion`` with no reduction / residual / prologue, and an epilogue
    that is an ordered subsequence of ``[bias?, <one of relu/gelu/silu>?]`` (the
    kernel does bias-add then a single pointwise activation before the store)."""
    if not isinstance(region, FusedRegion):
        return None
    if region.reduction is not None or region.residual or region.prologue:
        return None
    epi = list(region.epilogue)
    has_bias = False
    if epi and epi[0] == "bias":
        has_bias = True
        epi = epi[1:]
    if not epi:
        return has_bias, "none"
    if len(epi) == 1 and epi[0] in _WMMA_ACTS:
        return has_bias, epi[0]
    return None                               # bias-after-act, or an unfusable op


class RocmGenericHipCandidate(Candidate):
    """Tier-1: the generic one-thread-per-row HIP lane (arch-agnostic synth). Serves
    any ``FusedRegion`` — the floor-raising middle ground that is correctness-first,
    not a matrix-core GEMM."""

    name = "rocm_generic_hip"
    tier = Tier.SYNTHESIZED
    target = _TARGET
    op = OP_FUSED_REGION

    def run(self, region: Any, A: Any, B: Any, bias: Any = None,
            residual: Any = None, *a: Any, **k: Any) -> tuple[Any, str]:
        # residual positional-or-keyword so the arbiter's positional inputs
        # thread it (matches the A,B,bias,residual reference ABI; PR #290 review).
        return _SHARED_RUNNER.run_fused_region(region, A, B, bias,
                                               residual=residual)


class RocmWmmaGemmCandidate(Candidate):
    """Tier-3: the hand-tuned WMMA GEMM (`generate-wmma-gemm-kernel` pass) with a
    fused bias/relu/gelu/silu epilogue on the matrix cores, f16 storage / f32
    accumulate — the C3 tail's crown-jewel candidate. Declines (to the reference)
    off gfx1151 or for a region it cannot fuse, so it simply drops out of the
    arbiter's enumeration there."""

    name = "rocm_wmma_gemm"
    tier = Tier.HAND_TUNED
    target = _TARGET
    op = OP_FUSED_REGION
    accuracy_atol = _F16_ATOL              # f16 storage budget (Decision #28)

    def available(self) -> bool:
        # Probe the ACTUAL fused path (tessera-opt + generated kernel), not just
        # the shipped GEMM symbol — else this could win arbitration on a host where
        # only the shipped lib probes OK and then decline to the reference, starving
        # the working generic lane (PR #289 review).
        try:
            from tessera import runtime as rt
            return rt._rocm_wmma_fused_available()
        except Exception:
            return False

    def applies_to(self, region: Any) -> bool:
        return _wmma_epilogue(region) is not None

    def run(self, region: Any, A: Any, B: Any, bias: Any = None,
            *a: Any, **k: Any) -> tuple[Any, str]:
        import numpy as np
        epi = _wmma_epilogue(region)
        if epi is None:                    # not representable — honest decline
            return region.reference(A, B, bias), "reference"
        has_bias, activation = epi
        if has_bias and bias is None:      # NULL-buffer guard (as x86/generic)
            return region.reference(A, B, bias), "reference"
        try:
            from tessera import runtime as rt
            Ah = np.ascontiguousarray(A, np.float16)
            Bh = np.ascontiguousarray(B, np.float16)
            bias_arr = (np.ascontiguousarray(bias, np.float32)
                        if has_bias else None)
            out = rt._rocm_wmma_fused_2d(Ah, Bh, bias_arr, activation)
            return np.asarray(out, np.float32), _WMMA_TAG
        except Exception:
            return region.reference(A, B, bias), "reference"


class RocmFlashAttnCandidate(Candidate):
    """Tier-3: the shipped compiled FA-2 flash-attention lane (not generically
    emitted) — the crown-jewel attention candidate, gated by the same oracle."""

    name = "rocm_flash_attn"
    tier = Tier.HAND_TUNED
    target = _TARGET
    op = OP_ATTENTION
    accuracy_atol = _F16_ATOL

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return rt._rocm_compiled_flash_attn_available()
        except Exception:
            return False

    def run(self, region: Any, Q: Any, K: Any, V: Any,
            *a: Any, **k: Any) -> tuple[Any, str]:
        return _SHARED_RUNNER.run_fused_attention(region, Q, K, V)


# ── registration ──────────────────────────────────────────────────────────────
register_emitter(RocmHipEmitter())
register_compiler(_TARGET, _rocm_hip_compile_fn)
register_runner(RocmHipRunner(), default=False)

# D1 arbiter candidates — the generic lane and the crown-jewel WMMA/flash lanes
# side by side under one target, each independently F4-gated (C3 tail).
register_candidate(RocmGenericHipCandidate())
register_candidate(RocmWmmaGemmCandidate())
register_candidate(RocmFlashAttnCandidate())
