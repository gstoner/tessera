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
import math
import os
import shutil
import subprocess
import tempfile
import time
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
_PAGED_ATTN_ENTRY = "tessera_rocm_paged_attention_f32"
_paged_attn_artifact: str | None = None


def _synthesize_paged_kv_read_hip() -> str:
    """HIP gather for the stable PLHD pages + logical-page-table ABI."""
    return f'''#include <hip/hip_runtime.h>
__global__ void paged_read(const float*pages,const int*table,const long long*idx,float*out,int page_size,int H,int D,long long T){{long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=T*H*D;if(z>=n)return;int d=z%D,h=(z/D)%H;long long t=z/(D*H),tok=idx[t];int lp=(int)(tok/page_size),off=(int)(tok%page_size),pp=table[lp];out[z]=pages[(((long long)pp*page_size+off)*H+h)*D+d];}}
extern "C" int {_PAGED_KV_ENTRY}(const float*hp,const int*ht,const long long*hi,float*ho,int P,int LP,int page_size,int H,int D,long long T,int reps,float*device_ms){{if(!hp||!ht||!hi||!ho||!device_ms||P<1||LP<1||page_size<1||H<1||D<1||T<1||reps<1)return 2;size_t pb=(size_t)P*page_size*H*D*4,tb=(size_t)LP*4,ib=(size_t)T*8,ob=(size_t)T*H*D*4;float *p=0,*o=0;int*t=0;long long*i=0;hipEvent_t a=0,b=0;if(hipMalloc(&p,pb)!=hipSuccess||hipMalloc(&t,tb)!=hipSuccess||hipMalloc(&i,ib)!=hipSuccess||hipMalloc(&o,ob)!=hipSuccess)return 3;if(hipMemcpy(p,hp,pb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(t,ht,tb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(i,hi,ib,hipMemcpyHostToDevice)!=hipSuccess)return 3;long long n=T*H*D;hipLaunchKernelGGL(paged_read,dim3((unsigned)((n+255)/256)),dim3(256),0,0,p,t,i,o,page_size,H,D,T);if(hipDeviceSynchronize()!=hipSuccess||hipEventCreate(&a)!=hipSuccess||hipEventCreate(&b)!=hipSuccess)return 3;hipEventRecord(a,0);for(int r=0;r<reps;r++)hipLaunchKernelGGL(paged_read,dim3((unsigned)((n+255)/256)),dim3(256),0,0,p,t,i,o,page_size,H,D,T);hipEventRecord(b,0);hipEventSynchronize(b);float ms=0;int ok=hipEventElapsedTime(&ms,a,b)==hipSuccess&&hipMemcpy(ho,o,ob,hipMemcpyDeviceToHost)==hipSuccess;*device_ms=ms/reps;hipEventDestroy(a);hipEventDestroy(b);hipFree(p);hipFree(t);hipFree(i);hipFree(o);return ok?1:3;}}'''


def run_paged_kv_cache_read_f32(
    pages: Any, page_table: Any, token_indices: Any, *, return_device_ms: bool = False,
    reps: int = 1,
) -> Any:
    """Gather arbitrary logical tokens from stable physical f32 pages on ROCm.

    The optional device-event measurement is ``None`` when the HIP runtime
    reports a zero, negative, or non-finite interval. Some WSL ROCm runtimes
    execute correctly while exposing no usable HIP event timer; callers must
    not turn that absence into fabricated positive device evidence.
    """
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
                   + [ctypes.c_longlong, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_float)])
    P, page_size, H, D = (int(x) for x in p.shape)
    out = np.empty((idx.size, H, D), np.float32)
    device_ms = ctypes.c_float()
    rc = fn(_ptr(p), _ptr(table), _ptr(idx), _ptr(out), P, int(table.size),
            page_size, H, D, int(idx.size), int(reps), ctypes.byref(device_ms))
    if rc != 1:
        raise RuntimeError(f"ROCm paged KV read launch failed (rc={rc})")
    raw_ms = float(device_ms.value)
    measured_ms: float | None = raw_ms if math.isfinite(raw_ms) and raw_ms > 0 else None
    return (out, measured_ms) if return_device_ms else out


def _synthesize_paged_attention_direct_hip() -> str:
    """Correctness-first direct PLHD paged attention, mirroring CUDA's ABI.

    Query heads may be grouped over fewer KV heads (GQA/MQA). Causal masking is
    by the supplied token *order*, with the decode offset ``T-Q``; logical token
    values are used only for page-table addressing.
    """
    return f'''#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
__global__ void paged_attn(const float*q,const float*kp,const float*vp,const int*table,const long long*idx,float*out,int T,int L,int HQ,int HKV,int D,int Q,float scale,int causal){{int qi=blockIdx.x%Q,qh=blockIdx.x/Q,t=threadIdx.x,ratio=HQ/HKV,kh=qh/ratio;extern __shared__ float scores[];__shared__ float red[256];int limit=qi+(T>Q?T-Q:0);for(int j=0;j<T;j++){{float x=0.f;if(!causal||j<=limit){{long long tok=idx[j];int pp=table[tok/L],off=tok%L;const float*k=kp+(((long long)pp*L+off)*HKV+kh)*D;const float*qr=q+((long long)qh*Q+qi)*D;for(int d=t;d<D;d+=256)x+=qr[d]*k[d];}}red[t]=x;__syncthreads();for(int s=128;s;s>>=1){{if(t<s)red[t]+=red[t+s];__syncthreads();}}if(t==0)scores[j]=(causal&&j>limit)?-INFINITY:red[0]*scale;__syncthreads();}}float m=-FLT_MAX;for(int j=t;j<T;j+=256)m=fmaxf(m,scores[j]);red[t]=m;__syncthreads();for(int s=128;s;s>>=1){{if(t<s)red[t]=fmaxf(red[t],red[t+s]);__syncthreads();}}m=red[0];float z=0.f;for(int j=t;j<T;j+=256)z+=expf(scores[j]-m);red[t]=z;__syncthreads();for(int s=128;s;s>>=1){{if(t<s)red[t]+=red[t+s];__syncthreads();}}z=red[0];for(int d=t;d<D;d+=256){{float acc=0.f;for(int j=0;j<T;j++){{long long tok=idx[j];int pp=table[tok/L],off=tok%L;const float*v=vp+(((long long)pp*L+off)*HKV+kh)*D;acc+=expf(scores[j]-m)/z*v[d];}}out[((long long)qh*Q+qi)*D+d]=acc;}}}}
extern "C" int {_PAGED_ATTN_ENTRY}(const float*hq,const float*hkp,const float*hvp,const int*ht,const long long*hi,float*ho,int P,int LP,int L,int HQ,int HKV,int D,int Q,int T,float scale,int causal,int reps,float*device_ms){{if(!hq||!hkp||!hvp||!ht||!hi||!ho||!device_ms||P<1||LP<1||L<1||HQ<1||HKV<1||HQ%HKV||D<1||Q<1||T<1||T>8192||reps<1)return 2;size_t qb=(size_t)HQ*Q*D*4,pb=(size_t)P*L*HKV*D*4,tb=(size_t)LP*4,ib=(size_t)T*8,ob=qb;float *q=0,*kp=0,*vp=0,*o=0;int*table=0;long long*idx=0;hipEvent_t a=0,b=0;if(hipMalloc(&q,qb)!=hipSuccess||hipMalloc(&kp,pb)!=hipSuccess||hipMalloc(&vp,pb)!=hipSuccess||hipMalloc(&table,tb)!=hipSuccess||hipMalloc(&idx,ib)!=hipSuccess||hipMalloc(&o,ob)!=hipSuccess)return 3;if(hipMemcpy(q,hq,qb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(kp,hkp,pb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(vp,hvp,pb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(table,ht,tb,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(idx,hi,ib,hipMemcpyHostToDevice)!=hipSuccess)return 3;paged_attn<<<HQ*Q,256,(size_t)T*4>>>(q,kp,vp,table,idx,o,T,L,HQ,HKV,D,Q,scale,causal);if(hipDeviceSynchronize()!=hipSuccess||hipEventCreate(&a)!=hipSuccess||hipEventCreate(&b)!=hipSuccess)return 3;hipEventRecord(a,0);for(int r=0;r<reps;r++)paged_attn<<<HQ*Q,256,(size_t)T*4>>>(q,kp,vp,table,idx,o,T,L,HQ,HKV,D,Q,scale,causal);hipEventRecord(b,0);hipEventSynchronize(b);float ms=0;int ok=hipEventElapsedTime(&ms,a,b)==hipSuccess&&hipMemcpy(ho,o,ob,hipMemcpyDeviceToHost)==hipSuccess;*device_ms=ms/reps;hipEventDestroy(a);hipEventDestroy(b);hipFree(q);hipFree(kp);hipFree(vp);hipFree(table);hipFree(idx);hipFree(o);return ok?1:3;}}'''


def run_paged_attention_direct_f32(
    q: Any, k_pages: Any, v_pages: Any, page_table: Any, token_indices: Any,
    *, scale: float, causal: bool, reps: int = 20,
) -> tuple[Any, float | None, float]:
    """Run direct page-table attention; return output, device-event ms, wall ms.

    ``device-event ms`` is ``None`` when HIP event timing is unavailable or
    invalid. The end-to-end wall interval remains independently usable.
    """
    import numpy as np
    qq = np.ascontiguousarray(q, np.float32)
    kp = np.ascontiguousarray(k_pages, np.float32)
    vp = np.ascontiguousarray(v_pages, np.float32)
    table = np.ascontiguousarray(page_table, np.int32)
    idx = np.ascontiguousarray(token_indices, np.int64).reshape(-1)
    if qq.ndim != 3 or kp.ndim != 4 or kp.shape != vp.shape:
        raise ValueError("ROCm direct paged attention requires Q[HQ,Q,D], pages[P,L,HKV,D]")
    P, L, HKV, D = (int(x) for x in kp.shape)
    HQ, Q, QD = (int(x) for x in qq.shape)
    if QD != D or HQ % HKV or idx.size < 1 or idx.size > 8192:
        raise ValueError("ROCm direct paged attention geometry is unsupported")
    if (table.ndim != 1 or np.any(table < 0) or np.any(table >= P)
            or np.any(idx < 0) or np.any(idx >= table.size * L)):
        raise ValueError("ROCm direct paged attention table/index is invalid")
    global _paged_attn_artifact
    if _paged_attn_artifact is None:
        _paged_attn_artifact = _rocm_hip_compile_fn(KernelSource(
            source=_synthesize_paged_attention_direct_hip(),
            entry=_PAGED_ATTN_ENTRY, lang=_LANG, spec=SpecPolicy.DYNAMIC,
            shape_key=("paged-attention-direct-v1",)))
    fn = getattr(ctypes.CDLL(_paged_attn_artifact), _PAGED_ATTN_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 8
                   + [ctypes.c_float, ctypes.c_int, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_float)])
    out = np.empty_like(qq)
    device_ms = ctypes.c_float()
    start = time.perf_counter()
    rc = fn(_ptr(qq), _ptr(kp), _ptr(vp), _ptr(table), _ptr(idx), _ptr(out),
            P, int(table.size), L, HQ, HKV, D, Q, int(idx.size),
            float(scale), int(causal), int(reps), ctypes.byref(device_ms))
    wall_ms = (time.perf_counter() - start) * 1e3
    if rc != 1:
        raise RuntimeError(f"ROCm direct paged attention launch failed (rc={rc})")
    raw_ms = float(device_ms.value)
    measured_ms: float | None = raw_ms if math.isfinite(raw_ms) and raw_ms > 0 else None
    return out, measured_ms, wall_ms


# ── Persistent ReplaySSM serving context ───────────────────────────────────

_ssm_replay_device_artifact: str | None = None


def _synthesize_ssm_replay_device_hip() -> str:
    """HIP-owned scalar-A checkpoint, replay inputs, and async output ring."""
    return r'''#include <hip/hip_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
struct Slot{float*pd,*px,*pb,*pc,*py,*dy;hipEvent_t beg,done;int state,tokens;};
struct Ctx{float*d,*x,*b,*s0,*c,*a,*y,*gram;hipStream_t stream;Slot*slots;int B,D,N,L,nslots,next;};
__global__ void replay_gram(Ctx q,int M){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=M*q.B)return;int i=z/q.B,bi=z%q.B;float value=0.f;for(int n=0;n<q.N;n++)value+=q.c[(long long)bi*q.N+n]*q.b[((long long)i*q.B+bi)*q.N+n];q.gram[z]=value;}
__global__ void replay_out(Ctx q,int M){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=q.B*q.D)return;int bi=z/q.D,di=z%q.D;float total=0.f,value=0.f,prefix=0.f;for(int i=0;i<M;i++)total+=q.d[((long long)i*q.B+bi)*q.D+di]*q.a[di];for(int n=0;n<q.N;n++)value+=q.c[(long long)bi*q.N+n]*q.s0[((long long)bi*q.D+di)*q.N+n];value*=expf(total);for(int i=0;i<M;i++){long long k=((long long)i*q.B+bi)*q.D+di;prefix+=q.d[k]*q.a[di];value+=expf(total-prefix)*q.d[k]*q.x[k]*q.gram[(long long)i*q.B+bi];}q.y[(long long)bi*q.D+di]=value;}
__global__ void replay_flush(Ctx q,int M){long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x,total=(long long)q.B*q.D*q.N;if(z>=total)return;int ni=z%q.N,di=(z/q.N)%q.D,bi=z/(q.N*q.D);float value=q.s0[z];for(int i=0;i<M;i++){long long k=((long long)i*q.B+bi)*q.D+di;value=expf(q.d[k]*q.a[di])*value+q.d[k]*q.x[k]*q.b[((long long)i*q.B+bi)*q.N+ni];}q.s0[z]=value;}
extern "C" int cr(void**out,const float*s0,const float*a,int B,int D,int N,int L,int nslots){if(!out||!s0||!a||B<1||D<1||N<1||L<1||nslots<2)return 2;Ctx*q=(Ctx*)calloc(1,sizeof(Ctx));if(!q)return 3;q->B=B;q->D=D;q->N=N;q->L=L;q->nslots=nslots;q->slots=(Slot*)calloc(nslots,sizeof(Slot));size_t bd=(size_t)L*B*D*4,bn=(size_t)L*B*N*4,ss=(size_t)B*D*N*4,cc=(size_t)B*N*4,aa=(size_t)D*4,yy=(size_t)B*D*4,gg=(size_t)L*B*4;if(!q->slots||hipMalloc(&q->d,bd)!=hipSuccess||hipMalloc(&q->x,bd)!=hipSuccess||hipMalloc(&q->b,bn)!=hipSuccess||hipMalloc(&q->s0,ss)!=hipSuccess||hipMalloc(&q->c,cc)!=hipSuccess||hipMalloc(&q->a,aa)!=hipSuccess||hipMalloc(&q->y,yy)!=hipSuccess||hipMalloc(&q->gram,gg)!=hipSuccess||hipStreamCreateWithFlags(&q->stream,hipStreamNonBlocking)!=hipSuccess||hipMemcpy(q->s0,s0,ss,hipMemcpyHostToDevice)!=hipSuccess||hipMemcpy(q->a,a,aa,hipMemcpyHostToDevice)!=hipSuccess)return 3;for(int i=0;i<nslots;i++){Slot&z=q->slots[i];if(hipMalloc(&z.dy,bd)!=hipSuccess||hipHostMalloc(&z.pd,bd,hipHostMallocDefault)!=hipSuccess||hipHostMalloc(&z.px,bd,hipHostMallocDefault)!=hipSuccess||hipHostMalloc(&z.pb,bn,hipHostMallocDefault)!=hipSuccess||hipHostMalloc(&z.pc,bn,hipHostMallocDefault)!=hipSuccess||hipHostMalloc(&z.py,bd,hipHostMallocDefault)!=hipSuccess||hipEventCreate(&z.beg)!=hipSuccess||hipEventCreate(&z.done)!=hipSuccess)return 3;}*out=q;return 1;}
extern "C" int ap(void*v,const float*d,const float*x,const float*b,int pos){Ctx*q=(Ctx*)v;if(!q||!d||!x||!b||pos<0||pos>=q->L)return 2;size_t bd=(size_t)q->B*q->D*4,bn=(size_t)q->B*q->N*4;if(hipMemcpyAsync(q->d+(size_t)pos*q->B*q->D,d,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->x+(size_t)pos*q->B*q->D,x,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->b+(size_t)pos*q->B*q->N,b,bn,hipMemcpyHostToDevice,q->stream)!=hipSuccess)return 3;return hipStreamSynchronize(q->stream)==hipSuccess?1:3;}
extern "C" int de(void*v,const float*c,float*y,int M){Ctx*q=(Ctx*)v;if(!q||!c||!y||M<1||M>q->L)return 2;size_t cb=(size_t)q->B*q->N*4,yb=(size_t)q->B*q->D*4;if(hipMemcpyAsync(q->c,c,cb,hipMemcpyHostToDevice,q->stream)!=hipSuccess)return 3;hipLaunchKernelGGL(replay_gram,dim3((M*q->B+127)/128),dim3(128),0,q->stream,*q,M);hipLaunchKernelGGL(replay_out,dim3((q->B*q->D+127)/128),dim3(128),0,q->stream,*q,M);if(hipGetLastError()!=hipSuccess||hipMemcpyAsync(y,q->y,yb,hipMemcpyDeviceToHost,q->stream)!=hipSuccess)return 3;return hipStreamSynchronize(q->stream)==hipSuccess?1:3;}
extern "C" int fu(void*v,int M){Ctx*q=(Ctx*)v;if(!q||M<1||M>q->L)return 2;long long n=(long long)q->B*q->D*q->N;hipLaunchKernelGGL(replay_flush,dim3((unsigned)((n+127)/128)),dim3(128),0,q->stream,*q,M);return hipGetLastError()==hipSuccess&&hipStreamSynchronize(q->stream)==hipSuccess?1:3;}
extern "C" int su(void*v,const float*d,const float*x,const float*b,const float*c,float*y,float*ms){Ctx*q=(Ctx*)v;if(!q||!d||!x||!b||!c||!y||!ms)return 2;size_t bd=(size_t)q->B*q->D*4,bn=(size_t)q->B*q->N*4;hipEvent_t beg=0,end=0;if(hipEventCreate(&beg)!=hipSuccess||hipEventCreate(&end)!=hipSuccess)return 3;hipEventRecord(beg,q->stream);if(hipMemcpyAsync(q->d,d,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->x,x,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->b,b,bn,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->c,c,bn,hipMemcpyHostToDevice,q->stream)!=hipSuccess)return 3;hipLaunchKernelGGL(replay_gram,dim3((q->B+127)/128),dim3(128),0,q->stream,*q,1);hipLaunchKernelGGL(replay_out,dim3((q->B*q->D+127)/128),dim3(128),0,q->stream,*q,1);long long n=(long long)q->B*q->D*q->N;hipLaunchKernelGGL(replay_flush,dim3((unsigned)((n+127)/128)),dim3(128),0,q->stream,*q,1);if(hipMemcpyAsync(y,q->y,bd,hipMemcpyDeviceToHost,q->stream)!=hipSuccess||hipEventRecord(end,q->stream)!=hipSuccess||hipEventSynchronize(end)!=hipSuccess)return 3;int ok=hipEventElapsedTime(ms,beg,end)==hipSuccess;hipEventDestroy(beg);hipEventDestroy(end);return ok?1:3;}
extern "C" int bl(void*v,const float*d,const float*x,const float*b,const float*c,float*y,int T,int start){Ctx*q=(Ctx*)v;if(!q||!d||!x||!b||!c||!y||T<1||start<0||start+T>q->L)return 2;size_t bd=(size_t)q->B*q->D*4,bn=(size_t)q->B*q->N*4,rows=(size_t)q->B*q->D;for(int i=0;i<T;i++){int p=start+i;if(hipMemcpyAsync(q->d+(size_t)p*rows,d+(size_t)i*rows,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->x+(size_t)p*rows,x+(size_t)i*rows,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->b+(size_t)p*q->B*q->N,b+(size_t)i*q->B*q->N,bn,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->c,c+(size_t)i*q->B*q->N,bn,hipMemcpyHostToDevice,q->stream)!=hipSuccess)return 3;hipLaunchKernelGGL(replay_gram,dim3(((p+1)*q->B+127)/128),dim3(128),0,q->stream,*q,p+1);hipLaunchKernelGGL(replay_out,dim3((q->B*q->D+127)/128),dim3(128),0,q->stream,*q,p+1);if(hipGetLastError()!=hipSuccess||hipMemcpyAsync(y+(size_t)i*rows,q->y,bd,hipMemcpyDeviceToHost,q->stream)!=hipSuccess)return 3;}return hipStreamSynchronize(q->stream)==hipSuccess?1:3;}
static int take(Ctx*q){for(int n=0;n<q->nslots;n++){int i=(q->next+n)%q->nslots;Slot&z=q->slots[i];if(z.state==2&&hipEventQuery(z.done)==hipSuccess)z.state=0;if(z.state==0){z.state=1;q->next=(i+1)%q->nslots;return i;}}return -1;}
extern "C" int as(void*v,const float*d,const float*x,const float*b,const float*c,int T,int start,int*slot){Ctx*q=(Ctx*)v;if(!q||!d||!x||!b||!c||!slot||T<1||start<0||start+T>q->L)return 2;int si=take(q);if(si<0)return 4;Slot&z=q->slots[si];size_t bd=(size_t)q->B*q->D*4,bn=(size_t)q->B*q->N*4,rows=(size_t)q->B*q->D;memcpy(z.pd,d,(size_t)T*bd);memcpy(z.px,x,(size_t)T*bd);memcpy(z.pb,b,(size_t)T*bn);memcpy(z.pc,c,(size_t)T*bn);if(hipEventRecord(z.beg,q->stream)!=hipSuccess)return 3;for(int i=0;i<T;i++){int p=start+i;if(hipMemcpyAsync(q->d+(size_t)p*rows,z.pd+(size_t)i*rows,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->x+(size_t)p*rows,z.px+(size_t)i*rows,bd,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->b+(size_t)p*q->B*q->N,z.pb+(size_t)i*q->B*q->N,bn,hipMemcpyHostToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(q->c,z.pc+(size_t)i*q->B*q->N,bn,hipMemcpyHostToDevice,q->stream)!=hipSuccess)return 3;hipLaunchKernelGGL(replay_gram,dim3(((p+1)*q->B+127)/128),dim3(128),0,q->stream,*q,p+1);hipLaunchKernelGGL(replay_out,dim3((q->B*q->D+127)/128),dim3(128),0,q->stream,*q,p+1);if(hipGetLastError()!=hipSuccess||hipMemcpyAsync(z.dy+(size_t)i*rows,q->y,bd,hipMemcpyDeviceToDevice,q->stream)!=hipSuccess||hipMemcpyAsync(z.py+(size_t)i*rows,q->y,bd,hipMemcpyDeviceToHost,q->stream)!=hipSuccess)return 3;}z.tokens=T;if(hipEventRecord(z.done,q->stream)!=hipSuccess)return 3;*slot=si;return 1;}
extern "C" int aw(void*v,int si,float*y,int T){Ctx*q=(Ctx*)v;if(!q||si<0||si>=q->nslots||!y)return 2;Slot&z=q->slots[si];if(z.state!=1||T!=z.tokens)return 2;if(hipEventSynchronize(z.done)!=hipSuccess)return 3;memcpy(y,z.py,(size_t)T*q->B*q->D*4);z.state=0;return 1;}
extern "C" int ew(void*v,int si){Ctx*q=(Ctx*)v;if(!q||si<0||si>=q->nslots||q->slots[si].state!=1)return 2;return hipEventSynchronize(q->slots[si].done)==hipSuccess?1:3;}
extern "C" int et(void*v,int si,float*ms){Ctx*q=(Ctx*)v;if(!q||si<0||si>=q->nslots||q->slots[si].state!=1||!ms)return 2;Slot&z=q->slots[si];if(hipEventSynchronize(z.done)!=hipSuccess)return 3;return hipEventElapsedTime(ms,z.beg,z.done)==hipSuccess?1:3;}
extern "C" int ws(void*v,int si,void*stream){Ctx*q=(Ctx*)v;if(!q||si<0||si>=q->nslots||q->slots[si].state!=1||!stream)return 2;return hipStreamWaitEvent((hipStream_t)stream,q->slots[si].done,0)==hipSuccess?1:3;}
extern "C" int rs(void*v,int si,void*stream){Ctx*q=(Ctx*)v;if(!q||si<0||si>=q->nslots||q->slots[si].state!=1)return 2;Slot&z=q->slots[si];hipStream_t s=stream?(hipStream_t)stream:q->stream;if(hipEventRecord(z.done,s)!=hipSuccess)return 3;z.state=2;return 1;}
extern "C" void* dp(void*v,int si){Ctx*q=(Ctx*)v;if(!q||si<0||si>=q->nslots||q->slots[si].state!=1)return 0;return q->slots[si].dy;}
extern "C" void* ps(void*v){Ctx*q=(Ctx*)v;return q?(void*)q->stream:0;}
extern "C" void dl(void*v){Ctx*q=(Ctx*)v;if(!q)return;hipStreamSynchronize(q->stream);hipFree(q->d);hipFree(q->x);hipFree(q->b);hipFree(q->s0);hipFree(q->c);hipFree(q->a);hipFree(q->y);hipFree(q->gram);for(int i=0;i<q->nslots;i++){Slot&z=q->slots[i];if(z.state&&z.done)hipEventSynchronize(z.done);if(z.dy)hipFree(z.dy);if(z.pd)hipHostFree(z.pd);if(z.px)hipHostFree(z.px);if(z.pb)hipHostFree(z.pb);if(z.pc)hipHostFree(z.pc);if(z.py)hipHostFree(z.py);if(z.beg)hipEventDestroy(z.beg);if(z.done)hipEventDestroy(z.done);}free(q->slots);if(q->stream)hipStreamDestroy(q->stream);free(q);}'''


class RocmReplayDeviceState:
    """Persistent HIP checkpoint/replay storage with ordered async slots."""

    def __init__(self, s0: Any, a: Any, capacity: int,
                 async_slots: int = 3) -> None:
        import numpy as np
        global _ssm_replay_device_artifact
        s0 = np.ascontiguousarray(s0, np.float32)
        a = np.ascontiguousarray(a, np.float32)
        self.B, self.D, self.N = (int(v) for v in s0.shape)
        self.capacity, self.async_slots = int(capacity), int(async_slots)
        if async_slots < 2:
            raise ValueError("ReplaySSM async ring requires at least two slots")
        if _ssm_replay_device_artifact is None:
            _ssm_replay_device_artifact = _rocm_hip_compile_fn(KernelSource(
                source=_synthesize_ssm_replay_device_hip(), entry="cr",
                lang=_LANG, spec=SpecPolicy.DYNAMIC,
                shape_key=("ssm-replay-device-v1",)))
        self.lib = ctypes.CDLL(_ssm_replay_device_artifact)
        self.ctx = ctypes.c_void_p()
        fn = self.lib.cr
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.POINTER(ctypes.c_void_p)]
                       + [ctypes.c_void_p] * 2 + [ctypes.c_int] * 5)
        if fn(ctypes.byref(self.ctx), _ptr(s0), _ptr(a), self.B, self.D,
              self.N, self.capacity, self.async_slots) != 1:
            raise RuntimeError("ReplaySSM ROCm allocation failed")

    def append(self, delta: Any, x: Any, b: Any, index: int) -> None:
        import numpy as np
        values = [np.ascontiguousarray(v, np.float32) for v in (delta, x, b)]
        fn = self.lib.ap; fn.restype = ctypes.c_int
        if fn(self.ctx, *(_ptr(v) for v in values), int(index)) != 1:
            raise RuntimeError("ReplaySSM ROCm append failed")

    def decode(self, c: Any, count: int) -> Any:
        import numpy as np
        cc = np.ascontiguousarray(c, np.float32)
        out = np.empty((self.B, self.D), np.float32)
        fn = self.lib.de; fn.restype = ctypes.c_int
        if fn(self.ctx, _ptr(cc), _ptr(out), int(count)) != 1:
            raise RuntimeError("ReplaySSM ROCm decode failed")
        return out

    def flush(self, count: int) -> None:
        fn = self.lib.fu; fn.restype = ctypes.c_int
        if fn(self.ctx, int(count)) != 1:
            raise RuntimeError("ReplaySSM ROCm flush failed")

    def summary_step(self, delta: Any, x: Any, b: Any, c: Any) -> tuple[Any, float]:
        """Eager summary baseline: output then materialize resident S0."""
        import numpy as np
        delta, x, b, c = (np.ascontiguousarray(v, np.float32)
                          for v in (delta, x, b, c))
        out = np.empty((self.B, self.D), np.float32)
        elapsed = ctypes.c_float()
        fn = self.lib.su; fn.restype = ctypes.c_int
        if fn(self.ctx, _ptr(delta), _ptr(x), _ptr(b), _ptr(c), _ptr(out),
              ctypes.byref(elapsed)) != 1:
            raise RuntimeError("ReplaySSM ROCm summary step failed")
        return out, float(elapsed.value)

    def submit_block(self, delta: Any, x: Any, b: Any, c: Any,
                     start: int) -> Any:
        import numpy as np
        delta, x, b, c = (np.ascontiguousarray(v, np.float32)
                          for v in (delta, x, b, c))
        tokens = int(delta.shape[0])
        out = np.empty((tokens, self.B, self.D), np.float32)
        fn = self.lib.bl; fn.restype = ctypes.c_int
        if fn(self.ctx, _ptr(delta), _ptr(x), _ptr(b), _ptr(c), _ptr(out),
              tokens, int(start)) != 1:
            raise RuntimeError("ReplaySSM ROCm block submit failed")
        return out

    def submit_block_async(self, delta: Any, x: Any, b: Any, c: Any,
                           start: int) -> "RocmReplayAsyncResult":
        import numpy as np
        delta, x, b, c = (np.ascontiguousarray(v, np.float32)
                          for v in (delta, x, b, c))
        tokens = int(delta.shape[0])
        slot = ctypes.c_int(-1)
        fn = getattr(self.lib, "as"); fn.restype = ctypes.c_int
        rc = fn(self.ctx, _ptr(delta), _ptr(x), _ptr(b), _ptr(c), tokens,
                int(start), ctypes.byref(slot))
        if rc == 4:
            raise RuntimeError("ReplaySSM ROCm async ring is full")
        if rc != 1:
            raise RuntimeError("ReplaySSM ROCm async submit failed")
        return RocmReplayAsyncResult(self, tokens, int(slot.value))

    def close(self) -> None:
        if getattr(self, "ctx", None) is not None and self.ctx.value:
            self.lib.dl(self.ctx)
            self.ctx = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RocmEvent:
    """Opaque HIP completion event for a leased ReplaySSM result."""

    def __init__(self, result: "RocmReplayAsyncResult") -> None:
        self._result = result

    def wait(self) -> None:
        self._result._wait_event()

    def wait_on(self, stream: int) -> None:
        self._result._wait_on_stream(stream)

    def elapsed_ms(self) -> float:
        return self._result._elapsed_ms()


class RocmDeviceBuffer:
    """Opaque HIP device-output lease, valid until wait/release consumes it."""

    def __init__(self, result: "RocmReplayAsyncResult",
                 shape: tuple[int, ...]) -> None:
        self._result, self.shape, self.dtype = result, shape, "float32"

    def numpy(self) -> Any:
        return self._result.wait()

    @property
    def __hip_array_interface__(self) -> dict[str, Any]:
        return {"shape": self.shape, "strides": None, "typestr": "<f4",
                "data": (self._result._device_pointer(), False), "version": 1,
                "stream": self._result._producer_stream()}


class RocmReplayAsyncResult:
    """One in-flight HIP ReplaySSM submission and its output lease."""

    def __init__(self, state: RocmReplayDeviceState, tokens: int,
                 slot: int) -> None:
        self._state, self._tokens, self._slot = state, tokens, slot
        self._done = False
        self.event = RocmEvent(self)
        self.device_buffer = RocmDeviceBuffer(
            self, (tokens, state.B, state.D))

    def _guard(self) -> None:
        if self._done:
            raise RuntimeError("ReplaySSM ROCm async result already consumed")

    def wait(self) -> Any:
        import numpy as np
        self._guard()
        out = np.empty((self._tokens, self._state.B, self._state.D), np.float32)
        fn = self._state.lib.aw; fn.restype = ctypes.c_int
        if fn(self._state.ctx, self._slot, _ptr(out), self._tokens) != 1:
            raise RuntimeError("ReplaySSM ROCm async wait failed")
        self._done = True
        return out

    def _wait_event(self) -> None:
        self._guard(); fn = self._state.lib.ew; fn.restype = ctypes.c_int
        if fn(self._state.ctx, self._slot) != 1:
            raise RuntimeError("ReplaySSM ROCm event wait failed")

    def _elapsed_ms(self) -> float:
        self._guard(); fn = self._state.lib.et; fn.restype = ctypes.c_int
        elapsed = ctypes.c_float()
        if fn(self._state.ctx, self._slot, ctypes.byref(elapsed)) != 1:
            raise RuntimeError("ReplaySSM ROCm event timing failed")
        return float(elapsed.value)

    def _wait_on_stream(self, stream: int) -> None:
        self._guard()
        if not isinstance(stream, int) or stream == 0:
            raise ValueError("HIP stream handle must be a nonzero integer")
        fn = self._state.lib.ws; fn.restype = ctypes.c_int
        if fn(self._state.ctx, self._slot, ctypes.c_void_p(stream)) != 1:
            raise RuntimeError("ReplaySSM ROCm stream wait failed")

    def release(self, *, stream: int | None = None) -> None:
        self._guard()
        if stream is not None and (not isinstance(stream, int) or stream == 0):
            raise ValueError("HIP stream handle must be a nonzero integer")
        fn = self._state.lib.rs; fn.restype = ctypes.c_int
        if fn(self._state.ctx, self._slot,
              ctypes.c_void_p(stream or 0)) != 1:
            raise RuntimeError("ReplaySSM ROCm result release failed")
        self._done = True

    def _device_pointer(self) -> int:
        self._guard(); fn = self._state.lib.dp; fn.restype = ctypes.c_void_p
        ptr = fn(self._state.ctx, self._slot)
        if not ptr:
            raise RuntimeError("ReplaySSM ROCm device buffer unavailable")
        return int(ptr)

    def _producer_stream(self) -> int:
        fn = self._state.lib.ps; fn.restype = ctypes.c_void_p
        ptr = fn(self._state.ctx)
        if not ptr:
            raise RuntimeError("ReplaySSM ROCm producer stream unavailable")
        return int(ptr)


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
