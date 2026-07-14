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
_SOFTMAX_ENTRY = "tessera_nvidia_softmax"
_FLASH_FWD_ENTRY = "tessera_nvidia_flash_attn_fwd"
_FLASH_BWD_ENTRY = "tessera_nvidia_flash_attn_bwd"
_LINEAR_ATTN_ENTRY = "tessera_nvidia_linear_attn"
_LINEAR_ATTN_BWD_ENTRY = "tessera_nvidia_linear_attn_bwd"
_MLA_FUSED_ENTRY = "tessera_nvidia_mla_decode_fused"
_CONTROL_FOR_ENTRY = "tessera_nvidia_control_for_f32"
_CONTROL_IF_ENTRY = "tessera_nvidia_control_if_f32"
_CONTROL_WHILE_ENTRY = "tessera_nvidia_control_while_f32"
_CONTROL_SCAN_ENTRY = "tessera_nvidia_control_scan_f32"
_ROPE_ENTRY = "tessera_nvidia_rope_f32"
_ALIBI_ENTRY = "tessera_nvidia_alibi_f32"
_SSM_ENTRY = "tessera_nvidia_selective_ssm_f32"
_SSM_REPLAY_ENTRY = "tessera_nvidia_ssm_replay_decode_f32"
_DELTANET_ENTRY = "tessera_nvidia_deltanet_f32"
_MOE_DISPATCH_ENTRY = "tessera_nvidia_moe_dispatch_f32"
_MOE_COMBINE_ENTRY = "tessera_nvidia_moe_combine_f32"
_GROUPED_GEMM_ENTRY = "tessera_nvidia_grouped_gemm_f32"
_DEQUANT_GROUPED_ENTRY = "tessera_nvidia_dequant_grouped_gemm_f32"
_OPTIMIZER_ENTRY = "tessera_nvidia_optimizer_f32"
_LOCAL_COLLECTIVE_ENTRY = "tessera_nvidia_local_collective_f32"
_FPQUANT_ENTRY = "tessera_nvidia_fpquant_f32"
_REAL_TAG = "nvidia_cuda"
#: Max head dim (Dv) the one-thread-per-query flash kernel holds in its per-thread
#: online-softmax accumulator; larger Dv declines to the reference.
_ATTN_DV_CAP = 256
#: 16-bit storage accuracy budget vs the f32 reference (Decision #28) — shared by
#: the tensor-core fused lane and the bf16/f16 GEMM candidates.
_GEMM_F16_ATOL = 5e-3
_GEMM_BF16_ATOL = 5e-2
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


# ── batched Flash Attention contract lane (CUDA parity P2) ───────────────────

_flash_fwd_artifact: dict[str, str] = {}
_flash_bwd_artifact: dict[str, str] = {}
_linear_attn_artifact: str | None = None
_linear_attn_bwd_artifact: str | None = None
_linear_attn_variant_artifact: str | None = None
_linear_attn_variant_bwd_artifact: str | None = None
_mla_fused_artifact: str | None = None
_control_flow_artifact: str | None = None
_posenc_artifact: str | None = None
_ssm_artifact: str | None = None
_ssm_replay_artifact: str | None = None
_deltanet_artifact: str | None = None
_moe_artifact: str | None = None
_dequant_artifact: str | None = None
_optimizer_artifact: str | None = None
_local_collective_artifact: str | None = None
_fpquant_artifact: str | None = None


def _synthesize_flash_fwd_cuda() -> str:
    """CUDA f32 Flash-Attention forward for MHA/GQA/MQA contract variants.

    One thread owns one ``(batch, query-head, query-position)`` row and streams
    the mapped KV head with online softmax.  This correctness-first kernel keeps
    the contract explicit: optional dense additive bias, causal/window mask and
    logit soft-cap are runtime ABI arguments, not host-side preprocessing.
    """
    return (
        "#include <cuda_runtime.h>\n#include <math.h>\n"
        f"#define TSR_FA_DV_CAP {_ATTN_DV_CAP}\n"
        "__global__ void tsr_flash_fwd(const float*q,const float*k,const float*v,"
        " const float*bias,float*o,long B,int Hq,int Hkv,long Sq,long Sk,int D,int Dv,"
        " float scale,int causal,long wl,long wr,float softcap){\n"
        " long row=(long)blockIdx.x*blockDim.x+threadIdx.x,total=B*(long)Hq*Sq; if(row>=total)return;\n"
        " long m=row%Sq,tmp=row/Sq;int qh=(int)(tmp%Hq);long b=tmp/Hq;int ratio=Hq/Hkv, hk=qh/ratio;\n"
        " float acc[TSR_FA_DV_CAP];for(int d=0;d<Dv;++d)acc[d]=0.f;float mi=-INFINITY,li=0.f;\n"
        " for(long n=0;n<Sk;++n){if((causal&&n>m)||(wl>=0&&n<m-wl)||(wr>=0&&n>m+wr))continue;\n"
        "  float s=0.f;long qo=(((b*(long)Hq+qh)*Sq+m)*D),ko=(((b*(long)Hkv+hk)*Sk+n)*D);\n"
        "  for(int d=0;d<D;++d)s+=q[qo+d]*k[ko+d];s*=scale;\n"
        "  if(bias)s+=bias[(((b*(long)Hq+qh)*Sq+m)*Sk+n)];if(softcap>0.f)s=softcap*tanhf(s/softcap);\n"
        "  float mn=fmaxf(mi,s),corr=expf(mi-mn),p=expf(s-mn);li=li*corr+p;\n"
        "  long vo=(((b*(long)Hkv+hk)*Sk+n)*Dv);for(int d=0;d<Dv;++d)acc[d]=acc[d]*corr+p*v[vo+d];mi=mn;}\n"
        " float inv=li>0.f?1.f/li:0.f;long oo=(((b*(long)Hq+qh)*Sq+m)*Dv);for(int d=0;d<Dv;++d)o[oo+d]=acc[d]*inv;}\n"
        f'extern "C" int {_FLASH_FWD_ENTRY}(const float*hq,const float*hk,const float*hv,const float*hb,float*ho,long B,int Hq,int Hkv,long Sq,long Sk,int D,int Dv,float scale,int causal,long wl,long wr,float softcap){{\n'
        " if(!hq||!hk||!hv||!ho||B<=0||Hq<=0||Hkv<=0||Hq%Hkv||Sq<=0||Sk<=0||D<=0||Dv<=0||Dv>TSR_FA_DV_CAP)return 2;\n"
        " size_t nq=(size_t)B*Hq*Sq*D*4,nk=(size_t)B*Hkv*Sk*D*4,nv=(size_t)B*Hkv*Sk*Dv*4,no=(size_t)B*Hq*Sq*Dv*4,nb=(size_t)B*Hq*Sq*Sk*4;float *dq=0,*dk=0,*dv=0,*db=0,*dout=0;\n"
        " if(cudaMalloc(&dq,nq)!=cudaSuccess||cudaMalloc(&dk,nk)!=cudaSuccess||cudaMalloc(&dv,nv)!=cudaSuccess||cudaMalloc(&dout,no)!=cudaSuccess){if(dq)cudaFree(dq);if(dk)cudaFree(dk);if(dv)cudaFree(dv);if(dout)cudaFree(dout);return 2;}\n"
        " if(hb&&cudaMalloc(&db,nb)!=cudaSuccess){cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dout);return 2;}\n"
        " if(cudaMemcpy(dq,hq,nq,cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(dk,hk,nk,cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(dv,hv,nv,cudaMemcpyHostToDevice)!=cudaSuccess||(hb&&cudaMemcpy(db,hb,nb,cudaMemcpyHostToDevice)!=cudaSuccess)){cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dout);if(db)cudaFree(db);return 3;}\n"
        " long rows=B*(long)Hq*Sq;tsr_flash_fwd<<<(unsigned)((rows+127)/128),128>>>(dq,dk,dv,db,dout,B,Hq,Hkv,Sq,Sk,D,Dv,scale,causal,wl,wr,softcap);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;\n"
        " if(ok==1&&cudaMemcpy(ho,dout,no,cudaMemcpyDeviceToHost)!=cudaSuccess)ok=3;cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dout);if(db)cudaFree(db);return ok;}\n"
    )


def _synthesize_flash_fwd_f16_cuda() -> str:
    """The Flash-forward contract with fp16 Q/K/V/O storage and fp32 math.

    Keep the f32 implementation as the semantic source of truth: this variant
    only changes boundary storage/conversions.  Scores, online-softmax state and
    output accumulators remain float, which is the storage contract used by the
    tensor-core lanes as well.
    """
    src = _synthesize_flash_fwd_cuda()
    src = src.replace("#include <cuda_runtime.h>\n#include <math.h>\n",
                      "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <math.h>\n")
    src = src.replace(
        "__global__ void tsr_flash_fwd(const float*q,const float*k,const float*v,"
        " const float*bias,float*o,",
        "__global__ void tsr_flash_fwd(const __half*q,const __half*k,const __half*v,"
        " const float*bias,__half*o,")
    src = src.replace("s+=q[qo+d]*k[ko+d];", "s+=__half2float(q[qo+d])*__half2float(k[ko+d]);")
    src = src.replace("p*v[vo+d]", "p*__half2float(v[vo+d])")
    src = src.replace("o[oo+d]=acc[d]*inv;", "o[oo+d]=__float2half_rn(acc[d]*inv);")
    src = src.replace(
        f'extern "C" int {_FLASH_FWD_ENTRY}(const float*hq,const float*hk,const float*hv,const float*hb,float*ho,',
        f'extern "C" int {_FLASH_FWD_ENTRY}(const __half*hq,const __half*hk,const __half*hv,const float*hb,__half*ho,')
    src = src.replace(
        "size_t nq=(size_t)B*Hq*Sq*D*4,nk=(size_t)B*Hkv*Sk*D*4,nv=(size_t)B*Hkv*Sk*Dv*4,no=(size_t)B*Hq*Sq*Dv*4,nb=(size_t)B*Hq*Sq*Sk*4;float *dq=0,*dk=0,*dv=0,*db=0,*dout=0;",
        "size_t nq=(size_t)B*Hq*Sq*D*2,nk=(size_t)B*Hkv*Sk*D*2,nv=(size_t)B*Hkv*Sk*Dv*2,no=(size_t)B*Hq*Sq*Dv*2,nb=(size_t)B*Hq*Sq*Sk*4;__half *dq=0,*dk=0,*dv=0,*dout=0;float *db=0;")
    return src


def run_flash_attention_forward(q: Any, k: Any, v: Any, *, scale: float,
                                causal: bool = False, window_left: int | None = None,
                                window_right: int | None = None, bias: Any = None,
                                softcap: float | None = None) -> Any:
    """Execute the f32-accumulating MHA/GQA/MQA Flash-forward contract."""
    import numpy as np
    global _flash_fwd_artifact
    qa, ka, va = (np.asarray(x) for x in (q, k, v))
    if any(x.dtype not in (np.float32, np.float16) for x in (qa, ka, va)):
        raise ValueError("NVIDIA flash forward supports f32/f16 storage")
    if len({x.dtype for x in (qa, ka, va)}) != 1:
        raise ValueError("NVIDIA flash forward requires Q/K/V to share a storage dtype")
    storage = "f32" if qa.dtype == np.float32 else "f16"
    ctype = np.float32 if storage == "f32" else np.float16
    q, k, v = (np.ascontiguousarray(x, ctype) for x in (qa, ka, va))
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("NVIDIA flash forward requires Q/K/V rank-4 [B,H,S,D]")
    B, Hq, Sq, D = q.shape
    Bk, Hkv, Sk, Dk = k.shape
    Bv, Hvv, Sv, Dv = v.shape
    if (B, D) != (Bk, Dk) or (B, Hkv, Sk) != (Bv, Hvv, Sv) or Hq % Hkv or Dv > _ATTN_DV_CAP:
        raise ValueError("invalid NVIDIA flash forward Q/K/V shape or head mapping")
    bf = None if bias is None else np.ascontiguousarray(bias, np.float32)
    if bf is not None and bf.shape != (B, Hq, Sq, Sk):
        raise ValueError("NVIDIA flash attention bias must have shape [B,Hq,Sq,Sk]")
    wl = -1 if window_left is None else int(window_left)
    wr = -1 if window_right is None else int(window_right)
    cap = 0.0 if softcap is None else float(softcap)
    if wl < -1 or wr < -1 or cap < 0.0:
        raise ValueError("window bounds must be >= 0 and softcap must be >= 0")
    artifact = _flash_fwd_artifact.get(storage)
    if artifact is None:
        artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=(_synthesize_flash_fwd_cuda() if storage == "f32" else _synthesize_flash_fwd_f16_cuda()),
            entry=_FLASH_FWD_ENTRY, lang=_LANG, spec=SpecPolicy.DYNAMIC,
            shape_key=(f"flash-fwd-contract-{storage}",)))
        _flash_fwd_artifact[storage] = artifact
    fn = getattr(_load_lib(artifact), _FLASH_FWD_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_long, ctypes.c_int, ctypes.c_int,
                  ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                  ctypes.c_float, ctypes.c_int, ctypes.c_long, ctypes.c_long,
                  ctypes.c_float]
    out = np.empty((B, Hq, Sq, Dv), dtype=ctype)
    rc = fn(_ptr(q), _ptr(k), _ptr(v), _ptr(bf), _ptr(out), B, Hq, Hkv, Sq, Sk,
            D, Dv, float(scale), int(causal), wl, wr, cap)
    if rc != 1:
        raise RuntimeError("NVIDIA Flash Attention forward CUDA launch failed")
    return out


# ── fused MLA decode (latent projection + online-softmax in one CUDA kernel) ─

def _synthesize_mla_fused_cuda() -> str:
    """Correctness-first fused MLA decode for f32 storage.

    ``x @ W_dkv`` and the K/V up-projections are evaluated inside the streamed
    key loop.  No expanded K/V tensor is allocated by the ABI.  One thread owns
    a query row; the latent and output dimensions are capped to keep transient
    state local, matching the correctness-first Flash lane.
    """
    return (
        "#include <cuda_runtime.h>\n#include <math.h>\n"
        f"#define TSR_MLA_CAP {_ATTN_DV_CAP}\n"
        "__global__ void tsr_mla_fused(const float*x,const float*wd,const float*wk,const float*wv,const float*q,float*o,long B,int Hq,int Hkv,long Sq,long Sk,int Dx,int L,int D,int Dv,float scale,int causal){\n"
        " long row=(long)blockIdx.x*blockDim.x+threadIdx.x,total=B*(long)Hq*Sq;if(row>=total)return;long m=row%Sq,t=row/Sq;int qh=(int)(t%Hq);long b=t/Hq;int hk=qh/(Hq/Hkv);float acc[TSR_MLA_CAP];for(int d=0;d<Dv;d++)acc[d]=0;float mx=-INFINITY,z=0;\n"
        " for(long n=0;n<Sk;n++){if(causal&&n>m)continue;long xo=(((b*(long)Hkv+hk)*Sk+n)*Dx),qo=(((b*(long)Hq+qh)*Sq+m)*D);float c[TSR_MLA_CAP];for(int l=0;l<L;l++){float a=0;for(int j=0;j<Dx;j++)a+=x[xo+j]*wd[j*(long)L+l];c[l]=a;}float s=0;for(int d=0;d<D;d++){float kd=0;for(int l=0;l<L;l++)kd+=c[l]*wk[l*(long)D+d];s+=q[qo+d]*kd;}s*=scale;float nm=fmaxf(mx,s),corr=expf(mx-nm),p=expf(s-nm);z=z*corr+p;for(int d=0;d<Dv;d++){float vd=0;for(int l=0;l<L;l++)vd+=c[l]*wv[l*(long)Dv+d];acc[d]=acc[d]*corr+p*vd;}mx=nm;}\n"
        " long oo=(((b*(long)Hq+qh)*Sq+m)*Dv);float inv=z>0?1.f/z:0;for(int d=0;d<Dv;d++)o[oo+d]=acc[d]*inv;}\n"
        f'extern "C" int {_MLA_FUSED_ENTRY}(const float*hx,const float*hwd,const float*hwk,const float*hwv,const float*hq,float*ho,long B,int Hq,int Hkv,long Sq,long Sk,int Dx,int L,int D,int Dv,float scale,int causal){{\n'
        " if(!hx||!hwd||!hwk||!hwv||!hq||!ho||B<=0||Hq<=0||Hkv<=0||Hq%Hkv||Sq<=0||Sk<=0||Dx<=0||L<=0||D<=0||Dv<=0||L>TSR_MLA_CAP||Dv>TSR_MLA_CAP)return 2;size_t nx=(size_t)B*Hkv*Sk*Dx*4,nwd=(size_t)Dx*L*4,nwk=(size_t)L*D*4,nwv=(size_t)L*Dv*4,nq=(size_t)B*Hq*Sq*D*4,no=(size_t)B*Hq*Sq*Dv*4;float *x=0,*wd=0,*wk=0,*wv=0,*q=0,*o=0;\n"
        " if(cudaMalloc(&x,nx)||cudaMalloc(&wd,nwd)||cudaMalloc(&wk,nwk)||cudaMalloc(&wv,nwv)||cudaMalloc(&q,nq)||cudaMalloc(&o,no)){if(x)cudaFree(x);if(wd)cudaFree(wd);if(wk)cudaFree(wk);if(wv)cudaFree(wv);if(q)cudaFree(q);if(o)cudaFree(o);return 2;}\n"
        " int bad=cudaMemcpy(x,hx,nx,cudaMemcpyHostToDevice)||cudaMemcpy(wd,hwd,nwd,cudaMemcpyHostToDevice)||cudaMemcpy(wk,hwk,nwk,cudaMemcpyHostToDevice)||cudaMemcpy(wv,hwv,nwv,cudaMemcpyHostToDevice)||cudaMemcpy(q,hq,nq,cudaMemcpyHostToDevice);int ok=3;if(!bad){long rows=B*(long)Hq*Sq;tsr_mla_fused<<<(unsigned)((rows+127)/128),128>>>(x,wd,wk,wv,q,o,B,Hq,Hkv,Sq,Sk,Dx,L,D,Dv,scale,causal);ok=cudaDeviceSynchronize()==cudaSuccess?1:3;if(ok==1&&cudaMemcpy(ho,o,no,cudaMemcpyDeviceToHost))ok=3;}cudaFree(x);cudaFree(wd);cudaFree(wk);cudaFree(wv);cudaFree(q);cudaFree(o);return ok;}\n"
    )


def run_mla_decode_fused(x: Any, w_dkv: Any, w_uk: Any, w_uv: Any, q: Any,
                         *, scale: float, causal: bool = False) -> Any:
    """Launch the dedicated f32 fused MLA CUDA entry point."""
    import numpy as np
    global _mla_fused_artifact
    x, w_dkv, w_uk, w_uv, q = (
        np.ascontiguousarray(a, np.float32) for a in (x, w_dkv, w_uk, w_uv, q))
    if x.ndim != 4 or q.ndim != 4 or any(w.ndim != 2 for w in (w_dkv, w_uk, w_uv)):
        raise ValueError("NVIDIA fused MLA requires rank-4 x/q and rank-2 weights")
    B, Hkv, Sk, Dx = x.shape; Bq, Hq, Sq, D = q.shape
    if Bq != B or Hq % Hkv or w_dkv.shape[0] != Dx:
        raise ValueError("invalid NVIDIA fused MLA batch/head/down-projection contract")
    L = w_dkv.shape[1]
    if w_uk.shape != (L, D) or w_uv.shape[0] != L or max(L, w_uv.shape[1]) > _ATTN_DV_CAP:
        raise ValueError("invalid NVIDIA fused MLA up-projection contract")
    Dv = w_uv.shape[1]
    if _mla_fused_artifact is None:
        _mla_fused_artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_synthesize_mla_fused_cuda(), entry=_MLA_FUSED_ENTRY,
            lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=("mla-decode-fused-f32",)))
    fn = getattr(_load_lib(_mla_fused_artifact), _MLA_FUSED_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_long, ctypes.c_int, ctypes.c_int,
                  ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    out = np.empty((B, Hq, Sq, Dv), np.float32)
    rc = fn(_ptr(x), _ptr(w_dkv), _ptr(w_uk), _ptr(w_uv), _ptr(q), _ptr(out),
            B, Hq, Hkv, Sq, Sk, Dx, L, D, Dv, float(scale), int(causal))
    if rc != 1:
        raise RuntimeError("NVIDIA fused MLA CUDA launch failed")
    return out


def _synthesize_flash_bwd_cuda() -> str:
    """Correctness-first f32 Flash-Attention VJP with GQA atomic accumulation."""
    return (
        "#include <cuda_runtime.h>\n#include <math.h>\n"
        "#define TSR_FA_CAP 256\n"
        "__global__ void tsr_flash_bwd(const float*go,const float*q,const float*k,const float*v,const float*bias,float*dq,float*dk,float*dv,long B,int Hq,int Hkv,long Sq,long Sk,int D,int Dv,float scale,int causal,long wl,long wr,float cap){\n"
        " long row=(long)blockIdx.x*blockDim.x+threadIdx.x,total=B*(long)Hq*Sq;if(row>=total)return;long m=row%Sq,tmp=row/Sq;int qh=(int)(tmp%Hq);long b=tmp/Hq,hk=qh/(Hq/Hkv);float acc[TSR_FA_CAP];for(int d=0;d<Dv;++d)acc[d]=0.f;float mx=-INFINITY;\n"
        " for(long n=0;n<Sk;++n){if((causal&&n>m)||(wl>=0&&n<m-wl)||(wr>=0&&n>m+wr))continue;long qo=(((b*(long)Hq+qh)*Sq+m)*D),ko=(((b*(long)Hkv+hk)*Sk+n)*D);float s=0.f;for(int d=0;d<D;++d)s+=q[qo+d]*k[ko+d];s*=scale;if(bias)s+=bias[(((b*(long)Hq+qh)*Sq+m)*Sk+n)];if(cap>0)s=cap*tanhf(s/cap);mx=fmaxf(mx,s);}\n"
        " float z=0.f;for(long n=0;n<Sk;++n){if((causal&&n>m)||(wl>=0&&n<m-wl)||(wr>=0&&n>m+wr))continue;long qo=(((b*(long)Hq+qh)*Sq+m)*D),ko=(((b*(long)Hkv+hk)*Sk+n)*D);float s=0.f;for(int d=0;d<D;++d)s+=q[qo+d]*k[ko+d];s*=scale;if(bias)s+=bias[(((b*(long)Hq+qh)*Sq+m)*Sk+n)];if(cap>0)s=cap*tanhf(s/cap);float p=expf(s-mx);z+=p;long vo=(((b*(long)Hkv+hk)*Sk+n)*Dv);for(int d=0;d<Dv;++d)acc[d]+=p*v[vo+d];}\n"
        " for(int d=0;d<Dv;++d)acc[d]/=z;long goo=(((b*(long)Hq+qh)*Sq+m)*Dv);float delta=0.f;for(int d=0;d<Dv;++d)delta+=go[goo+d]*acc[d];float aq[TSR_FA_CAP];for(int d=0;d<D;++d)aq[d]=0.f;\n"
        " for(long n=0;n<Sk;++n){if((causal&&n>m)||(wl>=0&&n<m-wl)||(wr>=0&&n>m+wr))continue;long qo=(((b*(long)Hq+qh)*Sq+m)*D),ko=(((b*(long)Hkv+hk)*Sk+n)*D);float raw=0.f;for(int d=0;d<D;++d)raw+=q[qo+d]*k[ko+d];raw*=scale;float s=raw;if(bias)s+=bias[(((b*(long)Hq+qh)*Sq+m)*Sk+n)];float deriv=1.f;if(cap>0){float t=tanhf(s/cap);s=cap*t;deriv=1.f-t*t;}float p=expf(s-mx)/z;long vo=(((b*(long)Hkv+hk)*Sk+n)*Dv);float dp=0.f;for(int d=0;d<Dv;++d){dp+=go[goo+d]*v[vo+d];atomicAdd(&dv[vo+d],p*go[goo+d]);}float ds=p*(dp-delta)*deriv;for(int d=0;d<D;++d){aq[d]+=ds*scale*k[ko+d];atomicAdd(&dk[ko+d],ds*scale*q[qo+d]);}}for(int d=0;d<D;++d)dq[(((b*(long)Hq+qh)*Sq+m)*D)+d]=aq[d];}\n"
        f'extern "C" int {_FLASH_BWD_ENTRY}(const float*hgo,const float*hq,const float*hk,const float*hv,const float*hb,float*hdq,float*hdk,float*hdv,long B,int Hq,int Hkv,long Sq,long Sk,int D,int Dv,float scale,int causal,long wl,long wr,float cap){{\n'
        " if(!hgo||!hq||!hk||!hv||!hdq||!hdk||!hdv||B<=0||Hq<=0||Hkv<=0||Hq%Hkv||Sq<=0||Sk<=0||D<=0||Dv<=0||D>TSR_FA_CAP||Dv>TSR_FA_CAP)return 2;size_t nq=(size_t)B*Hq*Sq*D*4,nkv=(size_t)B*Hkv*Sk*D*4,no=(size_t)B*Hq*Sq*Dv*4,nb=(size_t)B*Hq*Sq*Sk*4;float *go=0,*q=0,*k=0,*v=0,*bi=0,*dq=0,*dk=0,*dv=0;\n"
        " if(cudaMalloc(&go,no)||cudaMalloc(&q,nq)||cudaMalloc(&k,nkv)||cudaMalloc(&v,(size_t)B*Hkv*Sk*Dv*4)||cudaMalloc(&dq,nq)||cudaMalloc(&dk,nkv)||cudaMalloc(&dv,(size_t)B*Hkv*Sk*Dv*4))return 2;if(hb&&cudaMalloc(&bi,nb))return 2;\n"
        " cudaMemcpy(go,hgo,no,cudaMemcpyHostToDevice);cudaMemcpy(q,hq,nq,cudaMemcpyHostToDevice);cudaMemcpy(k,hk,nkv,cudaMemcpyHostToDevice);cudaMemcpy(v,hv,(size_t)B*Hkv*Sk*Dv*4,cudaMemcpyHostToDevice);if(hb)cudaMemcpy(bi,hb,nb,cudaMemcpyHostToDevice);cudaMemset(dk,0,nkv);cudaMemset(dv,0,(size_t)B*Hkv*Sk*Dv*4);long rows=B*(long)Hq*Sq;tsr_flash_bwd<<<(unsigned)((rows+127)/128),128>>>(go,q,k,v,bi,dq,dk,dv,B,Hq,Hkv,Sq,Sk,D,Dv,scale,causal,wl,wr,cap);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;if(ok==1){cudaMemcpy(hdq,dq,nq,cudaMemcpyDeviceToHost);cudaMemcpy(hdk,dk,nkv,cudaMemcpyDeviceToHost);cudaMemcpy(hdv,dv,(size_t)B*Hkv*Sk*Dv*4,cudaMemcpyDeviceToHost);}cudaFree(go);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(dq);cudaFree(dk);cudaFree(dv);if(bi)cudaFree(bi);return ok;}\n"
    )


def _synthesize_flash_bwd_f16_cuda() -> str:
    """fp16-storage wrapper around the f32 Flash VJP device kernel.

    Inputs are copied as half to device and widened there; the VJP and GQA/MQA
    atomic accumulation run in float, then returned gradients are narrowed on
    device.  Thus no host-side widening hides the storage contract.
    """
    f32 = _synthesize_flash_bwd_cuda()
    kernel = f32[:f32.index('extern "C" int')]
    kernel = kernel.replace("#include <cuda_runtime.h>\n",
                            "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n")
    return kernel + (
        "__global__ void tsr_h2f(const __half*x,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__half2float(x[i]);}\n"
        "__global__ void tsr_f2h(const float*x,__half*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__float2half_rn(x[i]);}\n"
        f'extern "C" int {_FLASH_BWD_ENTRY}(const __half*hgo,const __half*hq,const __half*hk,const __half*hv,const float*hb,__half*hdq,__half*hdk,__half*hdv,long B,int Hq,int Hkv,long Sq,long Sk,int D,int Dv,float scale,int causal,long wl,long wr,float cap){{\n'
        " if(!hgo||!hq||!hk||!hv||!hdq||!hdk||!hdv||B<=0||Hq<=0||Hkv<=0||Hq%Hkv||Sq<=0||Sk<=0||D<=0||Dv<=0||D>TSR_FA_CAP||Dv>TSR_FA_CAP)return 2;long eq=(long)B*Hq*Sq*D,ek=(long)B*Hkv*Sk*D,eo=(long)B*Hq*Sq*Dv,ev=(long)B*Hkv*Sk*Dv;size_t nq2=(size_t)eq*2,nk2=(size_t)ek*2,no2=(size_t)eo*2,nv2=(size_t)ev*2,nq4=(size_t)eq*4,nk4=(size_t)ek*4,no4=(size_t)eo*4,nv4=(size_t)ev*4,nb=(size_t)B*Hq*Sq*Sk*4;__half *hgo_d=0,*hq_d=0,*hk_d=0,*hv_d=0,*hdq_d=0,*hdk_d=0,*hdv_d=0;float *go=0,*q=0,*k=0,*v=0,*dq=0,*dk=0,*dv=0,*bi=0;\n"
        " if(cudaMalloc(&hgo_d,no2)||cudaMalloc(&hq_d,nq2)||cudaMalloc(&hk_d,nk2)||cudaMalloc(&hv_d,nv2)||cudaMalloc(&hdq_d,nq2)||cudaMalloc(&hdk_d,nk2)||cudaMalloc(&hdv_d,nv2)||cudaMalloc(&go,no4)||cudaMalloc(&q,nq4)||cudaMalloc(&k,nk4)||cudaMalloc(&v,nv4)||cudaMalloc(&dq,nq4)||cudaMalloc(&dk,nk4)||cudaMalloc(&dv,nv4))return 2;if(hb&&cudaMalloc(&bi,nb))return 2;\n"
        " if(cudaMemcpy(hgo_d,hgo,no2,cudaMemcpyHostToDevice)||cudaMemcpy(hq_d,hq,nq2,cudaMemcpyHostToDevice)||cudaMemcpy(hk_d,hk,nk2,cudaMemcpyHostToDevice)||cudaMemcpy(hv_d,hv,nv2,cudaMemcpyHostToDevice)||(hb&&cudaMemcpy(bi,hb,nb,cudaMemcpyHostToDevice)))return 3;int t=128;tsr_h2f<<<(unsigned)((eo+t-1)/t),t>>>(hgo_d,go,eo);tsr_h2f<<<(unsigned)((eq+t-1)/t),t>>>(hq_d,q,eq);tsr_h2f<<<(unsigned)((ek+t-1)/t),t>>>(hk_d,k,ek);tsr_h2f<<<(unsigned)((ev+t-1)/t),t>>>(hv_d,v,ev);cudaMemset(dk,0,nk4);cudaMemset(dv,0,nv4);long rows=B*(long)Hq*Sq;tsr_flash_bwd<<<(unsigned)((rows+127)/128),128>>>(go,q,k,v,bi,dq,dk,dv,B,Hq,Hkv,Sq,Sk,D,Dv,scale,causal,wl,wr,cap);tsr_f2h<<<(unsigned)((eq+t-1)/t),t>>>(dq,hdq_d,eq);tsr_f2h<<<(unsigned)((ek+t-1)/t),t>>>(dk,hdk_d,ek);tsr_f2h<<<(unsigned)((ev+t-1)/t),t>>>(dv,hdv_d,ev);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;if(ok==1&&(cudaMemcpy(hdq,hdq_d,nq2,cudaMemcpyDeviceToHost)||cudaMemcpy(hdk,hdk_d,nk2,cudaMemcpyDeviceToHost)||cudaMemcpy(hdv,hdv_d,nv2,cudaMemcpyDeviceToHost)))ok=3;cudaFree(hgo_d);cudaFree(hq_d);cudaFree(hk_d);cudaFree(hv_d);cudaFree(hdq_d);cudaFree(hdk_d);cudaFree(hdv_d);cudaFree(go);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(dq);cudaFree(dk);cudaFree(dv);if(bi)cudaFree(bi);return ok;}\n"
    )


def run_flash_attention_backward(go: Any, q: Any, k: Any, v: Any, *, scale: float,
                                 causal: bool = False, window_left: int | None = None,
                                 window_right: int | None = None, bias: Any = None,
                                 softcap: float | None = None) -> tuple[Any, Any, Any]:
    """Execute f32-accumulating Flash-Attention VJP; returns ``(dQ, dK, dV)``."""
    import numpy as np
    global _flash_bwd_artifact
    goa, qa, ka, va = (np.asarray(x) for x in (go, q, k, v))
    if any(x.dtype not in (np.float32, np.float16) for x in (goa, qa, ka, va)) or len({x.dtype for x in (goa, qa, ka, va)}) != 1:
        raise ValueError("NVIDIA flash backward requires matching f32 or f16 dO/Q/K/V storage")
    storage = "f32" if qa.dtype == np.float32 else "f16"
    ctype = np.float32 if storage == "f32" else np.float16
    go, q, k, v = (np.ascontiguousarray(x, ctype) for x in (goa, qa, ka, va))
    if q.ndim != 4 or go.shape[:3] != q.shape[:3] or k.ndim != 4 or v.ndim != 4:
        raise ValueError("NVIDIA flash backward requires rank-4 dO/Q/K/V tensors")
    B, Hq, Sq, D = q.shape; Bk, Hkv, Sk, Dk = k.shape; Bv, Hvv, Sv, Dv = v.shape
    if (B, D) != (Bk, Dk) or (B, Hkv, Sk) != (Bv, Hvv, Sv) or go.shape[-1] != Dv or Hq % Hkv or max(D, Dv) > _ATTN_DV_CAP:
        raise ValueError("invalid NVIDIA flash backward Q/K/V/dO shape or head mapping")
    bf = None if bias is None else np.ascontiguousarray(bias, np.float32)
    if bf is not None and bf.shape != (B, Hq, Sq, Sk):
        raise ValueError("NVIDIA flash backward bias must have shape [B,Hq,Sq,Sk]")
    wl, wr = (-1 if window_left is None else int(window_left), -1 if window_right is None else int(window_right))
    cap = 0.0 if softcap is None else float(softcap)
    if wl < -1 or wr < -1 or cap < 0.0:
        raise ValueError("window bounds must be >= 0 and softcap must be >= 0")
    artifact = _flash_bwd_artifact.get(storage)
    if artifact is None:
        artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=(_synthesize_flash_bwd_cuda() if storage == "f32" else _synthesize_flash_bwd_f16_cuda()),
            entry=_FLASH_BWD_ENTRY, lang=_LANG, spec=SpecPolicy.DYNAMIC,
            shape_key=(f"flash-bwd-contract-{storage}",)))
        _flash_bwd_artifact[storage] = artifact
    fn = getattr(_load_lib(artifact), _FLASH_BWD_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_long, ctypes.c_int, ctypes.c_int,
                  ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                  ctypes.c_float, ctypes.c_int, ctypes.c_long, ctypes.c_long,
                  ctypes.c_float]
    dq, dk, dv = np.empty_like(q), np.empty_like(k), np.empty_like(v)
    rc = fn(_ptr(go), _ptr(q), _ptr(k), _ptr(v), _ptr(bf), _ptr(dq), _ptr(dk), _ptr(dv),
            B, Hq, Hkv, Sq, Sk, D, Dv, float(scale), int(causal), wl, wr, cap)
    if rc != 1:
        raise RuntimeError("NVIDIA Flash Attention backward CUDA launch failed")
    return dq, dk, dv


# ── causal linear-attention contract lane (CUDA parity P3) ──────────────────

def _synthesize_linear_attn_cuda() -> str:
    """f32 identity-feature-map causal linear attention, O=(QKᵀ tril) V.

    This is deliberately a direct CUDA implementation rather than composing
    NumPy matmuls: each output element streams legal keys.  It is the stable
    base ABI for later feature-map and decay variants.
    """
    return (
        "#include <cuda_runtime.h>\n"
        "__global__ void tsr_la(const float*q,const float*k,const float*v,float*o,long B,int H,long S,int D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x,total=B*(long)H*S*D;if(x>=total)return;int d=x%D;long t=x/D,m=t%S,z=t/S,h=z%H,b=z/H;float y=0;for(long n=0;n<=m;n++){long qo=(((b*(long)H+h)*S+m)*D),ko=(((b*(long)H+h)*S+n)*D);float s=0;for(int j=0;j<D;j++)s+=q[qo+j]*k[ko+j];y+=s*v[ko+d];}o[x]=y;}\n"
        f'extern "C" int {_LINEAR_ATTN_ENTRY}(const float*hq,const float*hk,const float*hv,float*ho,long B,int H,long S,int D){{'
        "if(!hq||!hk||!hv||!ho||B<=0||H<=0||S<=0||D<=0)return 2;size_t n=(size_t)B*H*S*D*4;float*q=0,*k=0,*v=0,*o=0;if(cudaMalloc(&q,n)||cudaMalloc(&k,n)||cudaMalloc(&v,n)||cudaMalloc(&o,n))return 2;if(cudaMemcpy(q,hq,n,cudaMemcpyHostToDevice)||cudaMemcpy(k,hk,n,cudaMemcpyHostToDevice)||cudaMemcpy(v,hv,n,cudaMemcpyHostToDevice))return 3;long all=B*(long)H*S*D;tsr_la<<<(unsigned)((all+127)/128),128>>>(q,k,v,o,B,H,S,D);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;if(ok==1&&cudaMemcpy(ho,o,n,cudaMemcpyDeviceToHost))ok=3;cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);return ok;}\n"
    )


def run_linear_attention(q: Any, k: Any, v: Any) -> Any:
    """Launch causal f32 identity linear attention on the NVIDIA device."""
    import numpy as np
    global _linear_attn_artifact
    q, k, v = (np.ascontiguousarray(x, np.float32) for x in (q, k, v))
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("NVIDIA linear_attn requires matching f32 [B,H,S,D] Q/K/V")
    B, H, S, D = q.shape
    if _linear_attn_artifact is None:
        _linear_attn_artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_synthesize_linear_attn_cuda(), entry=_LINEAR_ATTN_ENTRY,
            lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=("linear-attn-f32",)))
    fn = getattr(_load_lib(_linear_attn_artifact), _LINEAR_ATTN_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_long, ctypes.c_int,
                  ctypes.c_long, ctypes.c_int]
    out = np.empty_like(q)
    if fn(_ptr(q), _ptr(k), _ptr(v), _ptr(out), B, H, S, D) != 1:
        raise RuntimeError("NVIDIA linear_attn CUDA launch failed")
    return out


def _synthesize_linear_attn_bwd_cuda() -> str:
    """f32 VJP for the causal identity linear-attention base contract."""
    return (
        "#include <cuda_runtime.h>\n#define D_CAP 256\n"
        "__global__ void tsr_la_bwd(const float*go,const float*q,const float*k,const float*v,float*dq,float*dk,float*dv,long B,int H,long S,int D){long x=(long)blockIdx.x*blockDim.x+threadIdx.x,total=B*(long)H*S;if(x>=total)return;long m=x%S,z=x/S,h=z%H,b=z/H,qo=(((b*(long)H+h)*S+m)*D);float aq[D_CAP];for(int j=0;j<D;j++)aq[j]=0;for(long n=0;n<=m;n++){long ko=(((b*(long)H+h)*S+n)*D);float score=0,ds=0;for(int j=0;j<D;j++)score+=q[qo+j]*k[ko+j];for(int d=0;d<D;d++){ds+=go[qo+d]*v[ko+d];atomicAdd(&dv[ko+d],score*go[qo+d]);}for(int j=0;j<D;j++){aq[j]+=ds*k[ko+j];atomicAdd(&dk[ko+j],ds*q[qo+j]);}}for(int j=0;j<D;j++)dq[qo+j]=aq[j];}\n"
        f'extern "C" int {_LINEAR_ATTN_BWD_ENTRY}(const float*hgo,const float*hq,const float*hk,const float*hv,float*hdq,float*hdk,float*hdv,long B,int H,long S,int D){{'
        "if(!hgo||!hq||!hk||!hv||!hdq||!hdk||!hdv||B<=0||H<=0||S<=0||D<=0||D>D_CAP)return 2;size_t n=(size_t)B*H*S*D*4;float*go=0,*q=0,*k=0,*v=0,*dq=0,*dk=0,*dv=0;if(cudaMalloc(&go,n)||cudaMalloc(&q,n)||cudaMalloc(&k,n)||cudaMalloc(&v,n)||cudaMalloc(&dq,n)||cudaMalloc(&dk,n)||cudaMalloc(&dv,n))return 2;if(cudaMemcpy(go,hgo,n,cudaMemcpyHostToDevice)||cudaMemcpy(q,hq,n,cudaMemcpyHostToDevice)||cudaMemcpy(k,hk,n,cudaMemcpyHostToDevice)||cudaMemcpy(v,hv,n,cudaMemcpyHostToDevice))return 3;cudaMemset(dk,0,n);cudaMemset(dv,0,n);long rows=B*(long)H*S;tsr_la_bwd<<<(unsigned)((rows+127)/128),128>>>(go,q,k,v,dq,dk,dv,B,H,S,D);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;if(ok==1&&(cudaMemcpy(hdq,dq,n,cudaMemcpyDeviceToHost)||cudaMemcpy(hdk,dk,n,cudaMemcpyDeviceToHost)||cudaMemcpy(hdv,dv,n,cudaMemcpyDeviceToHost)))ok=3;cudaFree(go);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(dq);cudaFree(dk);cudaFree(dv);return ok;}\n"
    )


def run_linear_attention_backward(go: Any, q: Any, k: Any, v: Any) -> tuple[Any, Any, Any]:
    """Launch f32 dQ/dK/dV for causal identity linear attention."""
    import numpy as np
    global _linear_attn_bwd_artifact
    go, q, k, v = (np.ascontiguousarray(x, np.float32) for x in (go, q, k, v))
    if q.ndim != 4 or go.shape != q.shape or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("NVIDIA linear_attn VJP requires matching f32 [B,H,S,D] tensors")
    B, H, S, D = q.shape
    if _linear_attn_bwd_artifact is None:
        _linear_attn_bwd_artifact = _nvidia_cuda_compile_fn(KernelSource(source=_synthesize_linear_attn_bwd_cuda(), entry=_LINEAR_ATTN_BWD_ENTRY, lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=("linear-attn-bwd-f32",)))
    fn = getattr(_load_lib(_linear_attn_bwd_artifact), _LINEAR_ATTN_BWD_ENTRY)
    fn.restype = ctypes.c_int; fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_long, ctypes.c_int, ctypes.c_long, ctypes.c_int]
    dq, dk, dv = np.empty_like(q), np.empty_like(k), np.empty_like(v)
    if fn(_ptr(go), _ptr(q), _ptr(k), _ptr(v), _ptr(dq), _ptr(dk), _ptr(dv), B, H, S, D) != 1:
        raise RuntimeError("NVIDIA linear_attn VJP CUDA launch failed")
    return dq, dk, dv


def _synthesize_linear_attn_variant_cuda() -> str:
    """General f32 causal linear-attention forward: Dqk/Dv, fmap, decay."""
    return (
        "#include <cuda_runtime.h>\n#include <math.h>\n"
        "__device__ float phi(float x,int f){return f==1?fmaxf(x,0.f):(f==2?x*x:x);}\n"
        "__global__ void tsr_lav(const float*q,const float*k,const float*v,const float*dec,float*o,long B,int H,long S,int D,int Dv,int fmap){long x=(long)blockIdx.x*blockDim.x+threadIdx.x,total=B*(long)H*S*Dv;if(x>=total)return;int d=x%Dv;long t=x/Dv,m=t%S,z=t/S,h=z%H,b=z/H;float y=0;for(long n=0;n<=m;n++){long qo=(((b*(long)H+h)*S+m)*D),ko=(((b*(long)H+h)*S+n)*D);float s=0,fac=1;for(int j=0;j<D;j++)s+=phi(q[qo+j],fmap)*phi(k[ko+j],fmap);if(dec)for(long u=n+1;u<=m;u++)fac*=dec[((b*(long)H+h)*S+u)];y+=fac*s*v[(((b*(long)H+h)*S+n)*Dv+d)];}o[(((b*(long)H+h)*S+m)*Dv+d)]=y;}\n"
        "extern \"C\" int tessera_nvidia_linear_attn_variant(const float*hq,const float*hk,const float*hv,const float*hd,float*ho,long B,int H,long S,int D,int Dv,int fmap){if(!hq||!hk||!hv||!ho||B<=0||H<=0||S<=0||D<=0||Dv<=0||fmap<0||fmap>2)return 2;size_t nq=(size_t)B*H*S*D*4,nv=(size_t)B*H*S*Dv*4,nd=(size_t)B*H*S*4;float*q=0,*k=0,*v=0,*d=0,*o=0;if(cudaMalloc(&q,nq)||cudaMalloc(&k,nq)||cudaMalloc(&v,nv)||cudaMalloc(&o,nv))return 2;if(hd&&cudaMalloc(&d,nd))return 2;if(cudaMemcpy(q,hq,nq,cudaMemcpyHostToDevice)||cudaMemcpy(k,hk,nq,cudaMemcpyHostToDevice)||cudaMemcpy(v,hv,nv,cudaMemcpyHostToDevice)||(hd&&cudaMemcpy(d,hd,nd,cudaMemcpyHostToDevice)))return 3;long n=B*(long)H*S*Dv;tsr_lav<<<(unsigned)((n+127)/128),128>>>(q,k,v,d,o,B,H,S,D,Dv,fmap);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;if(ok==1&&cudaMemcpy(ho,o,nv,cudaMemcpyDeviceToHost))ok=3;cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);if(d)cudaFree(d);return ok;}\n"
    )


def run_linear_attention_variant(q: Any, k: Any, v: Any, *, feature_map: str,
                                 decay: Any = None) -> Any:
    import numpy as np
    global _linear_attn_variant_artifact
    q, k, v = (np.ascontiguousarray(x, np.float32) for x in (q, k, v))
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("NVIDIA linear-attention variants require Q/K [B,H,S,D], V [B,H,S,Dv]")
    code = {"identity": 0, "relu": 1, "polynomial_2": 2}.get(feature_map)
    if code is None: raise ValueError("unsupported NVIDIA linear-attention feature map")
    dec = None if decay is None else np.ascontiguousarray(decay, np.float32)
    if dec is not None and dec.shape != q.shape[:3]: raise ValueError("decay must have shape [B,H,S]")
    if _linear_attn_variant_artifact is None:
        _linear_attn_variant_artifact = _nvidia_cuda_compile_fn(KernelSource(source=_synthesize_linear_attn_variant_cuda(), entry="tessera_nvidia_linear_attn_variant", lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=("linear-attn-variant-f32",)))
    fn = getattr(_load_lib(_linear_attn_variant_artifact), "tessera_nvidia_linear_attn_variant")
    fn.restype = ctypes.c_int; fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_long, ctypes.c_int, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    out = np.empty_like(v)
    if fn(_ptr(q), _ptr(k), _ptr(v), _ptr(dec), _ptr(out), q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3], code) != 1: raise RuntimeError("NVIDIA linear-attention variant CUDA launch failed")
    return out


def _synthesize_linear_attn_variant_bwd_cuda() -> str:
    return ("#include <cuda_runtime.h>\n#include <math.h>\n#define DC 256\n"
        "__device__ float p(float x,int f){return f==2?x*x:x;}__device__ float dp(float x,int f){return f==2?2*x:1;}\n"
        "__global__ void kb(const float*go,const float*q,const float*k,const float*v,const float*de,float*dq,float*dk,float*dv,long B,int H,long S,int D,int Vd,int f){long x=(long)blockIdx.x*blockDim.x+threadIdx.x,total=B*(long)H*S;if(x>=total)return;long m=x%S,z=x/S,h=z%H,b=z/H,qo=(((b*(long)H+h)*S+m)*D),g=(((b*(long)H+h)*S+m)*Vd);float a[DC];for(int j=0;j<D;j++)a[j]=0;for(long n=0;n<=m;n++){long ko=(((b*(long)H+h)*S+n)*D),vo=(((b*(long)H+h)*S+n)*Vd);float s=0,fac=1,ds=0;for(int j=0;j<D;j++)s+=p(q[qo+j],f)*p(k[ko+j],f);if(de)for(long u=n+1;u<=m;u++)fac*=de[((b*(long)H+h)*S+u)];for(int d=0;d<Vd;d++){ds+=go[g+d]*v[vo+d];atomicAdd(&dv[vo+d],fac*s*go[g+d]);}ds*=fac;for(int j=0;j<D;j++){a[j]+=ds*dp(q[qo+j],f)*p(k[ko+j],f);atomicAdd(&dk[ko+j],ds*dp(k[ko+j],f)*p(q[qo+j],f));}}for(int j=0;j<D;j++)dq[qo+j]=a[j];}\n"
        "extern \"C\" int tessera_nvidia_linear_attn_variant_bwd(const float*hg,const float*hq,const float*hk,const float*hv,const float*hd,float*hdq,float*hdk,float*hdv,long B,int H,long S,int D,int Vd,int f){if(!hg||!hq||!hk||!hv||!hdq||!hdk||!hdv||D>DC)return 2;size_t nq=(size_t)B*H*S*D*4,nv=(size_t)B*H*S*Vd*4,nd=(size_t)B*H*S*4;float*g=0,*q=0,*k=0,*v=0,*d=0,*dq=0,*dk=0,*dv=0;int ok=1;if(cudaMalloc(&g,nv)||cudaMalloc(&q,nq)||cudaMalloc(&k,nq)||cudaMalloc(&v,nv)||cudaMalloc(&dq,nq)||cudaMalloc(&dk,nq)||cudaMalloc(&dv,nv)||(hd&&cudaMalloc(&d,nd))){ok=2;}if(ok==1&&(cudaMemcpy(g,hg,nv,cudaMemcpyHostToDevice)||cudaMemcpy(q,hq,nq,cudaMemcpyHostToDevice)||cudaMemcpy(k,hk,nq,cudaMemcpyHostToDevice)||cudaMemcpy(v,hv,nv,cudaMemcpyHostToDevice)||(hd&&cudaMemcpy(d,hd,nd,cudaMemcpyHostToDevice))||cudaMemset(dk,0,nq)||cudaMemset(dv,0,nv))){ok=3;}if(ok==1){long r=B*(long)H*S;kb<<<(r+127)/128,128>>>(g,q,k,v,d,dq,dk,dv,B,H,S,D,Vd,f);ok=cudaDeviceSynchronize()==cudaSuccess?1:3;}if(ok==1&&(cudaMemcpy(hdq,dq,nq,cudaMemcpyDeviceToHost)||cudaMemcpy(hdk,dk,nq,cudaMemcpyDeviceToHost)||cudaMemcpy(hdv,dv,nv,cudaMemcpyDeviceToHost)))ok=3;if(g)cudaFree(g);if(q)cudaFree(q);if(k)cudaFree(k);if(v)cudaFree(v);if(d)cudaFree(d);if(dq)cudaFree(dq);if(dk)cudaFree(dk);if(dv)cudaFree(dv);return ok;}\n")


def run_linear_attention_variant_backward(go: Any,q: Any,k: Any,v: Any,*,feature_map: str,decay: Any=None) -> tuple[Any,Any,Any]:
    import numpy as np
    global _linear_attn_variant_bwd_artifact
    go,q,k,v=(np.ascontiguousarray(x,np.float32) for x in (go,q,k,v)); code={"identity":0,"polynomial_2":2}.get(feature_map)
    if code is None or q.ndim!=4 or k.shape!=q.shape or go.shape!=v.shape or v.shape[:3]!=q.shape[:3]: raise ValueError("invalid NVIDIA linear-attention variant VJP contract")
    de=None if decay is None else np.ascontiguousarray(decay,np.float32)
    if de is not None and de.shape!=q.shape[:3]: raise ValueError("decay must be [B,H,S]")
    if _linear_attn_variant_bwd_artifact is None: _linear_attn_variant_bwd_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_linear_attn_variant_bwd_cuda(),entry="tessera_nvidia_linear_attn_variant_bwd",lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("linear-attn-variant-bwd",)))
    fn=getattr(_load_lib(_linear_attn_variant_bwd_artifact),"tessera_nvidia_linear_attn_variant_bwd");fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]*8+[ctypes.c_long,ctypes.c_int,ctypes.c_long,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    dq,dk,dv=np.empty_like(q),np.empty_like(k),np.empty_like(v)
    if fn(_ptr(go),_ptr(q),_ptr(k),_ptr(v),_ptr(de),_ptr(dq),_ptr(dk),_ptr(dv),q.shape[0],q.shape[1],q.shape[2],q.shape[3],v.shape[3],code)!=1: raise RuntimeError("NVIDIA variant VJP failed")
    return dq,dk,dv


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
        # DYNAMIC is supported: every generic CUDA lane takes M/N/K as runtime args
        # with in-kernel bounds guards (dims-invariant source), so one compiled
        # kernel serves every shape (Workstream G / W2). DYNAMIC only changes the
        # shape_key below to the symbolic identity, collapsing the cache to one
        # entry across all shapes.
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


# ── standalone row-softmax (CUDA parity P1) ──────────────────────────────────

_softmax_artifact: dict[str, str] = {}


def _synthesize_softmax_cuda() -> str:
    """Stable f32 softmax over the last axis, one CUDA block per row.

    This is deliberately a small, compiler-emitted vertical slice matching the
    ROCm row-softmax contract: input is flattened to ``[M, K]``, reduction and
    exponentiation stay in f32, and ``K`` may exceed the block size.  Norm and
    generic reduction lanes can reuse this launch/cache shape.
    """
    return (
        "#include <cuda_runtime.h>\n"
        "#include <float.h>\n"
        "#include <math.h>\n"
        "#define TSR_SM_BLOCK 256\n"
        "__global__ void tsr_softmax_kernel(const float* x, float* o, long K) {\n"
        "  const long row = (long)blockIdx.x; const int tid = threadIdx.x;\n"
        "  __shared__ float scratch[TSR_SM_BLOCK];\n"
        "  float mx = -FLT_MAX;\n"
        "  for (long j = tid; j < K; j += blockDim.x) mx = fmaxf(mx, x[row*K+j]);\n"
        "  scratch[tid] = mx; __syncthreads();\n"
        "  for (int s = blockDim.x/2; s; s >>= 1) {\n"
        "    if (tid < s) scratch[tid] = fmaxf(scratch[tid], scratch[tid+s]);\n"
        "    __syncthreads();\n"
        "  }\n"
        "  mx = scratch[0]; float sum = 0.0f;\n"
        "  for (long j = tid; j < K; j += blockDim.x) sum += expf(x[row*K+j] - mx);\n"
        "  scratch[tid] = sum; __syncthreads();\n"
        "  for (int s = blockDim.x/2; s; s >>= 1) {\n"
        "    if (tid < s) scratch[tid] += scratch[tid+s]; __syncthreads();\n"
        "  }\n"
        "  sum = scratch[0];\n"
        "  for (long j = tid; j < K; j += blockDim.x) o[row*K+j] = expf(x[row*K+j]-mx)/sum;\n"
        "}\n"
        f'extern "C" int {_SOFTMAX_ENTRY}(const float* hx, float* ho, long M, long K) {{\n'
        "  if (!hx || !ho || M <= 0 || K <= 0) return 2;\n"
        "  const size_t bytes = (size_t)M * (size_t)K * sizeof(float);\n"
        "  float *dx = 0, *dout = 0;\n"
        "  if (cudaMalloc(&dx, bytes) != cudaSuccess) return 2;\n"
        "  if (cudaMalloc(&dout, bytes) != cudaSuccess) { cudaFree(dx); return 2; }\n"
        "  if (cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(dx); cudaFree(dout); return 3; }\n"
        "  tsr_softmax_kernel<<<(unsigned)M, TSR_SM_BLOCK>>>(dx, dout, K);\n"
        "  int ok = (cudaDeviceSynchronize() == cudaSuccess) ? 1 : 3;\n"
        "  if (ok == 1 && cudaMemcpy(ho, dout, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ok = 3;\n"
        "  cudaFree(dx); cudaFree(dout); return ok;\n"
        "}\n"
    )


def _synthesize_softmax_f16_cuda() -> str:
    """The fp16-storage sibling of :func:`_synthesize_softmax_cuda`."""
    return (
        "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <float.h>\n#include <math.h>\n"
        "#define TSR_SM_BLOCK 256\n"
        "__global__ void tsr_softmax_kernel(const __half*x,__half*o,long K){\n"
        " long r=(long)blockIdx.x;int t=threadIdx.x;__shared__ float s[TSR_SM_BLOCK];float mx=-FLT_MAX;\n"
        " for(long j=t;j<K;j+=blockDim.x)mx=fmaxf(mx,__half2float(x[r*K+j]));s[t]=mx;__syncthreads();\n"
        " for(int d=blockDim.x/2;d;d>>=1){if(t<d)s[t]=fmaxf(s[t],s[t+d]);__syncthreads();}mx=s[0];float sum=0.f;\n"
        " for(long j=t;j<K;j+=blockDim.x)sum+=expf(__half2float(x[r*K+j])-mx);s[t]=sum;__syncthreads();\n"
        " for(int d=blockDim.x/2;d;d>>=1){if(t<d)s[t]+=s[t+d];__syncthreads();}sum=s[0];\n"
        " for(long j=t;j<K;j+=blockDim.x)o[r*K+j]=__float2half(expf(__half2float(x[r*K+j])-mx)/sum);}\n"
        f'extern "C" int {_SOFTMAX_ENTRY}(const __half*hx,__half*ho,long M,long K){{\n'
        " if(!hx||!ho||M<=0||K<=0)return 2;size_t n=(size_t)M*K*sizeof(__half);__half *dx=0,*dout=0;\n"
        " if(cudaMalloc(&dx,n)!=cudaSuccess)return 2;if(cudaMalloc(&dout,n)!=cudaSuccess){cudaFree(dx);return 2;}\n"
        " if(cudaMemcpy(dx,hx,n,cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(dx);cudaFree(dout);return 3;}\n"
        " tsr_softmax_kernel<<<(unsigned)M,TSR_SM_BLOCK>>>(dx,dout,K);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;\n"
        " if(ok==1&&cudaMemcpy(ho,dout,n,cudaMemcpyDeviceToHost)!=cudaSuccess)ok=3;cudaFree(dx);cudaFree(dout);return ok;}\n"
    )


def run_row_softmax(x: Any) -> Any:
    """Execute the compiler-emitted f32 row-softmax on NVIDIA hardware.

    The public runtime calls this only after validating its artifact contract;
    unavailable CUDA/toolchain errors propagate rather than becoming a falsely
    labelled reference result.
    """
    import numpy as np
    xf = np.ascontiguousarray(x)
    if xf.dtype == np.float32:
        dtype, source = "f32", _synthesize_softmax_cuda()
    elif xf.dtype == np.float16:
        dtype, source = "f16", _synthesize_softmax_f16_cuda()
    else:
        raise ValueError(f"NVIDIA softmax supports f32/f16 storage, got {xf.dtype}")
    if xf.ndim < 1 or xf.shape[-1] <= 0:
        raise ValueError("softmax operand must have a non-empty last dimension")
    m = int(np.prod(xf.shape[:-1])) if xf.ndim > 1 else 1
    k = int(xf.shape[-1])
    artifact = _softmax_artifact.get(dtype)
    if artifact is None:
        artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=source, entry=_SOFTMAX_ENTRY, lang=_LANG,
            spec=SpecPolicy.DYNAMIC, shape_key=(f"row-softmax-{dtype}",)))
        _softmax_artifact[dtype] = artifact
    fn = getattr(_load_lib(artifact), _SOFTMAX_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
    out = np.empty_like(xf)
    if fn(_ptr(xf), _ptr(out), m, k) != 1:
        raise RuntimeError("NVIDIA row-softmax CUDA launch failed")
    return out


# ── standalone row norms (CUDA parity P1) ────────────────────────────────────

_norm_artifact: dict[str, str] = {}


def _synthesize_norm_cuda(dtype: str = "f32") -> str:
    """CUDA source for RMSNorm / LayerNorm with f32 accumulation.

    ``f16`` retains half storage at the ABI boundary but converts each element to
    f32 before both reduction passes.  That is the same storage/accumulation
    contract as the ROCm row-norm lane and avoids a silent accuracy downgrade.
    """
    if dtype not in {"f32", "f16"}:
        raise ValueError(f"unsupported NVIDIA norm dtype {dtype!r}")
    ctype = "float" if dtype == "f32" else "__half"
    load = "x[r*K+j]" if dtype == "f32" else "__half2float(x[r*K+j])"
    store = "v" if dtype == "f32" else "__float2half_rn(v)"
    preamble = "#include <cuda_runtime.h>\n#include <math.h>\n"
    if dtype == "f16":
        preamble += "#include <cuda_fp16.h>\n"
    return (
        preamble +
        "#define TSR_NM_BLOCK 256\n"
        "__device__ float tsr_sum(float v, float* s) {\n"
        "  int t=threadIdx.x; s[t]=v; __syncthreads();\n"
        "  for(int d=blockDim.x/2; d; d>>=1) { if(t<d) s[t]+=s[t+d]; __syncthreads(); }\n"
        "  return s[0];\n"
        "}\n"
        f"__global__ void tsr_norm_kernel(const {ctype}* x, {ctype}* o, long K, float eps, int layer) {{\n"
        "  const long r=(long)blockIdx.x; const int t=threadIdx.x; __shared__ float s[TSR_NM_BLOCK];\n"
        "  float sum=0.f, sq=0.f;\n"
        f"  for(long j=t;j<K;j+=blockDim.x) {{ float v={load}; sum+=v; sq+=v*v; }}\n"
        "  const float mean=layer ? tsr_sum(sum,s)/(float)K : 0.f;\n"
        "  float denom;\n"
        f"  if(layer) {{ float dev=0.f; for(long j=t;j<K;j+=blockDim.x) {{ float d={load}-mean; dev+=d*d; }} denom=rsqrtf(tsr_sum(dev,s)/(float)K+eps); }}\n"
        "  else denom=rsqrtf(tsr_sum(sq,s)/(float)K+eps);\n"
        f"  for(long j=t;j<K;j+=blockDim.x) {{ float v=({load}-(layer?mean:0.f))*denom; o[r*K+j]={store}; }}\n"
        "}\n"
        f'extern "C" int tessera_nvidia_norm(const {ctype}* hx,{ctype}* ho,long M,long K,float eps,int layer) {{\n'
        f"  if(!hx||!ho||M<=0||K<=0||eps<0.f) return 2; size_t n=(size_t)M*K*sizeof({ctype}); {ctype} *dx=0,*dout=0;\n"
        "  if(cudaMalloc(&dx,n)!=cudaSuccess) return 2; if(cudaMalloc(&dout,n)!=cudaSuccess){cudaFree(dx);return 2;}\n"
        "  if(cudaMemcpy(dx,hx,n,cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(dx);cudaFree(dout);return 3;}\n"
        "  tsr_norm_kernel<<<(unsigned)M,TSR_NM_BLOCK>>>(dx,dout,K,eps,layer); int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;\n"
        "  if(ok==1&&cudaMemcpy(ho,dout,n,cudaMemcpyDeviceToHost)!=cudaSuccess) ok=3; cudaFree(dx);cudaFree(dout);return ok;\n"
        "}\n"
    )


def run_row_norm(x: Any, kind: str, eps: float) -> Any:
    """Execute generated f32/f16-storage RMSNorm or LayerNorm on NVIDIA."""
    import numpy as np
    global _norm_artifact
    if kind not in {"rmsnorm", "layer_norm"}:
        raise ValueError(f"unknown NVIDIA norm kind {kind!r}")
    xa = np.asarray(x)
    if xa.dtype not in (np.float32, np.float16):
        raise ValueError(f"NVIDIA row norm supports f32/f16 storage; got {xa.dtype}")
    dtype = "f32" if xa.dtype == np.float32 else "f16"
    xf = np.ascontiguousarray(xa)
    if xf.ndim < 1 or xf.shape[-1] <= 0:
        raise ValueError("norm operand must have a non-empty last dimension")
    m = int(np.prod(xf.shape[:-1])) if xf.ndim > 1 else 1
    k = int(xf.shape[-1])
    artifact = _norm_artifact.get(dtype)
    if artifact is None:
        artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_synthesize_norm_cuda(dtype), entry="tessera_nvidia_norm",
            lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=(f"row-norm-{dtype}",)))
        _norm_artifact[dtype] = artifact
    fn = getattr(_load_lib(artifact), "tessera_nvidia_norm")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long,
                   ctypes.c_float, ctypes.c_int]
    out = np.empty_like(xf)
    if fn(_ptr(xf), _ptr(out), m, k, float(eps), int(kind == "layer_norm")) != 1:
        raise RuntimeError("NVIDIA row-norm CUDA launch failed")
    return out


# ── standalone row reductions (CUDA parity P1) ───────────────────────────────

_reduce_artifact: dict[str, str] = {}


def _synthesize_reduce_cuda(dtype: str = "f32") -> str:
    """CUDA source for f32-accumulating reductions over each flattened row."""
    if dtype not in {"f32", "f16"}:
        raise ValueError(f"unsupported NVIDIA reduction dtype {dtype!r}")
    ctype = "float" if dtype == "f32" else "__half"
    load = "x[r*K+j]" if dtype == "f32" else "__half2float(x[r*K+j])"
    preamble = "#include <cuda_runtime.h>\n#include <math.h>\n#include <float.h>\n"
    if dtype == "f16":
        preamble += "#include <cuda_fp16.h>\n"
    return (
        preamble +
        "#define TSR_RD_BLOCK 256\n"
        "__device__ float tsr_rcombine(float a,float b,int kind){\n"
        " if(kind<2)return a+b; if(isnan(a)||isnan(b))return NAN; return kind==2?fmaxf(a,b):fminf(a,b);}\n"
        f"__global__ void tsr_reduce_kernel(const {ctype}*x,float*o,long K,int kind){{\n"
        " long r=(long)blockIdx.x;int t=threadIdx.x;__shared__ float s[TSR_RD_BLOCK];\n"
        " float v=kind==2?-INFINITY:(kind==3?INFINITY:0.f);\n"
        f" for(long j=t;j<K;j+=blockDim.x)v=tsr_rcombine(v,{load},kind);\n"
        " s[t]=v;__syncthreads();for(int d=blockDim.x/2;d;d>>=1){if(t<d)s[t]=tsr_rcombine(s[t],s[t+d],kind);__syncthreads();}\n"
        " if(t==0)o[r]=(kind==1?s[0]/(float)K:s[0]);}\n"
        f"extern \"C\" int tessera_nvidia_reduce(const {ctype}*hx,float*ho,long M,long K,int kind){{\n"
        f" if(!hx||!ho||M<=0||K<=0||kind<0||kind>3)return 2;size_t n=(size_t)M*K*sizeof({ctype});{ctype} *dx=0;float *dout=0;\n"
        " if(cudaMalloc(&dx,n)!=cudaSuccess)return 2;if(cudaMalloc(&dout,(size_t)M*sizeof(float))!=cudaSuccess){cudaFree(dx);return 2;}\n"
        " if(cudaMemcpy(dx,hx,n,cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(dx);cudaFree(dout);return 3;}\n"
        " tsr_reduce_kernel<<<(unsigned)M,TSR_RD_BLOCK>>>(dx,dout,K,kind);int ok=cudaDeviceSynchronize()==cudaSuccess?1:3;\n"
        " if(ok==1&&cudaMemcpy(ho,dout,(size_t)M*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)ok=3;cudaFree(dx);cudaFree(dout);return ok;}\n"
    )


def run_row_reduce(x2d: Any, kind: str) -> Any:
    """Execute a generated f32/f16-storage row reduction with f32 output."""
    import numpy as np
    global _reduce_artifact
    code = {"sum": 0, "mean": 1, "max": 2, "min": 3}.get(kind)
    if code is None:
        raise ValueError(f"unknown NVIDIA reduction kind {kind!r}")
    xa = np.asarray(x2d)
    if xa.dtype not in (np.float32, np.float16):
        raise ValueError(f"NVIDIA row reduction supports f32/f16 storage; got {xa.dtype}")
    dtype = "f32" if xa.dtype == np.float32 else "f16"
    xf = np.ascontiguousarray(xa)
    if xf.ndim != 2 or xf.shape[0] <= 0 or xf.shape[1] <= 0:
        raise ValueError("NVIDIA row reduction requires a non-empty [M, K] input")
    artifact = _reduce_artifact.get(dtype)
    if artifact is None:
        artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_synthesize_reduce_cuda(dtype), entry="tessera_nvidia_reduce",
            lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=(f"row-reduce-{dtype}",)))
        _reduce_artifact[dtype] = artifact
    fn = getattr(_load_lib(artifact), "tessera_nvidia_reduce")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_int]
    out = np.empty((xf.shape[0],), dtype=np.float32)
    if fn(_ptr(xf), _ptr(out), xf.shape[0], xf.shape[1], code) != 1:
        raise RuntimeError("NVIDIA row-reduce CUDA launch failed")
    return out


# ── runner (execute → (out, tag)) ─────────────────────────────────────────────

def _synthesize_fpquant_cuda() -> str:
    return r'''#include <cuda_runtime.h>
#include <math.h>
__global__ void fq_k(const float*x,float*o,long n,float mx,int mb,int me){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=x[i];if(isnan(z)){o[i]=z;return;}float a=fminf(fabsf(z),mx);if(a==0){o[i]=z;return;}int e=(int)floorf(log2f(a));e=e<me?me:e;float step=ldexpf(1.f,e-mb);float q=nearbyintf(z/step)*step;o[i]=fminf(fmaxf(q,-mx),mx);}}
extern "C" int tessera_nvidia_fpquant_f32(const float*x,float*o,long n,float mx,int mb,int me){float *dx=0,*d=0;size_t z=n*4;if(cudaMalloc(&dx,z)!=cudaSuccess||cudaMalloc(&d,z)!=cudaSuccess||cudaMemcpy(dx,x,z,cudaMemcpyHostToDevice)!=cudaSuccess)return 3;fq_k<<<(n+255)/256,256>>>(dx,d,n,mx,mb,me);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(d);return ok?1:3;}
'''


def run_fpquant_f32(x: Any, max_normal: float, mantissa_bits: int,
                    min_exp: int, scale: float) -> Any:
    import numpy as np
    x=np.ascontiguousarray(x)
    if x.dtype!=np.float32:raise ValueError("NVIDIA fpquant requires f32")
    if not x.size:return np.array(x,copy=True)
    scaled=np.clip(x/np.float32(scale),-max_normal,max_normal).astype(np.float32)
    global _fpquant_artifact
    if _fpquant_artifact is None:_fpquant_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_fpquant_cuda(),entry=_FPQUANT_ENTRY,lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("fpquant-f32",)))
    out=np.empty_like(x);fn=getattr(_load_lib(_fpquant_artifact),_FPQUANT_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long,ctypes.c_float,ctypes.c_int,ctypes.c_int]
    if fn(_ptr(scaled),_ptr(out),x.size,max_normal,mantissa_bits,min_exp)!=1:raise RuntimeError("NVIDIA fpquant launch failed")
    return (out*np.float32(scale)).astype(np.float32)


def _synthesize_local_collective_cuda() -> str:
    return r'''#include <cuda_runtime.h>
__global__ void copy_k(const float*x,float*o,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=x[i];}
extern "C" int tessera_nvidia_local_collective_f32(const float*x,float*o,long n){float *dx=0,*d=0;size_t z=n*4;if(cudaMalloc(&dx,z)!=cudaSuccess||cudaMalloc(&d,z)!=cudaSuccess||cudaMemcpy(dx,x,z,cudaMemcpyHostToDevice)!=cudaSuccess)return 3;copy_k<<<(n+255)/256,256>>>(dx,d,n);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(d);return ok?1:3;}
'''


def run_local_collective_f32(x: Any) -> Any:
    import numpy as np
    x=np.ascontiguousarray(x)
    if x.dtype!=np.float32:raise ValueError("NVIDIA local collective v1 requires f32")
    if not x.size:return np.array(x,copy=True)
    global _local_collective_artifact
    if _local_collective_artifact is None:_local_collective_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_local_collective_cuda(),entry=_LOCAL_COLLECTIVE_ENTRY,lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("local-collective-f32",)))
    out=np.empty_like(x);fn=getattr(_load_lib(_local_collective_artifact),_LOCAL_COLLECTIVE_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long]
    if fn(_ptr(x),_ptr(out),x.size)!=1:raise RuntimeError("NVIDIA local collective launch failed")
    return out


def _synthesize_optimizer_cuda() -> str:
    return r'''#include <cuda_runtime.h>
#include <math.h>
__global__ void opt_k(const float*p,const float*g,const float*m,const float*v,float*po,float*mo,float*vo,long n,int kind,float lr,float b1,float b2,float eps,float wd,float b1c,float b2c){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float P=p[i],G=g[i],M=m[i],V=v[i],PN=P,MN=M,VN=V;if(kind==0)PN=P-lr*G;else if(kind==1||kind==5){VN=b1*V+G;PN=P-lr*(kind==5?G+b1*VN:VN);}else if(kind==4){float u=b1*M+(1-b1)*G;MN=b2*M+(1-b2)*G;float pp=wd!=0?P*(1-lr*wd):P;PN=pp-lr*((u>0)-(u<0));}else{MN=b1*M+(1-b1)*G;VN=b2*V+(1-b2)*G*G;float pp=(kind==3&&wd!=0)?P*(1-lr*wd):P;PN=pp-lr*(MN/b1c)/(sqrtf(VN/b2c)+eps);}po[i]=PN;mo[i]=MN;vo[i]=VN;}}
static int ac(float**d,const float*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_optimizer_f32(const float*p,const float*g,const float*m,const float*v,float*po,float*mo,float*vo,long n,int kind,float lr,float b1,float b2,float eps,float wd,float b1c,float b2c){float *dp=0,*dg=0,*dm=0,*dv=0,*dpo=0,*dmo=0,*dvo=0;size_t z=n*4;if(!ac(&dp,p,z)||!ac(&dg,g,z)||!ac(&dm,m,z)||!ac(&dv,v,z)||cudaMalloc(&dpo,z)!=cudaSuccess||cudaMalloc(&dmo,z)!=cudaSuccess||cudaMalloc(&dvo,z)!=cudaSuccess)return 3;opt_k<<<(n+255)/256,256>>>(dp,dg,dm,dv,dpo,dmo,dvo,n,kind,lr,b1,b2,eps,wd,b1c,b2c);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(po,dpo,z,cudaMemcpyDeviceToHost)==cudaSuccess&&cudaMemcpy(mo,dmo,z,cudaMemcpyDeviceToHost)==cudaSuccess&&cudaMemcpy(vo,dvo,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dp);cudaFree(dg);cudaFree(dm);cudaFree(dv);cudaFree(dpo);cudaFree(dmo);cudaFree(dvo);return ok?1:3;}
'''


def run_optimizer_f32(kind: int, p: Any, g: Any, m: Any, v: Any, n: int,
                      lr: float, b1: float, b2: float, eps: float, wd: float,
                      b1c: float, b2c: float):
    import numpy as np
    vals=[np.ascontiguousarray(x,dtype=np.float32).reshape(-1) for x in (p,g,m,v)]
    if any(x.size!=n for x in vals):raise ValueError("optimizer buffers must have n elements")
    global _optimizer_artifact
    if _optimizer_artifact is None:_optimizer_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_optimizer_cuda(),entry=_OPTIMIZER_ENTRY,lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("optimizer-f32",)))
    outs=[np.empty(n,np.float32) for _ in range(3)];fn=getattr(_load_lib(_optimizer_artifact),_OPTIMIZER_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_long,ctypes.c_int]+[ctypes.c_float]*7
    if fn(*[_ptr(x) for x in vals],*[_ptr(x) for x in outs],n,kind,lr,b1,b2,eps,wd,b1c,b2c)!=1:raise RuntimeError("NVIDIA optimizer launch failed")
    return tuple(outs)


def _synthesize_dequant_grouped_cuda() -> str:
    return r'''#include <cuda_runtime.h>
__global__ void dqg_k(const float*x,const unsigned char*codes,const float*sc,const int*off,float*o,int T,int K,int N,int E,int G,int mode){long z=(long)blockIdx.x*blockDim.x+threadIdx.x,total=(long)T*N;if(z<total){int r=z/N,c=z%N,e=0;while(e+1<E&&r>=off[e+1])++e;float a=0;for(int k=0;k<K;++k){int gi=k/G;float q;if(mode==4){int kb=K/2;unsigned char b=codes[((long)e*N+c)*kb+k/2];int v=(k&1)?(b>>4):(b&15);q=(float)(v>=8?v-16:v);}else{q=(float)((const signed char*)codes)[((long)e*K+k)*N+c];}a+=x[(long)r*K+k]*q*sc[((long)e*(K/G)+gi)*N+c];}o[z]=a;}}
static int ac(void**d,const void*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_dequant_grouped_gemm_f32(const float*x,const unsigned char*c,const float*s,const int*off,float*o,int T,int K,int N,int E,int G,int mode){size_t nx=(size_t)T*K*4,nc=(size_t)E*(mode==4?N*(K/2):K*N),ns=(size_t)E*(K/G)*N*4,no=(size_t)T*N*4;float *dx=0,*ds=0,*dout=0;unsigned char*dc=0;int*df=0;if(!ac((void**)&dx,x,nx)||!ac((void**)&dc,c,nc)||!ac((void**)&ds,s,ns)||!ac((void**)&df,off,(E+1)*4)||cudaMalloc(&dout,no)!=cudaSuccess)return 3;dqg_k<<<(T*N+255)/256,256>>>(dx,dc,ds,df,dout,T,K,N,E,G,mode);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,dout,no,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(dc);cudaFree(ds);cudaFree(df);cudaFree(dout);return ok?1:3;}
'''


def run_dequant_grouped_gemm_f32(x: Any, packed_experts: Any,
                                  group_sizes: Any) -> Any:
    import numpy as np
    x=np.ascontiguousarray(x);experts=list(packed_experts);gs=np.asarray(group_sizes,dtype=np.int64).reshape(-1)
    if x.dtype!=np.float32 or x.ndim!=2 or not experts:raise ValueError("dequant grouped GEMM requires x[T,K] f32 and experts")
    E=len(experts);first=experts[0];K,N=first.shape;scheme=first.scheme;G=scheme.group_size or K
    if scheme.dtype not in ("int4","int8") or K%G or (scheme.dtype=="int4" and K%2):raise ValueError("NVIDIA dequant grouped GEMM supports aligned int4/int8")
    if gs.shape!=(E,) or np.any(gs<0) or int(gs.sum())!=x.shape[0] or x.shape[1]!=K:raise ValueError("invalid dequant grouped GEMM partition")
    for p in experts:
        if p.shape!=(K,N) or p.scheme!=scheme:raise ValueError("dequant grouped experts must share shape and scheme")
    codes=np.ascontiguousarray(np.stack([p.codes for p in experts])).view(np.uint8);sc=np.ascontiguousarray(np.stack([p.scales for p in experts]),dtype=np.float32)
    off=np.ascontiguousarray(np.concatenate(([0],np.cumsum(gs))),dtype=np.int32);out=np.empty((x.shape[0],N),np.float32)
    if not x.shape[0]:return out
    global _dequant_artifact
    if _dequant_artifact is None:_dequant_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_dequant_grouped_cuda(),entry=_DEQUANT_GROUPED_ENTRY,lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("dequant-grouped-f32",)))
    fn=getattr(_load_lib(_dequant_artifact),_DEQUANT_GROUPED_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*6
    if fn(_ptr(x),_ptr(codes),_ptr(sc),_ptr(off),_ptr(out),x.shape[0],K,N,E,G,4 if scheme.dtype=="int4" else 8)!=1:raise RuntimeError("NVIDIA dequant grouped GEMM launch failed")
    return out


def _synthesize_moe_cuda() -> str:
    return r'''#include <cuda_runtime.h>
__global__ void gather_k(const float*x,const int*tok,float*o,int S,int H){long z=(long)blockIdx.x*blockDim.x+threadIdx.x,n=(long)S*H;if(z<n){int s=z/H,h=z%H;o[z]=x[(long)tok[s]*H+h];}}
__global__ void combine_k(const float*x,const int*tok,const float*w,float*o,int S,int H){long z=(long)blockIdx.x*blockDim.x+threadIdx.x,n=(long)S*H;if(z<n){int s=z/H,h=z%H;atomicAdd(&o[(long)tok[s]*H+h],w[s]*x[z]);}}
__global__ void gg_k(const float*x,const float*w,const int*off,float*o,int T,int K,int N,int E){long z=(long)blockIdx.x*blockDim.x+threadIdx.x,n=(long)T*N;if(z<n){int r=z/N,c=z%N,e=0;while(e+1<E&&r>=off[e+1])++e;float a=0;for(int k=0;k<K;++k)a+=x[(long)r*K+k]*w[((long)e*K+k)*N+c];o[z]=a;}}
static int ac(void**d,const void*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_moe_dispatch_f32(const float*x,const int*tok,float*o,int T,int S,int H){float *dx=0,*dout=0;int*dt=0;size_t nx=(size_t)T*H*4,no=(size_t)S*H*4;if(!ac((void**)&dx,x,nx)||!ac((void**)&dt,tok,S*4)||cudaMalloc(&dout,no)!=cudaSuccess)return 3;gather_k<<<(S*H+255)/256,256>>>(dx,dt,dout,S,H);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,dout,no,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(dt);cudaFree(dout);return ok?1:3;}
extern "C" int tessera_nvidia_moe_combine_f32(const float*x,const int*tok,const float*w,float*o,int T,int S,int H){float *dx=0,*dw=0,*dout=0;int*dt=0;size_t ni=(size_t)S*H*4,no=(size_t)T*H*4;if(!ac((void**)&dx,x,ni)||!ac((void**)&dt,tok,S*4)||!ac((void**)&dw,w,S*4)||cudaMalloc(&dout,no)!=cudaSuccess)return 3;cudaMemset(dout,0,no);combine_k<<<(S*H+255)/256,256>>>(dx,dt,dw,dout,S,H);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,dout,no,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(dt);cudaFree(dw);cudaFree(dout);return ok?1:3;}
extern "C" int tessera_nvidia_grouped_gemm_f32(const float*x,const float*w,const int*off,float*o,int T,int K,int N,int E){float *dx=0,*dw=0,*dout=0;int*df=0;size_t nx=(size_t)T*K*4,nw=(size_t)E*K*N*4,no=(size_t)T*N*4;if(!ac((void**)&dx,x,nx)||!ac((void**)&dw,w,nw)||!ac((void**)&df,off,(E+1)*4)||cudaMalloc(&dout,no)!=cudaSuccess)return 3;gg_k<<<(T*N+255)/256,256>>>(dx,dw,df,dout,T,K,N,E);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,dout,no,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(dw);cudaFree(df);cudaFree(dout);return ok?1:3;}
'''


def _moe_lib():
    global _moe_artifact
    if _moe_artifact is None:
        _moe_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_moe_cuda(),entry=_GROUPED_GEMM_ENTRY,lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("moe-transport-f32",)))
    return _load_lib(_moe_artifact)


def run_moe_dispatch_f32(x: Any, token_of_slot: Any) -> Any:
    import numpy as np
    x=np.ascontiguousarray(x);raw=np.asarray(token_of_slot,dtype=np.int64).reshape(-1)
    if np.any(raw>np.iinfo(np.int32).max):raise ValueError("MoE token indices exceed i32 CUDA ABI")
    tok=np.ascontiguousarray(raw,dtype=np.int32)
    if x.dtype!=np.float32 or x.ndim!=2:raise ValueError("NVIDIA MoE dispatch requires x[T,H] f32")
    if np.any(tok<0) or np.any(tok>=x.shape[0]):raise ValueError("MoE token indices out of range")
    if not tok.size:return np.empty((0,x.shape[1]),np.float32)
    out=np.empty((tok.size,x.shape[1]),np.float32);fn=getattr(_moe_lib(),_MOE_DISPATCH_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3
    if fn(_ptr(x),_ptr(tok),_ptr(out),x.shape[0],tok.size,x.shape[1])!=1:raise RuntimeError("NVIDIA MoE dispatch launch failed")
    return out


def run_moe_combine_f32(partials: Any, token_of_slot: Any, weights: Any,
                        num_tokens: int) -> Any:
    import numpy as np
    p=np.ascontiguousarray(partials);tok=np.ascontiguousarray(token_of_slot,dtype=np.int32).reshape(-1);w=np.ascontiguousarray(weights,dtype=np.float32).reshape(-1)
    if p.dtype!=np.float32 or p.ndim!=2 or tok.size!=p.shape[0] or w.size!=p.shape[0]:raise ValueError("invalid NVIDIA MoE combine shapes")
    if np.any(tok<0) or np.any(tok>=num_tokens):raise ValueError("MoE combine token indices out of range")
    if not p.shape[0]:return np.zeros((num_tokens,p.shape[1]),np.float32)
    out=np.empty((num_tokens,p.shape[1]),np.float32);fn=getattr(_moe_lib(),_MOE_COMBINE_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3
    if fn(_ptr(p),_ptr(tok),_ptr(w),_ptr(out),num_tokens,p.shape[0],p.shape[1])!=1:raise RuntimeError("NVIDIA MoE combine launch failed")
    return out


def run_grouped_gemm_f32(x: Any, weights: Any, group_sizes: Any) -> Any:
    import numpy as np
    x=np.ascontiguousarray(x);w=np.ascontiguousarray(weights);gs=np.asarray(group_sizes,dtype=np.int64).reshape(-1)
    if x.dtype!=np.float32 or w.dtype!=np.float32 or x.ndim!=2 or w.ndim!=3:raise ValueError("grouped_gemm requires f32 x[T,K], weights[E,K,N]")
    E,K,N=w.shape
    if x.shape[1]!=K or gs.shape!=(E,) or np.any(gs<0) or int(gs.sum())!=x.shape[0]:raise ValueError("invalid grouped_gemm K/group_sizes")
    if not x.shape[0]:return np.empty((0,N),np.float32)
    off64=np.concatenate(([0],np.cumsum(gs)))
    if int(off64[-1])>np.iinfo(np.int32).max:raise ValueError("grouped_gemm offsets exceed i32 CUDA ABI")
    off=np.ascontiguousarray(off64,dtype=np.int32);out=np.empty((x.shape[0],N),np.float32)
    fn=getattr(_moe_lib(),_GROUPED_GEMM_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4
    if fn(_ptr(x),_ptr(w),_ptr(off),_ptr(out),x.shape[0],K,N,E)!=1:raise RuntimeError("NVIDIA grouped_gemm launch failed")
    return out


def _synthesize_deltanet_cuda() -> str:
    return r'''#include <cuda_runtime.h>
#include <math.h>
__global__ void dn_k(const float*q,const float*k,const float*v,const float*g,const float*beta,const float*decay,float*o,int BH,int S,int DK,int DV,int erase,int modified){int bh=blockIdx.x,e=threadIdx.x;if(bh>=BH||e>=DV)return;extern __shared__ float sm[];float*st=sm;float*target=st+DK*DV;float*ms=target+DV;for(int z=e;z<DK*DV;z+=DV)st[z]=0.f;__syncthreads();for(int t=0;t<S;++t){long base=((long)bh*S+t);float de=decay?decay[base]:1.f;float tar=v[base*DV+e];if(erase){float vh=0.f;for(int d=0;d<DK;++d)vh+=k[base*DK+d]*st[d*DV+e];tar-=de*vh;}target[e]=tar;__syncthreads();if(e==0){float sc=1.f;if(modified){float kn=0.f,tn=0.f;for(int d=0;d<DK;++d){float z=k[base*DK+d];kn+=z*z;}for(int j=0;j<DV;++j)tn+=target[j]*target[j];sc=1.f/(1.f+sqrtf(kn*tn));}*ms=sc;}__syncthreads();float w=beta?beta[base]:1.f;for(int d=0;d<DK;++d)st[d*DV+e]=de*st[d*DV+e]+w*k[base*DK+d]*target[e]*(*ms);float y=0.f;for(int d=0;d<DK;++d)y+=q[base*DK+d]*st[d*DV+e];if(g)y*=1.f/(1.f+expf(-g[base*DV+e]));o[base*DV+e]=y;__syncthreads();}}
static int ac(float**d,const float*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_deltanet_f32(const float*q,const float*k,const float*v,const float*g,const float*beta,const float*decay,float*o,int B,int H,int S,int DK,int DV,int erase,int modified){if(DK<1||DK>64||DV<1||DV>64)return 2;int BH=B*H;size_t nq=(size_t)BH*S*DK*4,nv=(size_t)BH*S*DV*4,ns=(size_t)BH*S*4;float *dq=0,*dk=0,*dv=0,*dg=0,*db=0,*dd=0,*dout=0;if(!ac(&dq,q,nq)||!ac(&dk,k,nq)||!ac(&dv,v,nv)||cudaMalloc(&dout,nv)!=cudaSuccess)return 3;if(g&&!ac(&dg,g,nv))return 3;if(beta&&!ac(&db,beta,ns))return 3;if(decay&&!ac(&dd,decay,ns))return 3;size_t sh=((size_t)DK*DV+DV+1)*4;dn_k<<<BH,DV,sh>>>(dq,dk,dv,dg,db,dd,dout,BH,S,DK,DV,erase,modified);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,dout,nv,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dout);if(dg)cudaFree(dg);if(db)cudaFree(db);if(dd)cudaFree(dd);return ok?1:3;}
'''


def run_deltanet_f32(q: Any, k: Any, v: Any, *, gate: Any = None,
                     beta: Any = None, decay: Any = None, erase: bool = False,
                     modified: bool = False) -> Any:
    import numpy as np
    q=np.ascontiguousarray(q);k=np.ascontiguousarray(k);v=np.ascontiguousarray(v)
    if any(x.dtype != np.float32 or x.ndim != 4 for x in (q,k,v)):
        raise ValueError("NVIDIA DeltaNet v1 requires rank-4 f32 Q/K/V")
    if k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("DeltaNet K must match Q and V must share B,H,S")
    B,H,S,DK=q.shape;DV=v.shape[-1]
    if DK>64 or DV>64: raise ValueError("DeltaNet v1 requires Dqk,Dv <= 64")
    def opt(x,shape,label):
        if x is None:return None
        a=np.ascontiguousarray(x,dtype=np.float32)
        if a.shape != shape:raise ValueError(f"DeltaNet {label} shape must be {shape}")
        return a
    gate=opt(gate,v.shape,"gate");beta=opt(beta,(B,H,S),"beta");decay=opt(decay,(B,H,S),"decay")
    global _deltanet_artifact
    if _deltanet_artifact is None:_deltanet_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_deltanet_cuda(),entry=_DELTANET_ENTRY,lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("deltanet-f32",)))
    out=np.empty_like(v);fn=getattr(_load_lib(_deltanet_artifact),_DELTANET_ENTRY);fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*7;null=ctypes.c_void_p()
    if fn(_ptr(q),_ptr(k),_ptr(v),_ptr(gate) if gate is not None else null,_ptr(beta) if beta is not None else null,_ptr(decay) if decay is not None else null,_ptr(out),B,H,S,DK,DV,int(erase),int(modified))!=1:raise RuntimeError("NVIDIA DeltaNet CUDA launch failed")
    return out


def _synthesize_ssm_cuda() -> str:
    return r'''#include <cuda_runtime.h>
#include <math.h>
#define MAX_N 64
__global__ void ssm_k(const float*x,const float*A,const float*Bv,const float*C,const float*dt,const float*g,const float*st,float*y,int BS,int S,int D,int N){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=BS*D)return;int b=z/D,d=z%D;float h[MAX_N];for(int n=0;n<N;++n)h[n]=st?st[((long)b*D+d)*N+n]:0.f;for(int t=0;t<S;++t){long xd=((long)b*S+t)*D+d;float xv=x[xd],dl=dt[xd],sum=0.f;for(int n=0;n<N;++n){h[n]=expf(dl*A[(long)d*N+n])*h[n]+dl*Bv[((long)b*S+t)*N+n]*xv;sum+=C[((long)b*S+t)*N+n]*h[n];}y[xd]=g?sum*g[xd]:sum;}}
static int ac(float**d,const float*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_selective_ssm_f32(const float*x,const float*A,const float*Bv,const float*C,const float*dt,const float*g,const float*st,float*y,int BS,int S,int D,int N){if(N<1||N>MAX_N)return 2;size_t nx=(size_t)BS*S*D*4,na=(size_t)D*N*4,nb=(size_t)BS*S*N*4,ns=(size_t)BS*D*N*4;float *dx=0,*da=0,*db=0,*dc=0,*dd=0,*dg=0,*ds=0,*dy=0;if(!ac(&dx,x,nx)||!ac(&da,A,na)||!ac(&db,Bv,nb)||!ac(&dc,C,nb)||!ac(&dd,dt,nx)||cudaMalloc(&dy,nx)!=cudaSuccess)return 3;if(g&&!ac(&dg,g,nx))return 3;if(st&&!ac(&ds,st,ns))return 3;ssm_k<<<(BS*D+127)/128,128>>>(dx,da,db,dc,dd,dg,ds,dy,BS,S,D,N);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(y,dy,nx,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dd);cudaFree(dy);if(dg)cudaFree(dg);if(ds)cudaFree(ds);return ok?1:3;}
'''


def run_selective_ssm_f32(x: Any, A: Any, B: Any, C: Any, delta: Any,
                          *, gate: Any = None, state: Any = None) -> Any:
    import numpy as np
    vals=[np.ascontiguousarray(v) for v in (x,A,B,C,delta)]
    x,A,B,C,delta=vals
    if any(v.dtype != np.float32 for v in vals) or x.ndim != 3:
        raise ValueError("NVIDIA selective_ssm v1 requires rank-3 f32 inputs")
    bs,s,d=x.shape
    if A.ndim != 2 or A.shape[0] != d: raise ValueError("selective_ssm A must be [D,N]")
    n=A.shape[1]
    if n < 1 or n > 64 or B.shape != (bs,s,n) or C.shape != B.shape or delta.shape != x.shape:
        raise ValueError("invalid selective_ssm shapes or N outside [1,64]")
    gate=None if gate is None else np.ascontiguousarray(gate,np.float32)
    state=None if state is None else np.ascontiguousarray(state,np.float32)
    if gate is not None and gate.shape != x.shape: raise ValueError("selective_ssm gate shape must match x")
    if state is not None and state.shape != (bs,d,n): raise ValueError("selective_ssm state must be [B,D,N]")
    global _ssm_artifact
    if _ssm_artifact is None:
        _ssm_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_ssm_cuda(),entry=_SSM_ENTRY,lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("ssm-f32",)))
    out=np.empty_like(x);fn=getattr(_load_lib(_ssm_artifact),_SSM_ENTRY);fn.restype=ctypes.c_int
    fn.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*4
    null=ctypes.c_void_p()
    if fn(_ptr(x),_ptr(A),_ptr(B),_ptr(C),_ptr(delta),_ptr(gate) if gate is not None else null,_ptr(state) if state is not None else null,_ptr(out),bs,s,d,n)!=1: raise RuntimeError("NVIDIA selective_ssm CUDA launch failed")
    return out


# ── ReplaySSM output-only decode (CUDA vertical slice) ──────────────────────

def _synthesize_ssm_replay_decode_cuda() -> str:
    """One-launch scalar-A ReplaySSM output reconstruction.

    The kernel owns one ``(batch, channel)`` output and reconstructs it from a
    checkpoint plus the live replay window.  This intentionally mirrors the
    established Metal ABI so ``SSMStateHandle`` can use it as an optional
    output-only fast path.  It does not materialize the ``[B,D,N]`` live state.
    """
    return r'''#include <cuda_runtime.h>
#include <math.h>
__global__ void ssm_replay_k(const float*delta,const float*x,const float*b,const float*s0,const float*c,const float*a,float*y,int B,int D,int N,int M){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=B*D)return;int bi=z/D,d=z%D;float dc=0.f;for(int i=0;i<M;++i)dc+=delta[((long)i*B+bi)*D+d]*a[d];float out=0.f;for(int n=0;n<N;++n)out+=c[(long)bi*N+n]*s0[((long)bi*D+d)*N+n];out*=expf(dc);float prefix=0.f;for(int i=0;i<M;++i){long q=((long)i*B+bi)*D+d;prefix+=delta[q]*a[d];float gram=0.f;for(int n=0;n<N;++n)gram+=c[(long)bi*N+n]*b[((long)i*B+bi)*N+n];out+=expf(dc-prefix)*delta[q]*x[q]*gram;}y[(long)bi*D+d]=out;}
static int cp(float**d,const float*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_ssm_replay_decode_f32(const float*hd,const float*hx,const float*hb,const float*hs,const float*hc,const float*ha,float*hy,int B,int D,int N,int M){if(!hd||!hx||!hb||!hs||!hc||!ha||!hy||B<1||D<1||N<1||M<1)return 2;size_t zbd=(size_t)M*B*D*4,zbn=(size_t)M*B*N*4,zs=(size_t)B*D*N*4,zc=(size_t)B*N*4,za=(size_t)D*4,zy=(size_t)B*D*4;float *d=0,*x=0,*b=0,*s=0,*c=0,*a=0,*y=0;if(!cp(&d,hd,zbd)||!cp(&x,hx,zbd)||!cp(&b,hb,zbn)||!cp(&s,hs,zs)||!cp(&c,hc,zc)||!cp(&a,ha,za)||cudaMalloc(&y,zy)!=cudaSuccess){if(d)cudaFree(d);if(x)cudaFree(x);if(b)cudaFree(b);if(s)cudaFree(s);if(c)cudaFree(c);if(a)cudaFree(a);return 3;}ssm_replay_k<<<(B*D+127)/128,128>>>(d,x,b,s,c,a,y,B,D,N,M);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(hy,y,zy,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(d);cudaFree(x);cudaFree(b);cudaFree(s);cudaFree(c);cudaFree(a);cudaFree(y);return ok?1:3;}
'''


def run_ssm_replay_decode_f32(delta: Any, x: Any, b: Any, s0: Any, c: Any,
                              a: Any) -> Any:
    """Run the scalar-A ReplaySSM output-only CUDA kernel.

    Inputs use the ``SSMStateHandle`` layout: ``delta/x=[M,B,D]``,
    ``b=[M,B,N]``, ``s0=[B,D,N]``, ``c=[B,N]``, and ``a=[D]``.  The wrapper is
    deliberately a correctness vertical slice; persistent device ownership is
    introduced by the state-handle runtime ABI, not hidden behind this call.
    """
    import numpy as np
    delta, x, b, s0, c, a = (
        np.ascontiguousarray(v, dtype=np.float32) for v in (delta, x, b, s0, c, a)
    )
    if delta.ndim != 3 or x.shape != delta.shape:
        raise ValueError("ReplaySSM delta and x must be matching [M,B,D] f32 arrays")
    M, B, D = delta.shape
    if (M < 1 or b.shape[:2] != (M, B) or s0.ndim != 3
            or s0.shape[:2] != (B, D) or c.shape != (B, s0.shape[2])
            or b.shape[2] != s0.shape[2] or a.shape != (D,)):
        raise ValueError("invalid scalar-A ReplaySSM decode shapes")
    N = s0.shape[2]
    global _ssm_replay_artifact
    if _ssm_replay_artifact is None:
        _ssm_replay_artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_synthesize_ssm_replay_decode_cuda(), entry=_SSM_REPLAY_ENTRY,
            lang=_LANG, spec=SpecPolicy.DYNAMIC, shape_key=("ssm-replay-f32",)))
    out = np.empty((B, D), dtype=np.float32)
    fn = getattr(_load_lib(_ssm_replay_artifact), _SSM_REPLAY_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4
    if fn(_ptr(delta), _ptr(x), _ptr(b), _ptr(s0), _ptr(c), _ptr(a), _ptr(out),
          B, D, N, M) != 1:
        raise RuntimeError("NVIDIA ReplaySSM CUDA launch failed")
    return out


_ssm_replay_device_artifact: str | None = None


def _synthesize_ssm_replay_device_cuda() -> str:
    """Persistent scalar-A ReplaySSM context; inputs stay in CUDA allocations."""
    return r'''#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
struct Q{float*d,*x,*b,*s,*c,*a,*y,*dy,*pd,*px,*pb,*pc,*py;cudaStream_t st;cudaEvent_t ev;int B,D,N,L,pending;};
__global__ void out(Q q,int M){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=q.B*q.D)return;int bi=z/q.D,di=z%q.D;float t=0,v=0,p=0;for(int i=0;i<M;i++)t+=q.d[((long)i*q.B+bi)*q.D+di]*q.a[di];for(int n=0;n<q.N;n++)v+=q.c[(long)bi*q.N+n]*q.s[((long)bi*q.D+di)*q.N+n];v*=expf(t);for(int i=0;i<M;i++){long k=((long)i*q.B+bi)*q.D+di;p+=q.d[k]*q.a[di];float g=0;for(int n=0;n<q.N;n++)g+=q.c[(long)bi*q.N+n]*q.b[((long)i*q.B+bi)*q.N+n];v+=expf(t-p)*q.d[k]*q.x[k]*g;}q.y[(long)bi*q.D+di]=v;}
__global__ void fl(Q q,int M){long z=(long)blockIdx.x*blockDim.x+threadIdx.x,n=(long)q.B*q.D*q.N;if(z>=n)return;int ni=z%q.N,di=(z/q.N)%q.D,bi=z/(q.N*q.D);float v=q.s[z];for(int i=0;i<M;i++){long k=((long)i*q.B+bi)*q.D+di;v=expf(q.d[k]*q.a[di])*v+q.d[k]*q.x[k]*q.b[((long)i*q.B+bi)*q.N+ni];}q.s[z]=v;}
extern "C" int cr(void**o,const float*s,const float*a,int B,int D,int N,int L){Q*q=(Q*)calloc(1,sizeof(Q));if(!q||B<1||D<1||N<1||L<1)return 2;q->B=B;q->D=D;q->N=N;q->L=L;size_t bd=(size_t)L*B*D*4,bn=(size_t)L*B*N*4,st=(size_t)B*D*N*4,c=(size_t)B*N*4,aa=(size_t)D*4,y=(size_t)B*D*4;if(cudaMalloc(&q->d,bd)||cudaMalloc(&q->x,bd)||cudaMalloc(&q->b,bn)||cudaMalloc(&q->s,st)||cudaMalloc(&q->c,c)||cudaMalloc(&q->a,aa)||cudaMalloc(&q->y,y)||cudaMalloc(&q->dy,bd)||cudaHostAlloc(&q->pd,bd,cudaHostAllocDefault)||cudaHostAlloc(&q->px,bd,cudaHostAllocDefault)||cudaHostAlloc(&q->pb,bn,cudaHostAllocDefault)||cudaHostAlloc(&q->pc,bn,cudaHostAllocDefault)||cudaHostAlloc(&q->py,bd,cudaHostAllocDefault)||cudaStreamCreate(&q->st)||cudaEventCreateWithFlags(&q->ev,cudaEventDisableTiming)||cudaMemcpy(q->s,s,st,cudaMemcpyHostToDevice)||cudaMemcpy(q->a,a,aa,cudaMemcpyHostToDevice))return 3;*o=q;return 1;}
extern "C" int ap(void*v,const float*d,const float*x,const float*b,int i){Q*q=(Q*)v;if(!q||i<0||i>=q->L)return 2;size_t bd=(size_t)q->B*q->D*4,bn=(size_t)q->B*q->N*4;return cudaMemcpy(q->d+(size_t)i*q->B*q->D,d,bd,cudaMemcpyHostToDevice)||cudaMemcpy(q->x+(size_t)i*q->B*q->D,x,bd,cudaMemcpyHostToDevice)||cudaMemcpy(q->b+(size_t)i*q->B*q->N,b,bn,cudaMemcpyHostToDevice)?3:1;}
extern "C" int de(void*v,const float*c,float*y,int M){Q*q=(Q*)v;if(!q||M<1||M>q->L)return 2;if(cudaMemcpy(q->c,c,(size_t)q->B*q->N*4,cudaMemcpyHostToDevice))return 3;out<<<(q->B*q->D+127)/128,128>>>(*q,M);return cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(y,q->y,(size_t)q->B*q->D*4,cudaMemcpyDeviceToHost)==cudaSuccess?1:3;}
extern "C" int fu(void*v,int M){Q*q=(Q*)v;if(!q||M<1||M>q->L)return 2;long n=(long)q->B*q->D*q->N;fl<<<(n+127)/128,128>>>(*q,M);return cudaDeviceSynchronize()==cudaSuccess?1:3;}
extern "C" int bl(void*v,const float*d,const float*x,const float*b,const float*c,float*y,int T,int start){Q*q=(Q*)v;if(!q||!d||!x||!b||!c||!y||T<1||start<0||start+T>q->L)return 2;size_t bd=(size_t)q->B*q->D*4,bn=(size_t)q->B*q->N*4,yo=(size_t)q->B*q->D;for(int i=0;i<T;i++){int p=start+i;if(cudaMemcpy(q->d+(size_t)p*q->B*q->D,d+(size_t)i*q->B*q->D,bd,cudaMemcpyHostToDevice)||cudaMemcpy(q->x+(size_t)p*q->B*q->D,x+(size_t)i*q->B*q->D,bd,cudaMemcpyHostToDevice)||cudaMemcpy(q->b+(size_t)p*q->B*q->N,b+(size_t)i*q->B*q->N,bn,cudaMemcpyHostToDevice)||cudaMemcpy(q->c,c+(size_t)i*q->B*q->N,bn,cudaMemcpyHostToDevice))return 3;out<<<(q->B*q->D+127)/128,128>>>(*q,p+1);if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(y+(size_t)i*yo,q->y,bd,cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;}return 1;}
extern "C" int as(void*v,const float*d,const float*x,const float*b,const float*c,int T,int start){Q*q=(Q*)v;if(!q||q->pending||T<1||start<0||start+T>q->L)return 2;size_t bd=(size_t)q->B*q->D*4,bn=(size_t)q->B*q->N*4,yo=(size_t)q->B*q->D;memcpy(q->pd,d,(size_t)T*bd);memcpy(q->px,x,(size_t)T*bd);memcpy(q->pb,b,(size_t)T*bn);memcpy(q->pc,c,(size_t)T*bn);for(int i=0;i<T;i++){int p=start+i;if(cudaMemcpyAsync(q->d+(size_t)p*yo,q->pd+(size_t)i*yo,bd,cudaMemcpyHostToDevice,q->st)||cudaMemcpyAsync(q->x+(size_t)p*yo,q->px+(size_t)i*yo,bd,cudaMemcpyHostToDevice,q->st)||cudaMemcpyAsync(q->b+(size_t)p*q->B*q->N,q->pb+(size_t)i*q->B*q->N,bn,cudaMemcpyHostToDevice,q->st)||cudaMemcpyAsync(q->c,q->pc+(size_t)i*q->B*q->N,bn,cudaMemcpyHostToDevice,q->st))return 3;out<<<(q->B*q->D+127)/128,128,0,q->st>>>(*q,p+1);if(cudaMemcpyAsync(q->dy+(size_t)i*yo,q->y,bd,cudaMemcpyDeviceToDevice,q->st)||cudaMemcpyAsync(q->py+(size_t)i*yo,q->y,bd,cudaMemcpyDeviceToHost,q->st))return 3;}q->pending=T;return cudaEventRecord(q->ev,q->st)==cudaSuccess?1:3;}
extern "C" int aw(void*v,float*y,int T){Q*q=(Q*)v;if(!q||!q->pending||T!=q->pending)return 2;if(cudaEventSynchronize(q->ev)!=cudaSuccess)return 3;memcpy(y,q->py,(size_t)T*q->B*q->D*4);q->pending=0;return 1;}
extern "C" void dl(void*v){Q*q=(Q*)v;if(!q)return;cudaFree(q->d);cudaFree(q->x);cudaFree(q->b);cudaFree(q->s);cudaFree(q->c);cudaFree(q->a);cudaFree(q->y);cudaFree(q->dy);if(q->pd)cudaFreeHost(q->pd);if(q->px)cudaFreeHost(q->px);if(q->pb)cudaFreeHost(q->pb);if(q->pc)cudaFreeHost(q->pc);if(q->py)cudaFreeHost(q->py);if(q->ev)cudaEventDestroy(q->ev);if(q->st)cudaStreamDestroy(q->st);free(q);}'''


class NvidiaReplayDeviceState:
    """CUDA-owned S0 and replay ring; only current token inputs cross PCIe."""
    def __init__(self, s0: Any, a: Any, capacity: int):
        import numpy as np
        global _ssm_replay_device_artifact
        s0=np.ascontiguousarray(s0,np.float32); a=np.ascontiguousarray(a,np.float32)
        self.B,self.D,self.N=s0.shape; self.capacity=capacity
        if _ssm_replay_device_artifact is None: _ssm_replay_device_artifact=_nvidia_cuda_compile_fn(KernelSource(source=_synthesize_ssm_replay_device_cuda(),entry="cr",lang=_LANG,spec=SpecPolicy.DYNAMIC,shape_key=("ssm-replay-device",)))
        self.lib=_load_lib(_ssm_replay_device_artifact); self.ctx=ctypes.c_void_p(); f=self.lib.cr; f.restype=ctypes.c_int
        if f(ctypes.byref(self.ctx),_ptr(s0),_ptr(a),self.B,self.D,self.N,capacity)!=1: raise RuntimeError("ReplaySSM CUDA allocation failed")
    def append(self,d: Any,x: Any,b: Any,i:int)->None:
        import numpy as np
        f=self.lib.ap; f.restype=ctypes.c_int; vs=[np.ascontiguousarray(v,np.float32) for v in(d,x,b)]
        if f(self.ctx,*(_ptr(v) for v in vs),i)!=1: raise RuntimeError("ReplaySSM CUDA append failed")
    def decode(self,c: Any,m:int)->Any:
        import numpy as np
        c=np.ascontiguousarray(c,np.float32); y=np.empty((self.B,self.D),np.float32); f=self.lib.de; f.restype=ctypes.c_int
        if f(self.ctx,_ptr(c),_ptr(y),m)!=1: raise RuntimeError("ReplaySSM CUDA decode failed")
        return y
    def flush(self,m:int)->None:
        f=self.lib.fu; f.restype=ctypes.c_int
        if f(self.ctx,m)!=1: raise RuntimeError("ReplaySSM CUDA flush failed")
    def submit_block(self,d: Any,x: Any,b: Any,c: Any,start:int)->Any:
        import numpy as np
        d,x,b,c=(np.ascontiguousarray(v,np.float32) for v in (d,x,b,c)); T=d.shape[0]
        y=np.empty((T,self.B,self.D),np.float32); f=self.lib.bl; f.restype=ctypes.c_int
        if f(self.ctx,_ptr(d),_ptr(x),_ptr(b),_ptr(c),_ptr(y),T,start)!=1: raise RuntimeError("ReplaySSM CUDA block submit failed")
        return y
    def submit_block_async(self,d: Any,x: Any,b: Any,c: Any,start:int)->"NvidiaReplayAsyncResult":
        import numpy as np
        d,x,b,c=(np.ascontiguousarray(v,np.float32) for v in (d,x,b,c)); T=d.shape[0]
        f=getattr(self.lib, "as"); f.restype=ctypes.c_int
        if f(self.ctx,_ptr(d),_ptr(x),_ptr(b),_ptr(c),T,start)!=1: raise RuntimeError("ReplaySSM CUDA async submit failed")
        return NvidiaReplayAsyncResult(self,T)
    def close(self)->None:
        if getattr(self,"ctx",None) and self.ctx.value: self.lib.dl(self.ctx); self.ctx=ctypes.c_void_p()
    def __del__(self)->None: self.close()


class CudaEvent:
    """Opaque completion token owned by a CUDA result; no raw event escapes."""
    def __init__(self, result: "NvidiaReplayAsyncResult"):
        self._result = result
    def wait(self) -> None:
        self._result.wait()


class CudaDeviceBuffer:
    """Opaque lease of a CUDA output range, valid until its result is consumed."""
    def __init__(self, result: "NvidiaReplayAsyncResult", shape: tuple[int, ...]):
        self._result, self.shape, self.dtype = result, shape, "float32"
    def numpy(self) -> Any:
        return self._result.wait()


class NvidiaReplayAsyncResult:
    """One in-flight ReplaySSM submission; call :meth:`wait` for host output."""
    def __init__(self, state: NvidiaReplayDeviceState, tokens: int):
        self._state, self._tokens, self._done = state, tokens, False
        self.event = CudaEvent(self)
        self.device_buffer = CudaDeviceBuffer(self, (tokens, state.B, state.D))
    def wait(self) -> Any:
        import numpy as np
        if self._done: raise RuntimeError("ReplaySSM async result already consumed")
        out=np.empty((self._tokens,self._state.B,self._state.D),np.float32); f=self._state.lib.aw; f.restype=ctypes.c_int
        if f(self._state.ctx,_ptr(out),self._tokens)!=1: raise RuntimeError("ReplaySSM CUDA async wait failed")
        self._done=True
        return out


def _synthesize_posenc_cuda() -> str:
    return r'''#include <cuda_runtime.h>
#include <math.h>
__global__ void rope_k(const float*x,const float*t,float*o,long n){long p=(long)blockIdx.x*blockDim.x+threadIdx.x;if(p<n/2){long e=2*p;float a=t[e],c=cosf(a),s=sinf(a),xe=x[e],xo=x[e+1];o[e]=xe*c-xo*s;o[e+1]=xe*s+xo*c;}}
__global__ void alibi_k(const float*s,float*o,long H,long S){long x=(long)blockIdx.x*blockDim.x+threadIdx.x,n=H*S*S;if(x<n){long j=x%S,i=(x/S)%S,h=x/(S*S);o[x]=s[h]*(float)(j-i);}}
static int ac(float**d,const float*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_rope_f32(const float*x,const float*t,float*o,long n){float *dx=0,*dt=0,*d=0;size_t z=n*4;if(!ac(&dx,x,z)||!ac(&dt,t,z)||cudaMalloc(&d,z)!=cudaSuccess)return 3;rope_k<<<(n/2+255)/256,256>>>(dx,dt,d,n);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(dt);cudaFree(d);return ok?1:3;}
extern "C" int tessera_nvidia_alibi_f32(const float*s,float*o,long H,long S){float *ds=0,*d=0;size_t n=H*S*S,z=n*4;if(!ac(&ds,s,H*4)||cudaMalloc(&d,z)!=cudaSuccess)return 3;alibi_k<<<(n+255)/256,256>>>(ds,d,H,S);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(ds);cudaFree(d);return ok?1:3;}
'''


def _posenc_lib():
    global _posenc_artifact
    if _posenc_artifact is None:
        _posenc_artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_synthesize_posenc_cuda(), entry=_ROPE_ENTRY, lang=_LANG,
            spec=SpecPolicy.DYNAMIC, shape_key=("posenc-f32",)))
    return _load_lib(_posenc_artifact)


def run_rope_f32(x: Any, theta: Any) -> Any:
    import numpy as np
    x = np.ascontiguousarray(x); theta = np.ascontiguousarray(theta)
    if x.dtype != np.float32 or theta.dtype != np.float32:
        raise ValueError("NVIDIA RoPE v1 requires f32 x/theta")
    if x.shape != theta.shape or x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("NVIDIA RoPE requires matching shapes and even final dimension")
    out = np.empty_like(x); fn = getattr(_posenc_lib(), _ROPE_ENTRY)
    fn.restype = ctypes.c_int; fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long]
    if fn(_ptr(x), _ptr(theta), _ptr(out), x.size) != 1:
        raise RuntimeError("NVIDIA RoPE CUDA launch failed")
    return out


def run_alibi_f32(num_heads: int, seq_len: int, slopes: Any = None) -> Any:
    import numpy as np
    if num_heads <= 0 or seq_len <= 0:
        raise ValueError("NVIDIA ALiBi requires positive num_heads/seq_len")
    if slopes is None:
        k = np.arange(1, num_heads + 1, dtype=np.float32)
        slopes = np.float32(2.0) ** (-np.float32(8.0) * k / num_heads)
    slopes = np.ascontiguousarray(slopes, dtype=np.float32).reshape(-1)
    if slopes.size != num_heads:
        raise ValueError(f"ALiBi slopes must have length {num_heads}")
    out = np.empty((num_heads, seq_len, seq_len), np.float32)
    fn = getattr(_posenc_lib(), _ALIBI_ENTRY); fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
    if fn(_ptr(slopes), _ptr(out), num_heads, seq_len) != 1:
        raise RuntimeError("NVIDIA ALiBi CUDA launch failed")
    return out


def _synthesize_control_flow_cuda() -> str:
    """Emit one-launch rank-1 f32 kernels for the four typed control ops."""
    return r'''#include <cuda_runtime.h>
__global__ void cf_for(const float*x,float*o,long n,int trip,float a,float b){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float c=x[i];for(int t=0;t<trip;++t)c=a*c+b;o[i]=c;}}
__global__ void cf_if(const float*f,const float*x,float*o,long n,float ta,float tb,float ea,float eb){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=f[0]>0.f?ta*x[i]+tb:ea*x[i]+eb;}
__global__ void cf_while(const float*x,float*o,long n,int m,float limit,float a,float b){__shared__ int active;long i=threadIdx.x;if(i<n)o[i]=x[i];__syncthreads();for(int t=0;t<m;++t){if(i==0)active=o[0]<limit;__syncthreads();if(!active)break;if(i<n)o[i]=a*o[i]+b;__syncthreads();}}
__global__ void cf_scan(const float*init,const float*xs,float*c,float*ys,long n,int trip,float a){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=init[i];for(int t=0;t<trip;++t){v=a*v+xs[(long)t*n+i];ys[(long)t*n+i]=v;}c[i]=v;}}
static int ac(float**d,const float*h,size_t z){return cudaMalloc(d,z)==cudaSuccess&&cudaMemcpy(*d,h,z,cudaMemcpyHostToDevice)==cudaSuccess;}
extern "C" int tessera_nvidia_control_for_f32(const float*x,float*o,long n,int t,float a,float b){float *dx=0,*d=0;size_t z=n*4;if(!ac(&dx,x,z)||cudaMalloc(&d,z)!=cudaSuccess)return 3;cf_for<<<(n+255)/256,256>>>(dx,d,n,t,a,b);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(d);return ok?1:3;}
extern "C" int tessera_nvidia_control_if_f32(const float*f,const float*x,float*o,long n,float ta,float tb,float ea,float eb){float *df=0,*dx=0,*d=0;size_t z=n*4;if(!ac(&df,f,4)||!ac(&dx,x,z)||cudaMalloc(&d,z)!=cudaSuccess)return 3;cf_if<<<(n+255)/256,256>>>(df,dx,d,n,ta,tb,ea,eb);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(df);cudaFree(dx);cudaFree(d);return ok?1:3;}
extern "C" int tessera_nvidia_control_while_f32(const float*x,float*o,long n,int m,float l,float a,float b){if(n<1||n>1024)return 2;float *dx=0,*d=0;size_t z=n*4;if(!ac(&dx,x,z)||cudaMalloc(&d,z)!=cudaSuccess)return 3;cf_while<<<1,n>>>(dx,d,n,m,l,a,b);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(dx);cudaFree(d);return ok?1:3;}
extern "C" int tessera_nvidia_control_scan_f32(const float*i,const float*x,float*c,float*y,long n,int t,float a){float *di=0,*dx=0,*dc=0,*dy=0;size_t z=n*4,zs=z*t;if(!ac(&di,i,z)||!ac(&dx,x,zs)||cudaMalloc(&dc,z)!=cudaSuccess||cudaMalloc(&dy,zs)!=cudaSuccess)return 3;cf_scan<<<(n+255)/256,256>>>(di,dx,dc,dy,n,t,a);int ok=cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(c,dc,z,cudaMemcpyDeviceToHost)==cudaSuccess&&cudaMemcpy(y,dy,zs,cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(di);cudaFree(dx);cudaFree(dc);cudaFree(dy);return ok?1:3;}
'''


def _control_flow_lib():
    global _control_flow_artifact
    if _control_flow_artifact is None:
        _control_flow_artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_synthesize_control_flow_cuda(), entry=_CONTROL_FOR_ENTRY,
            lang=_LANG, spec=SpecPolicy.DYNAMIC,
            shape_key=("bounded-control-flow-f32",)))
    return _load_lib(_control_flow_artifact)


def _f32_vector(x: Any, label: str):
    import numpy as np
    a = np.ascontiguousarray(x)
    if a.dtype != np.float32 or a.ndim != 1 or not a.size:
        raise ValueError(f"NVIDIA {label} requires a non-empty rank-1 f32 tensor")
    return a


def run_control_for_f32(x: Any, *, trip: int, alpha: float = 1.0,
                        beta: float = 0.0) -> Any:
    import numpy as np
    x = _f32_vector(x, "control_for")
    if trip < 0: raise ValueError("control_for trip must be non-negative")
    out = np.empty_like(x); fn = getattr(_control_flow_lib(), _CONTROL_FOR_ENTRY)
    fn.restype = ctypes.c_int; fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_long, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    if fn(_ptr(x), _ptr(out), x.size, trip, alpha, beta) != 1: raise RuntimeError("NVIDIA control_for CUDA launch failed")
    return out


def run_control_if_f32(flag: Any, x: Any, *, then_alpha: float = 1.0,
                       then_beta: float = 0.0, else_alpha: float = 1.0,
                       else_beta: float = 0.0) -> Any:
    import numpy as np
    flag = _f32_vector(flag, "control_if flag"); x = _f32_vector(x, "control_if"); out = np.empty_like(x)
    fn = getattr(_control_flow_lib(), _CONTROL_IF_ENTRY); fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long] + [ctypes.c_float] * 4
    if fn(_ptr(flag), _ptr(x), _ptr(out), x.size, then_alpha, then_beta, else_alpha, else_beta) != 1: raise RuntimeError("NVIDIA control_if CUDA launch failed")
    return out


def run_control_while_f32(x: Any, *, max_iters: int, limit: float,
                          alpha: float = 1.0, beta: float = 0.0) -> Any:
    import numpy as np
    x = _f32_vector(x, "control_while")
    if max_iters <= 0 or x.size > 1024: raise ValueError("control_while requires max_iters>0 and N<=1024")
    out = np.empty_like(x); fn = getattr(_control_flow_lib(), _CONTROL_WHILE_ENTRY); fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_long, ctypes.c_int] + [ctypes.c_float] * 3
    if fn(_ptr(x), _ptr(out), x.size, max_iters, limit, alpha, beta) != 1: raise RuntimeError("NVIDIA control_while CUDA launch failed")
    return out


def run_control_scan_f32(init: Any, xs: Any, *, alpha: float = 1.0):
    import numpy as np
    init = _f32_vector(init, "control_scan init"); xs = np.ascontiguousarray(xs)
    if xs.dtype != np.float32 or xs.ndim != 2 or xs.shape[1] != init.size or not xs.shape[0]: raise ValueError("control_scan xs must be non-empty [trip,N] f32")
    carry = np.empty_like(init); ys = np.empty_like(xs); fn = getattr(_control_flow_lib(), _CONTROL_SCAN_ENTRY); fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_long, ctypes.c_int, ctypes.c_float]
    if fn(_ptr(init), _ptr(xs), _ptr(carry), _ptr(ys), init.size, xs.shape[0], alpha) != 1: raise RuntimeError("NVIDIA control_scan CUDA launch failed")
    return carry, ys


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
_mma_fused_fn_cache: dict[tuple[str, bool, str | None], Any] = {}


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


def _synthesize_mma_fused_cuda(has_bias: bool, act: str | None,
                               storage: str = "f16") -> str:
    """CUDA source for a warp-tiled ``mma.sync.m16n8k16`` f16 GEMM (row-major A/B,
    f32 accumulate) with a fused ``bias? + activation?`` epilogue. One warp per
    16x8 output tile, K-looped, boundary-checked (same fragment layout as the
    shipped tiled kernel). f16 (10-bit mantissa) keeps the rounding vs the f32
    reference inside the 5e-3 budget where bf16 (7-bit) would not."""
    from tessera.compiler.emit._fused_scalar_body import pointwise_snippet
    if storage not in ("f16", "bf16"):
        raise ValueError(f"unsupported tensor-core storage {storage}")
    mma_type = "f16" if storage == "f16" else "bf16"
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
        f"    asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.{mma_type}.{mma_type}.f32 \"\n"
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


def _mma_fused_fn(has_bias: bool, act: str | None, storage: str = "f16"):
    """Compile (once per epilogue signature) the mma.sync fused kernel and return
    its bound entry symbol: ``int(A bf16, B bf16, bias f32|NULL, D f32, M,N,K)``."""
    sig = (storage, has_bias, act)
    fn = _mma_fused_fn_cache.get(sig)
    if fn is not None:
        return fn
    from tessera.compiler.emit.kernel_emitter import KernelSource
    src = KernelSource(source=_synthesize_mma_fused_cuda(has_bias, act, storage),
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

    tier = Tier.EMITTED
    target = _TARGET
    op = OP_FUSED_REGION
    def __init__(self, storage: str = "f16") -> None:
        self.storage = storage
        self.name = ("nvidia_mma_fused" if storage == "f16"
                     else "nvidia_mma_fused_bf16")
        self.accuracy_atol = (_GEMM_F16_ATOL if storage == "f16"
                              else _GEMM_BF16_ATOL)

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return rt._nvidia_mma_runtime_available()
        except Exception:
            return False

    def applies_to(self, region: Any) -> bool:
        return (_mma_fused_epilogue(region) is not None
                and getattr(region, "storage_dtype", "f16") == self.storage)

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
            storage_dtype = np.float16
            if self.storage == "bf16":
                import ml_dtypes
                storage_dtype = ml_dtypes.bfloat16
            Ab = np.ascontiguousarray(Aa, storage_dtype)
            Bb = np.ascontiguousarray(Ba, storage_dtype)
            M, K = Ab.shape
            _, N = Bb.shape
            bias_arr = (np.ascontiguousarray(bias, np.float32)
                        if has_bias else None)
            out = np.zeros((M, N), np.float32)
            fn = _mma_fused_fn(has_bias, act, self.storage)
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
_mma_attn_fn_cache: dict[str, Any] = {}


def _synthesize_mma_attn_cuda(storage: str = "f16") -> str:
    """CUDA source for ``O = softmax(scale·Q·Kᵀ)·V`` via two ``mma.sync.m16n8k16``
    f16 matmuls with the row softmax staged in shared memory (16xNk f32 scores).
    Natural Q(M,D)/K(Nk,D)/V(Nk,Dv), row-major; boundary-checked; causal masks
    keys ``n > m``. One warp per 16-query block."""
    if storage not in ("f16", "bf16"):
        raise ValueError(f"unsupported tensor-core storage {storage}")
    e = _MMA_ATTN_ENTRY
    mma_type = "f16" if storage == "f16" else "bf16"
    probability_cast = ("__half_as_ushort(__float2half" if storage == "f16"
                        else "__bfloat16_as_ushort(__float2bfloat16")
    MMA = (f'asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.{mma_type}.{mma_type}.f32 "\n'
           '        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\\n"\n'
           '        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),'
           '"r"(a3),"r"(b0),"r"(b1));')
    return (
        "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <math.h>\n"
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
        f"        unsigned short lo=(r<16&&cc<Nk)?{probability_cast}(S[r*Nk+cc])):(unsigned short)0;\n"
        f"        unsigned short hi=(r<16&&cc+1<Nk)?{probability_cast}(S[r*Nk+cc+1])):(unsigned short)0;\n"
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


def _mma_attn_fn(storage: str = "f16"):
    """Compile (once) the mma.sync attention kernel and return its bound entry:
    ``int(Q f16, K f16, V f16, O f32, M,Nk,D,Dv, scale, causal)``."""
    if storage in _mma_attn_fn_cache:
        return _mma_attn_fn_cache[storage]
    from tessera.compiler.emit.kernel_emitter import KernelSource
    src = KernelSource(source=_synthesize_mma_attn_cuda(storage), entry=_MMA_ATTN_ENTRY,
                       lang=_LANG)
    fn = getattr(_load_lib(_nvidia_cuda_compile_fn(src)), _MMA_ATTN_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
                   + [ctypes.c_float, ctypes.c_int])
    _mma_attn_fn_cache[storage] = fn
    return fn


class NvidiaMmaAttnCandidate(Candidate):
    """Tier-2 (emitted): the tensor-core flash-attention lane — two ``mma.sync``
    matmuls with a smem-staged row softmax, f16 operands / f32 accumulate. Serves
    any ``AttentionRegion``; for a KV length past the smem cap it delegates to the
    scalar flash lane (which streams any length), so it never declines a shape the
    scalar lane can run. f16 accuracy budget."""

    tier = Tier.EMITTED
    target = _TARGET
    op = OP_ATTENTION
    def __init__(self, storage: str = "f16") -> None:
        self.storage = storage
        self.name = ("nvidia_mma_attn" if storage == "f16"
                     else "nvidia_mma_attn_bf16")
        self.accuracy_atol = (5e-3 if storage == "f16" else 5e-2)

    def applies_to(self, region: Any) -> bool:
        return getattr(region, "storage_dtype", "f16") == self.storage

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
            storage_dtype = np.float16
            if self.storage == "bf16":
                import ml_dtypes
                storage_dtype = ml_dtypes.bfloat16
            Qh = np.ascontiguousarray(Qn, storage_dtype)
            Kh = np.ascontiguousarray(Kn, storage_dtype)
            Vh = np.ascontiguousarray(Vf, storage_dtype)
            out = np.zeros((M, Dv), np.float32)
            rc = _mma_attn_fn(self.storage)(_ptr(Qh), _ptr(Kh), _ptr(Vh), _ptr(out),
                                M, Nk, D, Dv, ctypes.c_float(float(region.scale)),
                                1 if region.causal else 0)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
        return region.reference(Q, K, V), "reference"


# ── tensor-core GATED lane — paired mma.sync projections + gate epilogue ─────

_MMA_GATED_ENTRY = "tessera_nvidia_mma_gated"
_mma_gated_fn_cache: dict[tuple[str, str], Any] = {}


def _synthesize_mma_gated_cuda(storage: str, act: str) -> str:
    """Two tensor-core projections sharing A, followed by gate(AWg) * AWu."""
    from tessera.compiler.emit._fused_scalar_body import pointwise_snippet
    if storage not in ("f16", "bf16"):
        raise ValueError(f"unsupported tensor-core storage {storage}")
    mma_type = "f16" if storage == "f16" else "bf16"
    e = _MMA_GATED_ENTRY
    gate = "".join(f"    {pointwise_snippet(act, f'g{i}')}\n" for i in range(4))
    mma = (f'asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.{mma_type}.{mma_type}.f32 "\n'
           '      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\\n"')
    return (
        "#include <cuda_runtime.h>\n#include <math.h>\n"
        f"extern \"C\" __global__ void {e}_kernel(const unsigned short* A,\n"
        "    const unsigned short* Wg, const unsigned short* Wu, float* O,\n"
        "    int M, int H, int K) {\n"
        "  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;\n"
        "  float g0=0,g1=0,g2=0,g3=0,u0=0,u1=0,u2=0,u3=0;\n"
        "  for(int k0=0;k0<K;k0+=16){\n"
        "    auto la=[&](int r,int c)->unsigned{int rr=mt+r,cc=k0+c;\n"
        "      unsigned lo=(rr<M&&cc<K)?A[rr*K+cc]:0u, hi=(rr<M&&cc+1<K)?A[rr*K+cc+1]:0u; return (hi<<16)|lo;};\n"
        "    auto lw=[&](const unsigned short* W,int r,int c)->unsigned{int rr=k0+r,cc=nt+c;\n"
        "      unsigned lo=(rr<K&&cc<H)?W[rr*H+cc]:0u, hi=(rr+1<K&&cc<H)?W[(rr+1)*H+cc]:0u; return (hi<<16)|lo;};\n"
        "    unsigned a0=la(gid,2*tig),a1=la(gid+8,2*tig),a2=la(gid,2*tig+8),a3=la(gid+8,2*tig+8);\n"
        "    unsigned bg0=lw(Wg,2*tig,gid),bg1=lw(Wg,2*tig+8,gid);\n"
        f"    {mma}\n"
        "      :\"+f\"(g0),\"+f\"(g1),\"+f\"(g2),\"+f\"(g3):\"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(bg0),\"r\"(bg1));\n"
        "    unsigned bu0=lw(Wu,2*tig,gid),bu1=lw(Wu,2*tig+8,gid);\n"
        f"    {mma}\n"
        "      :\"+f\"(u0),\"+f\"(u1),\"+f\"(u2),\"+f\"(u3):\"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(bu0),\"r\"(bu1));\n"
        "  }\n"
        f"{gate}"
        "  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<H)O[rr*H+cc]=v;};\n"
        "  st(gid,2*tig,g0*u0);st(gid,2*tig+1,g1*u1);st(gid+8,2*tig,g2*u2);st(gid+8,2*tig+1,g3*u3);\n"
        "}\n"
        f'extern "C" int {e}(const unsigned short* hA, const unsigned short* hWg,\n'
        "    const unsigned short* hWu, float* hO, int M, int H, int K) {\n"
        "  size_t szA=(size_t)M*K*2, szW=(size_t)K*H*2, szO=(size_t)M*H*4;\n"
        "  unsigned short *dA=0,*dWg=0,*dWu=0; float* dO=0;\n"
        "  if(cudaMalloc(&dA,szA)!=cudaSuccess) return 3;\n"
        "  if(cudaMalloc(&dWg,szW)!=cudaSuccess||cudaMalloc(&dWu,szW)!=cudaSuccess||cudaMalloc(&dO,szO)!=cudaSuccess){cudaFree(dA);cudaFree(dWg);cudaFree(dWu);cudaFree(dO);return 3;}\n"
        "  cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);cudaMemcpy(dWg,hWg,szW,cudaMemcpyHostToDevice);cudaMemcpy(dWu,hWu,szW,cudaMemcpyHostToDevice);\n"
        f"  {e}_kernel<<<dim3((M+15)/16,(H+7)/8),32>>>(dA,dWg,dWu,dO,M,H,K);\n"
        "  int ok=(cudaDeviceSynchronize()==cudaSuccess)?1:3;if(ok==1)cudaMemcpy(hO,dO,szO,cudaMemcpyDeviceToHost);\n"
        "  cudaFree(dA);cudaFree(dWg);cudaFree(dWu);cudaFree(dO);return ok;\n}"
    )


def _mma_gated_fn(storage: str, act: str):
    key = (storage, act)
    if key in _mma_gated_fn_cache:
        return _mma_gated_fn_cache[key]
    src = KernelSource(source=_synthesize_mma_gated_cuda(storage, act),
                       entry=_MMA_GATED_ENTRY, lang=_LANG)
    fn = getattr(_load_lib(_nvidia_cuda_compile_fn(src)), _MMA_GATED_ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    _mma_gated_fn_cache[key] = fn
    return fn


class NvidiaMmaGatedCandidate(Candidate):
    tier = Tier.EMITTED
    target = _TARGET
    op = OP_GATED_MATMUL

    def __init__(self, storage: str = "f16") -> None:
        self.storage = storage
        self.name = ("nvidia_mma_gated" if storage == "f16"
                     else "nvidia_mma_gated_bf16")
        self.accuracy_atol = (1e-2 if storage == "f16" else 1e-1)

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return rt._nvidia_mma_runtime_available()
        except Exception:
            return False

    def applies_to(self, region: Any) -> bool:
        return (isinstance(region, GatedMatmulRegion)
                and region.storage_dtype == self.storage)

    def run(self, region: Any, A: Any, Wg: Any, Wu: Any,
            *a: Any, **k: Any) -> tuple[Any, str]:
        import numpy as np
        Aa, Wga, Wua = np.asarray(A), np.asarray(Wg), np.asarray(Wu)
        if (Aa.ndim != 2 or Wga.ndim != 2 or Wua.ndim != 2
                or Aa.shape[1] != Wga.shape[0] or Wga.shape != Wua.shape):
            return region.reference(A, Wg, Wu), "reference"
        try:
            storage_dtype = np.float16
            if self.storage == "bf16":
                import ml_dtypes
                storage_dtype = ml_dtypes.bfloat16
            Ah = np.ascontiguousarray(Aa, storage_dtype)
            Wgh = np.ascontiguousarray(Wga, storage_dtype)
            Wuh = np.ascontiguousarray(Wua, storage_dtype)
            M, K = Ah.shape
            _, H = Wgh.shape
            out = np.zeros((M, H), np.float32)
            rc = _mma_gated_fn(self.storage, region.gate_act)(
                _ptr(Ah), _ptr(Wgh), _ptr(Wuh), _ptr(out), M, H, K)
            if rc == 1:
                return out, _REAL_TAG
        except Exception:
            pass
        return region.reference(A, Wg, Wu), "reference"


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
            # Orient raw operands per the region's transpose flags before the kernel
            # (the GEMM consumes natural A(M,K)/B(K,N)) — the transpose contract, as
            # the attention lane already does via AttentionRegion._natural.
            An, Bn = region._natural(A, B, cast=False)
            return rt._nvidia_mma_gemm_2d(An, Bn, region.dtype), "nvidia_mma_shipped"
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
        # Orient per the transpose flags first — alignment + the kernel both consume
        # the natural A(M,K)/B(K,N) operands, so a transposed raw operand is flipped
        # before the aligned-only check (the transpose contract).
        An, Bn = region._natural(A, B, cast=False)
        if not _aligned_2d(An, Bn):            # emitter is aligned-only (for now)
            return region.reference(A, B), "reference"
        try:
            from tessera import runtime as rt
            return rt._nvidia_ptx_gemm_2d(An, Bn, region.dtype), "nvidia_ptx_gemm"
        except Exception:
            return region.reference(A, B), "reference"


class NvidiaTileMatmulCandidate(Candidate):
    """Compiler-generated Tile→fragment GEMM schedule.

    The two instances deliberately remain separate candidates: D2 measures them
    under the same device/shape/dtype key and persists the faster schedule. This
    preserves the observed sm_120 crossover instead of baking one tile into all
    workloads.
    """

    tier = Tier.EMITTED
    target = _TARGET
    op = OP_MATMUL
    accuracy_atol = _GEMM_F16_ATOL

    def __init__(self, schedule: str) -> None:
        if schedule not in ("direct", "shared"):
            raise ValueError(f"unknown Tile matmul schedule {schedule!r}")
        self.schedule = schedule
        self.name = f"nvidia_tile_matmul_{schedule}"

    def available(self) -> bool:
        try:
            from tessera import runtime as rt
            return rt._nvidia_tile_runtime_available()
        except Exception:
            return False

    def applies_to(self, region: Any) -> bool:
        return isinstance(region, MatmulRegion) and region.dtype in _GEMM_DTYPES

    def run(self, region: Any, A: Any, B: Any, *a: Any,
            **k: Any) -> tuple[Any, str]:
        try:
            from tessera import runtime as rt
            An, Bn = region._natural(A, B, cast=False)
            out = rt._nvidia_tile_matmul_2d(
                An, Bn, region.dtype, self.schedule)
            return out, f"nvidia_tile_{self.schedule}"
        except Exception:
            return region.reference(A, B), "reference"

    def measure_device_latency(self, region: Any, *inputs: Any,
                               reps: int = 100,
                               warmup: int = 10) -> float | None:
        if len(inputs) != 2:
            return None
        A, B = inputs
        try:
            from tessera import runtime as rt
            An, Bn = region._natural(A, B, cast=False)
            return rt._nvidia_tile_matmul_device_latency(
                An, Bn, region.dtype, self.schedule, reps=reps, warmup=warmup)
        except Exception:
            return None


# ── registration (import side effect, exactly like rocm_hip / x86_llvm) ────────
register_emitter(NvidiaCudaEmitter())
register_compiler(_TARGET, _nvidia_cuda_compile_fn)
register_runner(NvidiaCudaRunner(), default=False)

register_candidate(NvidiaGenericCudaCandidate())
register_candidate(NvidiaMmaFusedCandidate("f16"))   # tensor-core fused GEMM+epi
register_candidate(NvidiaMmaFusedCandidate("bf16"))
register_candidate(NvidiaFlashAttnCandidate())        # C4: synthesized attention
register_candidate(NvidiaMmaAttnCandidate("f16"))    # tensor-core flash attention
register_candidate(NvidiaMmaAttnCandidate("bf16"))
register_candidate(NvidiaGatedCandidate())            # C5: SwiGLU gate
register_candidate(NvidiaMmaGatedCandidate("f16"))
register_candidate(NvidiaMmaGatedCandidate("bf16"))
register_candidate(NvidiaPointwiseCandidate())        # C5: pointwise DAG
# Bare-GEMM lanes: hand-tuned shipped (Tier 3) + compiler-emitted (Tier 2).
register_candidate(NvidiaMmaGemmShippedCandidate())
register_candidate(NvidiaMmaGemmEmittedCandidate())
register_candidate(NvidiaTileMatmulCandidate("direct"))
register_candidate(NvidiaTileMatmulCandidate("shared"))
