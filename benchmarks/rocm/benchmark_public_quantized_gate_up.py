"""Authored native quantizer/weight expansion/BF16-store/gate-up Graph."""
from __future__ import annotations
import ml_dtypes
import numpy as np
import tessera as ts
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy,reference_nvfp4_expand


def make_gate_up(m=3,width=17,k=48,reverse=False):
    groups=m*k//16
    divisor=np.full((groups,1),448,np.float32);divisor.flags.writeable=False
    floor=np.full((groups,1),np.finfo(np.float32).tiny,np.float32);floor.flags.writeable=False
    def gate_up(x,packed,weight_scales,globals_):
        blocks=ts.ops.reshape(x,shape=(groups,16))
        wide=ts.ops.cast(blocks,"fp32")
        absolute=ts.ops.absolute(wide)
        peak=ts.ops.reduce(absolute,op="max",axis=1,keepdims=True)
        raw=ts.ops.div(peak,divisor)
        scale=ts.ops.maximum(raw,floor)
        expanded=ts.ops.expand(scale,shape=(groups,16))
        normalized=ts.ops.div(wide,expanded)
        encoded=ts.ops.cast(normalized,"fp8_e4m3")
        codes=ts.ops.reshape(encoded,shape=(m,k))
        scales=ts.ops.reshape(scale,shape=(m,k//16))
        b,sb=ts.ops.nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,width,2*width],
            numeric_policy=nvfp4_expansion_policy())
        accumulator=ts.ops.scaled_matmul(codes,b,scales,sb,transposeB=True,
            scale_layout={"granularity":"block","block":[1,16],"format":"fp32"},
            numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"})
        parent=ts.ops.cast(accumulator,"bf16")
        left=ts.ops.slice(parent,start_indices=[0,0],slice_sizes=[m,width])
        right=ts.ops.slice(parent,start_indices=[0,width],slice_sizes=[m,width])
        left_f32=ts.ops.cast(left,"fp32")
        right_f32=ts.ops.cast(right,"fp32")
        gated=ts.ops.silu_mul(right_f32,left_f32) if reverse else ts.ops.silu_mul(left_f32,right_f32)
        return ts.ops.cast(gated,"bf16")
    return gate_up


def inputs(m,width,k,seed=731):
    rng=np.random.default_rng(seed)
    return [rng.normal(size=(m,k)).astype(ml_dtypes.bfloat16),
            rng.integers(0,256,size=(2*width,k//2),dtype=np.uint8),
            rng.uniform(.125,2,size=(2*width,k//16)).astype(ml_dtypes.float8_e4m3fn),
            np.array([.0123456789,.007654321],np.float64)]


def oracle(values,reverse=False):
    x,packed,weight_scales,globals_=values;m,k=x.shape;width=packed.shape[0]//2
    blocks=x.astype(np.float32).reshape(m,k//16,16)
    sa=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/sa[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    b,sb=reference_nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,width,2*width],numeric_policy=nvfp4_expansion_policy())
    accum=np.einsum("mgj,ngj,mg,gn->mn",q,b.astype(np.float64).reshape(2*width,k//16,16),sa.astype(np.float64),sb.astype(np.float64))
    parent=accum.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    gate,up=(parent[:,width:],parent[:,:width]) if reverse else (parent[:,:width],parent[:,width:])
    return (gate/(np.float32(1)+np.exp(-gate))*up).astype(ml_dtypes.bfloat16)
