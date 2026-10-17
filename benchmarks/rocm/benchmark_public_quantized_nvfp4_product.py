"""Authored BF16 quantizer -> NVFP4 weight expansion -> exact scaled product."""
from __future__ import annotations
import ml_dtypes
import numpy as np
import tessera as ts
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy, reference_nvfp4_expand


def make_product(m=3,n=17,k=32):
    groups=m*k//16
    divisor=np.full((groups,1),448,np.float32);divisor.flags.writeable=False
    floor=np.full((groups,1),np.finfo(np.float32).tiny,np.float32);floor.flags.writeable=False
    def product(x,packed,weight_scales,globals_):
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
        b,sb=ts.ops.nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,n],
            numeric_policy=nvfp4_expansion_policy())
        return ts.ops.scaled_matmul(codes,b,scales,sb,transposeB=True,
            scale_layout={"granularity":"block","block":[1,16],"format":"fp32"},
            numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"})
    return product


def inputs(m,n,k,seed=731):
    rng=np.random.default_rng(seed)
    return [rng.normal(size=(m,k)).astype(ml_dtypes.bfloat16),
            rng.integers(0,256,size=(n,k//2),dtype=np.uint8),
            rng.uniform(.125,2,size=(n,k//16)).astype(ml_dtypes.float8_e4m3fn),
            np.array([.0123456789],np.float64)]


def oracle(values):
    x,packed,weight_scales,globals_=values;m,k=x.shape;n=packed.shape[0]
    blocks=x.astype(np.float32).reshape(m,k//16,16)
    scales=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/scales[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    b,sb=reference_nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,n],numeric_policy=nvfp4_expansion_policy())
    return np.einsum("mgj,ngj,mg,gn->mn",q,b.astype(np.float64).reshape(n,k//16,16),scales.astype(np.float64),sb.astype(np.float64))
