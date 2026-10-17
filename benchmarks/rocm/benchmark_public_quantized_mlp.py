"""Authored two-quantizer NVFP4 gate/up and down-projection Graph."""
from __future__ import annotations
import ml_dtypes
import numpy as np
import tessera as ts
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy


def make_mlp(m=3,width=32,k=48,n=19):
    groups=m*k//16;down_groups=m*width//16
    divisor=np.full((groups,1),448,np.float32);divisor.flags.writeable=False
    floor=np.full((groups,1),np.finfo(np.float32).tiny,np.float32);floor.flags.writeable=False
    down_divisor=np.full((down_groups,1),448,np.float32);down_divisor.flags.writeable=False
    down_floor=np.full((down_groups,1),np.finfo(np.float32).tiny,np.float32);down_floor.flags.writeable=False
    def mlp(x,packed,weight_scales,globals_,down_packed,down_weight_scales,down_globals):
        blocks=ts.ops.reshape(x,shape=(groups,16))
        wide=ts.ops.cast(blocks,'fp32')
        absolute=ts.ops.absolute(wide)
        peak=ts.ops.reduce(absolute,op='max',axis=1,keepdims=True)
        raw=ts.ops.div(peak,divisor)
        scale=ts.ops.maximum(raw,floor)
        expanded=ts.ops.expand(scale,shape=(groups,16))
        normalized=ts.ops.div(wide,expanded)
        encoded=ts.ops.cast(normalized,'fp8_e4m3')
        codes=ts.ops.reshape(encoded,shape=(m,k))
        scales=ts.ops.reshape(scale,shape=(m,k//16))
        b,sb=ts.ops.nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,width,2*width],numeric_policy=nvfp4_expansion_policy())
        accumulator=ts.ops.scaled_matmul(codes,b,scales,sb,transposeB=True,
            scale_layout={'granularity':'block','block':[1,16],'format':'fp32'},
            numeric_policy={'accum':'fp32','execution_mode':'exact_per_block'})
        parent=ts.ops.cast(accumulator,'bf16')
        left=ts.ops.slice(parent,start_indices=[0,0],slice_sizes=[m,width])
        right=ts.ops.slice(parent,start_indices=[0,width],slice_sizes=[m,width])
        left_f32=ts.ops.cast(left,'fp32')
        right_f32=ts.ops.cast(right,'fp32')
        gated=ts.ops.silu_mul(left_f32,right_f32)
        activation=ts.ops.cast(gated,'bf16')
        down_blocks=ts.ops.reshape(activation,shape=(down_groups,16))
        down_wide=ts.ops.cast(down_blocks,'fp32')
        down_absolute=ts.ops.absolute(down_wide)
        down_peak=ts.ops.reduce(down_absolute,op='max',axis=1,keepdims=True)
        down_raw=ts.ops.div(down_peak,down_divisor)
        down_scale=ts.ops.maximum(down_raw,down_floor)
        down_expanded=ts.ops.expand(down_scale,shape=(down_groups,16))
        down_normalized=ts.ops.div(down_wide,down_expanded)
        down_encoded=ts.ops.cast(down_normalized,'fp8_e4m3')
        down_codes=ts.ops.reshape(down_encoded,shape=(m,width))
        down_scales=ts.ops.reshape(down_scale,shape=(m,width//16))
        down_b,down_sb=ts.ops.nvfp4_expand(down_packed,down_weight_scales,down_globals,row_offsets=[0,n],numeric_policy=nvfp4_expansion_policy())
        down_accumulator=ts.ops.scaled_matmul(down_codes,down_b,down_scales,down_sb,transposeB=True,
            scale_layout={'granularity':'block','block':[1,16],'format':'fp32'},
            numeric_policy={'accum':'fp32','execution_mode':'exact_per_block'})
        return ts.ops.cast(down_accumulator,'bf16')
    return mlp


def inputs(m,width,k,n,seed=731):
    from benchmarks.rocm.benchmark_public_quantized_gate_up import inputs as gate_inputs
    roots=gate_inputs(m,width,k,seed)
    rng=np.random.default_rng(seed+1)
    return roots+[rng.integers(0,256,size=(n,width//2),dtype=np.uint8),
        rng.uniform(.125,2,size=(n,width//16)).astype(ml_dtypes.float8_e4m3fn),
        np.array([.0123456789],np.float64)]
