"""Pinned full layer-0 MLP quality through the seven-member native Graph route.

This is a layer characterization, not full-model accuracy acceptance. References
use independent nibble decoding and FP64 products with explicit BF16 stores;
they omit activation quantization to expose its combined quality cost.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import ml_dtypes
import numpy as np

from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint
from benchmarks.rocm.benchmark_public_quantized_mlp import run
from benchmarks.rocm.checkpoint_activations import load_capture


def decode_projection(projection):
    packed = projection.packed_codes
    levels = np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float64)
    codes = np.empty((packed.shape[0],packed.shape[1]*2),np.uint8)
    codes[:,0::2] = packed & 15
    codes[:,1::2] = packed >> 4
    scales = projection.e4m3_scales.astype(np.float64)*projection.global_scale
    return (levels[codes].reshape(*scales.shape,16)*scales[...,None]).reshape(codes.shape)


def layer_reference(x,gate,up,down):
    if (x.ndim != 2 or gate.ndim != 2 or up.shape != gate.shape or down.ndim != 2
            or x.shape[1] != gate.shape[1] or down.shape[1] != gate.shape[0]):
        raise ValueError('inconsistent full MLP projection shapes')
    x64=x.astype(np.float64)
    parent_gate=(x64@gate.astype(np.float64).T).astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    parent_up=(x64@up.astype(np.float64).T).astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    # Stable SiLU oracle handles extreme negative inputs without exp overflow.
    sigmoid=np.empty_like(parent_gate)
    positive=parent_gate>=0
    sigmoid[positive]=1/(1+np.exp(-parent_gate[positive]))
    exp=np.exp(parent_gate[~positive])
    sigmoid[~positive]=exp/(1+exp)
    activation=(parent_gate*sigmoid*parent_up).astype(ml_dtypes.bfloat16)
    return (activation.astype(np.float64)@down.astype(np.float64).T).astype(np.float32).astype(ml_dtypes.bfloat16)


def measure(activations,rows,output_dir,repeats=1):
    if not rows or any(type(m) is not int or m<=0 for m in rows) or type(repeats) is not int or repeats<=0:
        raise ValueError('positive captured rows and repeats required')
    pool,capture=load_capture(activations,rows)
    loaded=[checkpoint._load_projection('model.layers.0.mlp.'+role+'.weight')
            for role in ('gate_proj','up_proj','down_proj')]
    projections=[item['projection'] for item in loaded]
    gate,up,down=projections
    h,k=gate.packed_codes.shape[0],gate.packed_codes.shape[1]*2
    n=down.packed_codes.shape[0]
    if up.packed_codes.shape!=gate.packed_codes.shape or down.packed_codes.shape[1]*2!=h or pool.shape[1]!=k:
        raise ValueError('checkpoint and captured activation geometry differ')
    fixed=[np.ascontiguousarray(np.concatenate([gate.packed_codes,up.packed_codes])),
           np.ascontiguousarray(np.concatenate([gate.e4m3_scales,up.e4m3_scales])),
           np.array([gate.global_scale,up.global_scale],np.float64),
           down.packed_codes,down.e4m3_scales,np.array([down.global_scale],np.float64)]
    decoded=[decode_projection(p) for p in projections]
    source={'checkpoints':[item['source'] for item in loaded],'activation_capture':capture,
            'reference_policy':'FP64 products; FP32 then BF16 parent/down stores; FP32 SiLU and BF16 activation; no activation quantization',
            'recorder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'loader_sha256':hashlib.sha256(Path(checkpoint.__file__).read_bytes()).hexdigest()}
    results=[]
    for m in rows:
        x=np.ascontiguousarray(pool[:m].astype(ml_dtypes.bfloat16))
        refs={'bf16_weights':layer_reference(x,*(item['bf16'] for item in loaded)),
              'shipped_nvfp4_weights':layer_reference(x,*decoded)}
        metadata={**source,'input_bf16_sha256':hashlib.sha256(x.tobytes()).hexdigest()}
        results.append(run(m,h,k,n,output_dir,repeats,supplied_roots=[x,*fixed],
            source_metadata=metadata,quality_references=refs,windows=3,member_repeats=8,event_repeats=5))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activations',type=Path,required=True)
    parser.add_argument('--rows',type=int,nargs='+',default=[1,16,128])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args()
    measure(args.activations,args.rows,args.output_dir,args.repeats)
