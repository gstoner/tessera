"""NVIDIA-LSE-1: numerical and timing evidence for the native dLSE physical leaf."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np
from tessera.compiler.nvidia_native import _compile_tile_ir
from tests.device.nvidia.test_lse_cotangent_native import execute,oracle,tile_fixture


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scheduled',action='store_true',help='compile authored Graph through native Schedule/Tile instead of the diagnostic Tile fixture')
    args=parser.parse_args()
    device=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,uuid,compute_cap,driver_version','--format=csv,noheader'],text=True).strip()
    if len(device.splitlines())!=1 or device.split(',')[2].strip()!='12.0':
        raise RuntimeError('requires one exact SM120 device')
    rows=[]
    for shape in [(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)]:
      for causal in [False,True]:
       for bias in [False,True]:
        b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(121203)
        q,k,v,do=[(rng.normal(size=s)*.2).astype(np.float32) for s in ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
        score_bias=(rng.normal(size=(b,hq,sq,sk))*.2).astype(np.float32) if bias else None
        for mode in ['legacy_zero_lse_seed','enabled_zero_lse_seed','mixed_seed']:
            print(shape,causal,bias,mode,flush=True)
            enabled=mode!='legacy_zero_lse_seed'
            seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32) if mode=='mixed_seed' else np.zeros((b,hq,sq),np.float32)
            output,lse,expected=oracle(q,k,v,do,seed,causal,score_bias)
            ancestry={}
            entry='test_lse_cotangent'
            if args.scheduled:
                from tests.unit.test_lse_cotangent_schedule import module
                from tessera.compiler.scheduled_checkpoint import lower_checkpoint_graph
                graph=module(bias,shape,causal)
                if not enabled:
                    fn=graph.functions[0];op=fn.body[0]
                    fn.args.pop();op.operands.pop();op.operand_types.pop();op.kwargs.pop('lse_cotangent')
                artifact=lower_checkpoint_graph(graph,backward=True)
                tile,entry=artifact.tile_ir,artifact.entry
                ancestry=dict(graph_ir_digest=hashlib.sha256(artifact.graph_ir.encode()).hexdigest(),
                              schedule_ir_digest=hashlib.sha256(artifact.schedule_ir.encode()).hexdigest(),
                              schedule_digest=artifact.schedule_digest)
            else:
                tile=tile_fixture(1/math.sqrt(d),causal,bias,lse_cotangent=enabled)
            target,ptx,*_=_compile_tile_ir(tile,entry)
            values=[do,q,k,v,output]+([score_bias] if bias else [])+[lse]+([seed] if enabled else [])
            result=execute(ptx,values,[x.shape for x in (expected[:3] if args.scheduled else expected)],shape,timed_expected=expected[:3] if args.scheduled else expected,entry=entry)
            rows.append(dict(shape=shape,causal=causal,bias=bias,mode=mode,
                compiler_ancestry=ancestry,entry=entry,tile_digest=hashlib.sha256(tile.encode()).hexdigest(),target_digest=hashlib.sha256(target.encode()).hexdigest(),
                image_digest=hashlib.sha256(ptx.encode()).hexdigest(),**result,
                device_event_median_ms=statistics.median(result['device_event_samples_ms'])))
    sources=['src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp',
             'src/compiler/ir/TileOps.cpp','src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td',
             'src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp',
             'tests/device/nvidia/test_lse_cotangent_native.py','benchmarks/nvidia/record_lse_cotangent_leaf.py',
             'src/compiler/programming_model/lib/NativeCheckpoint.h','src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td',
             'src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp','src/compiler/ir/TesseraOps.cpp','python/tessera/compiler/scheduled_checkpoint.py',
             'tests/unit/test_lse_cotangent_schedule.py']
    packet=dict(schema='tessera.nvidia.lse_cotangent_leaf.v1',work_item='NVIDIA-LSE-1',device=device,rows=rows,
                boundary=('authored typed Graph -> native Schedule/Tile -> NVIDIA Target/NVVM/PTX; checked package and multi-result AD integration pending' if args.scheduled else 'handwritten diagnostic Tile fixture -> native NVIDIA Target/NVVM/PTX; Graph/Schedule/package/AD integration pending'),
                core_compiler_sha256=hashlib.sha256(Path(os.environ['TESSERA_OPT']).read_bytes()).hexdigest(),
                source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
                compiler_sha256=hashlib.sha256(Path(os.environ['TESSERA_NVIDIA_OPT']).read_bytes()).hexdigest(),
                runtime_library_sha256=hashlib.sha256(Path(os.environ['TESSERA_NVIDIA_PTX_LAUNCH_LIB']).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(packet,indent=2)+'\n')
if __name__=='__main__':main()
