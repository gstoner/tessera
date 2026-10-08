"""Separate producer/consumer timings for native O/LSE Graph differentiation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from benchmarks.nvidia.record_lse_cotangent_package import run
from tests.device.nvidia.test_multiresult_attention_ad import fixture


def record(samples=5,reps=20):
    cases=[dict(shape=shape,bias=bias,causal=causal,seed_mode=seed_mode,compact=compact,stage=stage)
           for shape in [(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)]
           for bias in [False,True] for causal in [False,True]
           for seed_mode in ['output_only','lse_only','mixed'] for compact in [False,True]
           for stage in ['forward','backward']]
    packet=run(samples,reps,cases=cases,fixture_factory=fixture)
    packet['schema']='native_multiresult_attention_ad_v1'
    for name in ['benchmarks/nvidia/record_multiresult_attention_ad.py',
                 'python/tessera/compiler/scheduled_checkpoint.py','src/compiler/ir/AttentionADContract.h',
                 'src/compiler/ir/AdjointInterface.cpp','src/transforms/lib/AutodiffPairedPass.cpp']:
        packet['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return packet


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=5);parser.add_argument('--reps',type=int,default=20)
    options=parser.parse_args()
    if options.samples<1 or options.reps<1:parser.error('positive samples and reps required')
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(record(options.samples,options.reps),indent=2)+'\n')
