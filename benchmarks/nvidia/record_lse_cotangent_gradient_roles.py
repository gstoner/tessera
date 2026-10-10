"""Compact/bias-gradient seeded package characterization via shared recorder."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from benchmarks.nvidia.record_lse_cotangent_package import run
from tests.device.nvidia.test_lse_cotangent_gradient_roles import fixture


def record(samples=5,reps=20):
    cases=[dict(mask=mask,launch=launch,threads=threads,causal=bool(mask&1))
           for mask in range(1,16) for launch in ['packed_v1','logical_v1'] for threads in [64,128]]
    cases += [dict(physical=physical,causal=causal)
              for physical in [(2,4,5,7),(1,1,1,7),(2,4,5,1)] for causal in [False,True]]
    packet=run(samples,reps,cases=cases,fixture_factory=fixture)
    packet['schema']='seeded_attention_gradient_roles_v1'
    import hashlib
    source=Path(__file__).relative_to(Path.cwd())
    packet['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    return packet


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=5);parser.add_argument('--reps',type=int,default=20)
    options=parser.parse_args()
    if options.samples<1 or options.reps<1:parser.error('positive samples and reps required')
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(record(options.samples,options.reps),indent=2)+'\n')
