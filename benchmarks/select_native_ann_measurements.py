#!/usr/bin/env python3
"""Bind nine independent package reports to scoped native ANN arbitration."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_native_ann_execution import source  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_ann import prepare_native_ann  # noqa: E402
from tessera.compiler.native_ann_gpu import materialize_native_ann_gpu, NativeANNDeviceRegistration  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('reports',nargs=9,type=Path)
    args=parser.parse_args()
    reports=[json.loads(p.read_text()) for p in args.reports]
    first=reports[0]
    Device(first['backend'])
    os.environ['TESSERA_OPT']=str(args.compiler.resolve())
    logical=prepare_native_ann(source(first.get("rows",3),first.get("width",2),first.get("activation","relu")),allow_reassociation=True)
    pair=materialize_native_ann_gpu(logical,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),
        backend=first['backend'],chip=first['chip'],fuse_elementwise=first.get("fuse_elementwise",True),parallel_rows=first["parallel_rows"],tune_transformed=first.get("tune_transformed",False))
    shape=(first.get("rows",3),first.get("width",2))
    values=[np.zeros(shape,np.float32),np.ones(shape,np.float32),-np.ones(shape,np.float32)]
    with NativeANNDeviceRegistration(pair,values,input_bound=1.0,absolute_budget=.001) as registration:
        chosen,evidence=registration.select_from_measurements(reports)
        _,tag=chosen.run(registration.region,values[1])
        assert tag=='native_gpu'
    evidence['reports']=[p.name for p in args.reports]
    evidence['candidate_retired']=not chosen.available()
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
