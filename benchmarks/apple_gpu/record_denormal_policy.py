"""Compile explicit denormal policy and measure status-returning Metal execution."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmarks.apple_gpu.denormal_inputs import operands, flush
from benchmarks.record_dtype_arithmetic import emit, oracle, check
from tessera.compiler.apple_native_arena import AppleNativeArena, materialize_apple_arena, build_apple_arena_package


def source(mode):
    return emit('fp32').replace('module {', f'module attributes {{tessera.denormal_mode = "{mode}"}} {{', 1)


def export(directory, compiler, llvm_bin):
    directory.mkdir(parents=True,exist_ok=True)
    for mode in ('gradual','flush_to_zero'):
        artifact = materialize_apple_arena(source(mode),compiler=compiler,llvm_bin=llvm_bin)
        (directory/f'{mode}.json').write_text(json.dumps(asdict(artifact)))


def record(directory):
    from tessera.runtime import DeviceTensor
    if not DeviceTensor.is_metal():
        raise RuntimeError('owning Metal device unavailable')
    rows = []
    a,b = operands()
    for mode in ('gradual','flush_to_zero'):
        artifact = AppleNativeArena(**json.loads((directory/f'{mode}.json').read_text()))
        if artifact.denormal_mode != mode:
            raise ValueError('native policy disagrees')
        package = build_apple_arena_package(artifact)
        buffers = []
        try:
            for array in (a,b,a,a,a,a):
                buf = DeviceTensor.from_numpy(array)
                if buf is None:
                    raise RuntimeError('Metal allocation failed')
                buffers.append(buf)
            with package.bind() as bound:
                bound.launch((*buffers,1),grid=(len(a),1,1),block=(1,1,1))
                mismatches = {}
                for op,buf in zip(('add','sub','mul','div'),buffers[2:],strict=True):
                    expected = oracle(flush(a),flush(b),op) if mode == 'flush_to_zero' else oracle(a,b,op)
                    if mode == 'flush_to_zero': expected = flush(expected)
                    mismatches[op] = check(buf.copy_to_host(),expected)
                interval = bound.last_device_interval()
            rows.append(dict(policy=mode,pairs=len(a),mismatches=mismatches,
                state='passed' if not any(mismatches.values()) else 'numerical_failure',
                artifact_digest=artifact.digest,compiler_sha256=artifact.compiler_digest,
                msl_sha256=hashlib.sha256(artifact.msl.encode()).hexdigest(),package_digest=package.binding_digest,
                bridge_sha256=package.bridge_digest,gpu_interval=interval,promotion_eligible=False))
        finally:
            for buf in buffers: buf.free()
    return dict(backend='apple',rows=rows,scope='explicit scalar f32 policy; no performance promotion')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts',required=True,type=Path)
    p.add_argument('--compiler',type=Path)
    p.add_argument('--llvm-bin',type=Path,default=Path('/usr/lib/llvm-23/bin'))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.compiler: export(args.artifacts,args.compiler,args.llvm_bin)
    else:
        result=record(args.artifacts)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        if any(row['state']!='passed' for row in result['rows']): raise SystemExit(1)
