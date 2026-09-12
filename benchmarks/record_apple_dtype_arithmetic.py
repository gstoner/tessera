#!/usr/bin/env python3
"""Compiler-owned MSL arithmetic proof; f32 buffers, no fallback or promotion."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmarks.record_dtype_arithmetic import emit, samples, oracle, check
from tessera.compiler.apple_native_arena import (
    AppleNativeArena, materialize_apple_arena, build_apple_arena_package,
)


def export(directory, compiler, llvm_bin):
    directory.mkdir(parents=True, exist_ok=True)
    for dtype in ('fp32', 'complex64'):
        artifact = materialize_apple_arena(emit(dtype), compiler=compiler, llvm_bin=llvm_bin)
        (directory / f'{dtype}.json').write_text(json.dumps(asdict(artifact)))


def record(directory):
    from tessera.runtime import DeviceTensor
    if not DeviceTensor.is_metal():
        raise RuntimeError('owning Metal device unavailable')
    rows = []
    for dtype in ('fp32', 'complex64'):
        artifact = AppleNativeArena(**json.loads((directory / f'{dtype}.json').read_text()))
        package = build_apple_arena_package(artifact)
        a, b = samples(dtype, 1)
        arrays = (a, b, a, a, a, a)
        buffers = []
        try:
            for value in arrays:
                buffers.append(DeviceTensor.from_numpy(value.view(np.float32)))
            if any(value is None for value in buffers):
                raise RuntimeError('Metal allocation failed')
            with package.bind() as bound:
                bound.launch((*buffers, 1), grid=(len(a), 1, 1), block=(1, 1, 1))
                errors, failures = {}, {}
                for op, buf in zip(('add','sub','mul','div'), buffers[2:], strict=True):
                    actual, expected = buf.copy_to_host().view(a.dtype), oracle(a,b,op)
                    errors[op] = check(actual, expected)
                    failures[op] = [dict(index=i, a=str(a[i]), b=str(b[i]),
                                        actual=str(actual[i]), expected=str(expected[i]))
                                    for i in range(len(a)) if check(actual[i:i+1], expected[i:i+1])]

                interval = bound.last_device_interval()
            rows.append(dict(dtype=dtype, state='passed' if not any(errors.values()) else 'numerical_failure',
                mismatches=errors, failures=failures, artifact_digest=artifact.digest, compiler_sha256=artifact.compiler_digest,
                msl_sha256=hashlib.sha256(artifact.msl.encode()).hexdigest(),
                package_digest=package.binding_digest, bridge_sha256=package.bridge_digest,
                gpu_interval=interval, promotion_eligible=False))
        finally:
            for buf in buffers:
                if buf is not None:
                    buf.free()
    return dict(backend='apple', scope='native status-returning MSL; f32 arithmetic and bounded interleaved complex components', rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', required=True, type=Path)
    p.add_argument('--compiler', type=Path)
    p.add_argument('--llvm-bin', type=Path, default=Path('/usr/lib/llvm-23/bin'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.compiler:
        export(args.artifacts, args.compiler, args.llvm_bin)
    else:
        if not args.output:
            p.error('--output required for owning-device execution')
        result = record(args.artifacts)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if any(row['state'] != 'passed' for row in result['rows']):
            raise SystemExit(1)
