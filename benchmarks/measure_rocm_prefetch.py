#!/usr/bin/env python3
"""Matched gfx1151 prefetch versus immediate-VMEM-drain experiment."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import random
import os
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_gpu_storage import build_native_gpu_storage  # noqa: E402


def immediate_wait(source):
    needle = '%loaded = llvm.load %ptr : !llvm.ptr<1> -> f32'
    if source.count(needle) != 1:
        raise ValueError('expected one next-generation load')
    return source.replace(needle, needle + '\n          llvm.inline_asm has_side_effects "s_waitcnt vmcnt(0)", "~{memory}" : () -> ()')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--launches', type=int, default=20)
    args = parser.parse_args()
    if args.trials < 1 or args.launches < 1:
        parser.error('trials and launches must be positive')
    rng = random.Random(args.seed)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_rocm_prefetch_device.mlir').read_text()
    device = Device('rocm')
    packages = {}
    for mode, text in [('prefetch', source), ('immediate_wait', immediate_wait(source)),
                       ('nonblocking_wait', immediate_wait(source).replace('vmcnt(0)', 'vmcnt(63)'))]:
        package = build_native_gpu_storage(text, compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'), backend='rocm', chip='gfx1151')
        packages[mode] = package
        path = args.artifacts / (mode + '.bin')
        path.write_bytes(package.image)
        (args.artifacts / (mode + '.mlir')).write_text(text)
        (args.artifacts / (mode + '.disasm')).write_text(subprocess.check_output(['/usr/lib/llvm-23/bin/llvm-objdump', '-d', '--mcpu=gfx1151', str(path)], text=True))
    resources = {}
    for mode in packages:
        resources[mode] = subprocess.check_output([
            "/usr/lib/llvm-23/bin/llvm-readobj", "--notes",
            str(args.artifacts / (mode + ".bin"))], text=True)
        (args.artifacts / (mode + ".resources.txt")).write_text(resources[mode])
    rows = []
    for blocks, width, rounds in [(32, 64, 7), (256, 256, 33), (256, 256, 65)]:
        inputs = (np.arange(rounds * blocks * width, dtype=np.float32) % 127).reshape(rounds, blocks, width)
        output = np.zeros((blocks, width), dtype=np.float32)
        expected = (np.roll(inputs, -1, axis=-1) * 2 + 1).sum(axis=0)
        src, dst = ct.c_void_p(), ct.c_void_p()
        device.check(device.alloc(ct.byref(src), inputs.nbytes))
        device.check(device.alloc(ct.byref(dst), output.nbytes))
        bounds = {name: p.bind() for name, p in packages.items()}
        try:
            device.check(device.copy(src, inputs.ctypes.data, inputs.nbytes, 1))
            values = [src, dst, ct.c_int64(width), ct.c_int64(rounds)]
            argv = (ct.c_void_p * 4)(*(ct.cast(ct.byref(v), ct.c_void_p) for v in values))
            def launch(bound):
                bound._check(bound._launch(bound._function, blocks, 1, 1, width, 1, 1, width * 4, None, argv, None))
            for bound in bounds.values():
                assert bound._size(*values) == width * 4
                launch(bound)
                device.check(device.sync())
                device.check(device.copy(output.ctypes.data, dst, output.nbytes, 2))
                np.testing.assert_array_equal(output, expected)
            samples = {name: [] for name in bounds}
            start, end = ct.c_void_p(), ct.c_void_p()
            device.check(device.event_create(ct.byref(start), 0))
            device.check(device.event_create(ct.byref(end), 0))
            try:
                for trial in range(args.trials):
                    names = list(bounds)
                    rng.shuffle(names)
                    for name in names:
                        launch(bounds[name])
                        device.check(device.event_record(start, None))
                        for _ in range(args.launches):
                            launch(bounds[name])
                        device.check(device.event_record(end, None))
                        device.check(device.event_sync(end))
                        elapsed = ct.c_float()
                        device.check(device.event_elapsed(ct.byref(elapsed), start, end))
                        samples[name].append(elapsed.value / args.launches)
            finally:
                device.check(device.event_destroy(start))
                device.check(device.event_destroy(end))
            rows.append(dict(blocks=blocks, width=width, rounds=rounds, oracle='exact', hip_event_ms=samples,
                             median_ms={k: statistics.median(v) for k, v in samples.items()}))
        finally:
            for bound in bounds.values():
                bound.close()
            device.check(device.free(src))
            device.check(device.free(dst))
    report = dict(backend='rocm', device=subprocess.check_output(['/opt/rocm/bin/rocminfo'], text=True),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(), recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        images={k: hashlib.sha256(p.image).hexdigest() for k, p in packages.items()},
        timing='resident HIP events; randomized per-trial order', process_id=os.getpid(), seed=args.seed, trials=args.trials,
        launches=args.launches, resources=resources, selector_promotion=False, rows=rows)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
