#!/usr/bin/env python3
"""CUDA/HIP queue interval evidence with serial controls and exact output checks.

Event intervals include queue scheduling and event overhead. Intersection is not
proof of simultaneous instruction issue; hardware attribution is recorded by an
external profiler against this same workload.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device, THREADS, compile_image, module  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia', 'rocm'), required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chip')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    source = module(2)
    with tempfile.TemporaryDirectory() as directory:
        image = compile_image(args.compiler, args.backend, source, Path(directory), chip=args.chip)
    device = Device(args.backend)
    P = ct.c_void_p
    def bind(cuda, hip, types):
        fn = getattr(device.lib, cuda if device.cuda else hip)
        fn.argtypes, fn.restype = types, ct.c_int
        return fn
    create = bind('cuStreamCreate', 'hipStreamCreateWithFlags', [ct.POINTER(P), ct.c_uint])
    destroy = bind('cuStreamDestroy_v2', 'hipStreamDestroy', [P])
    wait = bind('cuStreamWaitEvent', 'hipStreamWaitEvent', [P, P, ct.c_uint])
    streams, events, allocations, rows = [], [], [], []
    mod, fn = P(), P()
    blob = ct.create_string_buffer(image)
    try:
        device.check(device.load(ct.byref(mod), ct.cast(blob, P)))
        device.check(device.function(ct.byref(fn), mod, b'ring'))
        for _ in range(3):
            stream = P()
            device.check(create(ct.byref(stream), 1))  # nonblocking
            streams.append(stream)
        for _ in range(5):
            event = P()
            device.check(device.event_create(ct.byref(event), 0))
            events.append(event)
        anchor, start_a, end_a, start_b, end_b = events
        rounds = ct.c_int64(256)
        for blocks in (8, 32, 128):
            inputs = [(np.arange(blocks * rounds.value * THREADS, dtype=np.float32) + offset) % 127 for offset in (0, 37)]
            outputs = [np.zeros_like(x) for x in inputs]
            argv = []
            for x, y in zip(inputs, outputs, strict=True):
                src, dst = P(), P()
                device.check(device.alloc(ct.byref(src), x.nbytes))
                allocations.append(src)
                device.check(device.alloc(ct.byref(dst), y.nbytes))
                allocations.append(dst)
                device.check(device.htod(src, x.ctypes.data, x.nbytes) if device.cuda else device.copy(src, x.ctypes.data, x.nbytes, 1))
                argv.append((P * 3)(ct.cast(ct.byref(src), P), ct.cast(ct.byref(dst), P), ct.cast(ct.byref(rounds), P)))
            def launch(which, stream):
                device.check(device.launch(fn, blocks, 1, 1, THREADS, 1, 1, 0, stream, argv[which], None))
            for _ in range(3):
                launch(0, streams[1])
                launch(1, streams[2])
            device.check(device.sync())
            for repeat in range(args.repeats):
                # Alternate order so clock/thermal drift does not always favor one mode.
                for mode in (('serial', 'parallel') if repeat % 2 == 0 else ('parallel', 'serial')):
                    sa = streams[1]
                    sb = streams[1] if mode == 'serial' else streams[2]
                    device.check(device.event_record(anchor, streams[0]))
                    device.check(wait(sa, anchor, 0))
                    device.check(wait(sb, anchor, 0))
                    device.check(device.event_record(start_a, sa))
                    launch(0, sa)
                    device.check(device.event_record(end_a, sa))
                    device.check(device.event_record(start_b, sb))
                    launch(1, sb)
                    device.check(device.event_record(end_b, sb))
                    device.check(device.event_sync(end_a))
                    device.check(device.event_sync(end_b))
                    def elapsed(event):
                        result = ct.c_float()
                        device.check(device.event_elapsed(ct.byref(result), anchor, event))
                        return result.value
                    a0, a1, b0, b1 = map(elapsed, (start_a, end_a, start_b, end_b))
                    if not all(np.isfinite([a0, a1, b0, b1])) or a1 < a0 or b1 < b0:
                        raise RuntimeError('invalid device event interval')
                    for i, (x, y) in enumerate(zip(inputs, outputs, strict=True)):
                        dst = allocations[2 * i + 1]
                        device.check(device.dtoh(y.ctypes.data, dst, y.nbytes) if device.cuda else device.copy(y.ctypes.data, dst, y.nbytes, 2))
                        np.testing.assert_array_equal(y, np.roll(x.reshape(blocks, rounds.value, THREADS), -1, axis=2).reshape(-1) * 2 + 1)
                    rows.append(dict(blocks=blocks, repeat=repeat, mode=mode, interval_a_ms=[a0, a1], interval_b_ms=[b0, b1],
                                     intersection_ms=max(0., min(a1, b1)-max(a0, b0)), span_ms=max(a1,b1)-min(a0,b0), oracle='exact'))
            for pointer in allocations:
                device.check(device.free(pointer))
            allocations.clear()
    finally:
        device.check(device.sync())
        for pointer in allocations:
            device.check(device.free(pointer))
        for event in events:
            device.check(device.event_destroy(event))
        for stream in streams:
            device.check(destroy(stream))
        if mod.value:
            device.check(device.unload(mod))
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'] if device.cuda else ['/opt/rocm/bin/rocminfo'], text=True)
    args.output.write_text(json.dumps(dict(backend=args.backend, chip=args.chip or ('sm_120' if device.cuda else 'gfx1151'), device=identity, rows=rows,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(), compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        image_sha256=hashlib.sha256(image).hexdigest(), recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing='device events relative to a shared recorded anchor; disjoint allocations',
        hardware_counter_attribution=False, selector_promotion=False), indent=2)+'\n')
    print(f'{len(rows)} exact matched queue measurements')


if __name__ == '__main__':
    main()
