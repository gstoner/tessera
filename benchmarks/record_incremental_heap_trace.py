#!/usr/bin/env python3
"""Owning-device correctness for incremental sweeping and traced SSD VJP."""
import argparse
from collections import deque
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_pool_resident_ssd import Memory  # noqa: E402
from tessera.compiler.object_discovery import ExtensionLayout  # noqa: E402
from tessera.compiler.resident_object_pool import ResidentObjectPool  # noqa: E402
from tessera.compiler.resident_ssd import ResidentSSDProgram  # noqa: E402
from tessera.compiler.resident_trace import ResidentSSDTrace  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.control import vjp  # noqa: E402


def reference(values):
    x, a, b, c, state = values
    state = state.copy()
    out = np.empty_like(x)
    for t in range(len(x)):
        state = a[t, :, None, None] * state + b[t, :, :, None] * x[t, :, None, :]
        out[t] = (c[t, :, :, None] * state).sum(axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['nvidia', 'rocm'], required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d = Device(args.backend)
    memory = Memory(d)
    P = ct.c_void_p
    create = getattr(d.lib, 'cuStreamCreate' if d.cuda else 'hipStreamCreateWithFlags')
    destroy = getattr(d.lib, 'cuStreamDestroy_v2' if d.cuda else 'hipStreamDestroy')
    create.argtypes, create.restype = [ct.POINTER(P), ct.c_uint], ct.c_int
    destroy.argtypes, destroy.restype = [P], ct.c_int
    synchronize = getattr(d.lib, 'cuStreamSynchronize' if d.cuda else 'hipStreamSynchronize')
    synchronize.argtypes, synchronize.restype = [P], ct.c_int
    def read(view):
        producer = view.__cuda_array_interface__.get('stream')
        if producer is not None:
            d.check(synchronize(P(producer)))
        return memory.get(view)
    streams = [P(), P()]
    for s in streams:
        d.check(create(ct.byref(s), 1))
    s0, s1 = [s.value for s in streams]
    options = dict(compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
                   backend=args.backend, chip='sm_120' if d.cuda else 'gfx1151')
    try:
        obj = deque()
        obj.append(obj)
        layout = ExtensionLayout(deque, 'deque-v1', lambda value: (b'cycle', tuple(value)))
        with ResidentObjectPool.from_objects(obj, stream=s0, extension_layouts=(layout,), **options) as pool:
            pool.collect(s1).wait()
            with pool.read(s0) as views:
                assert read(views[0])[0, 2] == 1
        with ResidentObjectPool(4, 8, 1, stream=s0, **options) as pool:
            data = memory.put(np.zeros(8, np.int8))
            for _ in range(4):
                pool.allocate(s0, data, 8).wait()
            roots = memory.put(np.array([0, 0, 1, 0], np.int64))
            links = np.tile(np.array([-1, 0], np.int64), (4, 1))
            links[1] = [3, 1]
            links[3] = [1, 1]
            edges = memory.put(links)
            pool.set_graph(s0, roots, edges)
            pool.begin_collection(s0, marker_stream=s1)
            pool.finish_collection(s0, sweep_budget=2).wait()
            with pool.read(s1) as views:
                np.testing.assert_array_equal(read(views[0])[:, 2], [0, 0, 1, 2])
            # Reuse a swept slot and publish a new root between batches.
            pool.allocate(s1, data, 8).wait()
            memory.write(roots, np.array([2, 0, 1, 0], np.int64))
            pool.set_graph(s1, roots, edges)
            pool.finish_collection(s0, sweep_budget=2).wait()
            with pool.read(s1) as views:
                state = read(views[0])
                np.testing.assert_array_equal(state[:, 2], [1, 0, 1, 0])
                assert state[0, 0] == 2
            heap_binding = pool._seeded.package.binding_digest
        rng = np.random.default_rng(744)
        shapes = [(3, 2, 2), (3, 2), (3, 2, 2), (3, 2, 2), (2, 2, 2)]
        values = [rng.uniform(-.5, .5, shape).astype(np.float32) for shape in shapes + shapes[1:]]
        inputs = [memory.put(value) for value in values]
        seed = rng.uniform(-.5, .5, shapes[0]).astype(np.float32)
        dy = memory.put(seed)
        logical = lower_scheduled_ssd(3, 2, 2, 2, 2, compiler=args.compiler)
        with ResidentSSDProgram(logical, **options) as program:
            for binding, package in ((program._forward, program.forward), (program._reverse, program.reverse)):
                binding._bound = package.package.bind()
                def forbidden():
                    raise AssertionError('trace synchronized a context')
                binding._bound._sync = forbidden
            trace = ResidentSSDTrace(lambda *x: program(program(*x[:5]), *x[5:]))
            value, pullback = vjp(trace, *inputs, stream=s0)
            gradients = pullback(dy)
            with value.read(s1) as output:
                got = read(output)
            with gradients.read(s1) as outputs:
                observed = [read(output) for output in outputs]
            precise = [v.astype(np.float64) for v in values]
            def composed(v):
                return reference([reference(v[:5]), *v[5:]])
            np.testing.assert_allclose(got, composed(precise), rtol=1e-5, atol=1e-7)
            errors = []
            for i, array in enumerate(precise):
                expected = np.empty_like(array)
                for index in np.ndindex(array.shape):
                    saved = array[index]
                    array[index] = saved + 1e-5
                    plus = np.sum(composed(precise) * seed)
                    array[index] = saved - 1e-5
                    minus = np.sum(composed(precise) * seed)
                    array[index] = saved
                    expected[index] = (plus - minus) / 2e-5
                np.testing.assert_allclose(observed[i], expected, rtol=1e-4, atol=1e-7)
                errors.append(float(np.max(np.abs(observed[i] - expected))))
            program.retire_async(s1)
            deadline = time.monotonic() + 30
            while not program.poll_close():
                if time.monotonic() > deadline:
                    raise RuntimeError('retirement observation deadline')
                time.sleep(.001)
            bindings = [program.forward.package.binding_digest, program.reverse.package.binding_digest]
        args.output.write_text(json.dumps(dict(schema=1, backend=args.backend, chip=options['chip'],
            host_kernel=platform.release(), compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
            recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            execution='native_gpu', heap_binding=heap_binding, ssd_bindings=bindings,
            incremental_sweep_with_mutation=True, trusted_extension_cycle=True,
            traced_ssd_calls=2, public_input_gradients=9, gradient_max_abs_errors=errors,
            asynchronous_retirement=True, promotion_eligible=False), indent=2) + '\n')
    finally:
        d.check(d.sync())
        memory.close()
        for s in streams:
            d.check(destroy(s))


if __name__ == '__main__':
    main()
