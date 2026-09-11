#!/usr/bin/env python3
"""Independent owning-device proof and non-promoting heap protocol comparison.

Event intervals include host submission gaps. They are not attributed kernel
execution time. The recorder never installs selector evidence.
"""
import argparse
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
from tessera.compiler.resident_object_pool import ResidentObjectPool  # noqa: E402
from tessera.compiler.heap_barrier_contract import read_heap_contract  # noqa: E402
from tessera.compiler.heap_protocol_model import explore  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['nvidia', 'rocm'], required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.samples <= 100:
        parser.error('samples must be 1..100')
    d, P = Device(args.backend), ct.c_void_p
    memory = Memory(d)
    def bind(cu, hip, params):
        fn = getattr(d.lib, cu if d.cuda else hip)
        fn.argtypes, fn.restype = params, ct.c_int
        return fn
    create = bind('cuStreamCreate', 'hipStreamCreateWithFlags', [ct.POINTER(P), ct.c_uint])
    destroy = bind('cuStreamDestroy_v2', 'hipStreamDestroy', [P])
    sync = bind('cuStreamSynchronize', 'hipStreamSynchronize', [P])
    copy = bind('cuMemcpyDtoDAsync_v2', 'hipMemcpyDtoDAsync', [P, P, ct.c_size_t, P])
    streams = [P(), P(), P()]
    for stream in streams:
        d.check(create(ct.byref(stream), 1))
    s0, s1, s2 = [s.value for s in streams]
    options = dict(compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
                   backend=args.backend, chip='sm_120' if d.cuda else 'gfx1151')
    def read(view):
        spec = view.__cuda_array_interface__
        if spec.get('stream'):
            d.check(sync(P(spec['stream'])))
        return memory.get(view)
    def status(pool):
        with pool.read(s0) as views:
            return read(views[-1])
    def graph(pool):
        with pool.read(s0) as views:
            return read(views[1]), read(views[2])
    proofs, timings, bindings = [], {}, {}
    try:
        with ResidentObjectPool(4, 8, 1, stream=s0, **options) as pool:
            payload = memory.put(np.full(8, 7, np.int8))
            for _ in range(4):
                pool.allocate(s0, payload, 8).wait()
            root_data = np.array([1, 0, 0, 0], np.int64)
            edge_data = np.array([[1, 1], [-1, 0], [3, 1], [2, 1]], np.int64)
            roots, edges = memory.put(root_data), memory.put(edge_data)
            pool.set_graph(s0, roots, edges).wait()
            assert status(pool)[0] == 0
            before = graph(pool)
            bad = edge_data.copy()
            bad[0] = [1, 99]
            memory.write(edges, bad)
            pool.set_graph(s1, roots, edges).wait()
            assert status(pool)[0] == 2
            for a, b in zip(before, graph(pool), strict=True):
                np.testing.assert_array_equal(a, b)
            proofs.append('stale edge rejects the entire graph transaction')
            # Snapshot mark retains original roots; new publication is remarked.
            memory.write(edges, edge_data)
            pool.begin_collection(s0, marker_stream=s2)
            root_data[2] = 1
            memory.write(roots, root_data)
            pool.set_graph(s1, roots, edges).wait()
            pool.finish_collection(s0).wait()
            with pool.read(s1) as views:
                np.testing.assert_array_equal(read(views[0])[:, 2], [1, 1, 1, 1])
            proofs.append('new root between snapshot and final remark preserves cycle')
            memory.write(roots, np.zeros(4, np.int64))
            pool.set_graph(s0, roots, edges).wait()
            pool.retire_unreachable(s1).wait()
            with pool.read(s0) as views:
                np.testing.assert_array_equal(read(views[0])[:, 2], [2] * 4)
            pool.allocate(s0, payload, 8).wait()
            assert status(pool)[0] == 1  # retired slots cannot satisfy allocation
            memory.write(roots, np.array([1, 0, 0, 0], np.int64))
            pool.set_graph(s0, roots, edges).wait()
            assert status(pool)[0] == 2
            proofs.append('retired slots neither allocate nor resurrect')
            captured = [memory.put(np.zeros((4, 8), np.int8)) for _ in range(2)]
            new_payload = memory.put(np.full(8, 9, np.int8))
            for stream, destination in zip((s1, s2), captured, strict=True):
                with pool.read(stream) as views:
                    try:
                        pool.reclaim_retired(s0)
                    except ValueError as exc:
                        assert 'scopes' in str(exc)
                    else:
                        raise AssertionError('reuse admitted an open reader')
                    src = views[3].__cuda_array_interface__['data'][0]
                    dst = destination.__cuda_array_interface__['data'][0]
                    d.check(copy(P(dst), P(src), 32, P(stream)))
            # No host wait between the reader copies, reclaim and reallocation.
            pool.reclaim_retired(s0)
            pool.allocate(s0, new_payload, 8).wait()
            assert status(pool)[2] == 2
            for view in captured:
                np.testing.assert_array_equal(read(view), np.full((4, 8), 7, np.int8))
            proofs.append('two reader streams complete before reclaimed payload reuse')
            for name in ('_graph', '_retire', '_reclaim'):
                package = getattr(pool, name).package
                bindings[name] = dict(binding=package.binding_digest, protocol=read_heap_contract(package.arena_ir))
        with ResidentObjectPool(16, 32, 1, stream=s0, **options) as pool:
            state = np.array([[1, 32, 1]] * 16, np.int64)
            roots = np.array([1] + [0] * 15, np.int64)
            edges = np.tile(np.array([-1, 0], np.int64), (16, 1))
            edges[0] = [1, 1]
            # Independent reachability oracle: only slots 0 and 1 survive.
            def reset():
                pool.wait()
                with pool._access().write(s0):
                    for buffer, data in zip((pool._state, pool._roots, pool._edges),
                                            (state, roots, edges), strict=True):
                        pool.check(pool._upload(buffer.pointer, P(data.ctypes.data), data.nbytes))
            def execute(name):
                if name == 'exclusive_collect':
                    return pool.collect(s0)
                pool.retire_unreachable(s0)
                return pool.reclaim_retired(s0)
            # Both routes compiled and warmed before either is measured.
            for name in ('exclusive_collect', 'split_retire_reclaim'):
                reset()
                execute(name).wait()
                timings[name] = []
            for iteration in range(args.samples):
                names = ('exclusive_collect', 'split_retire_reclaim')
                for name in names[::1 if iteration % 2 == 0 else -1]:
                    reset()
                    start, end = P(), P()
                    d.check(d.event_create(ct.byref(start), 0))
                    d.check(d.event_create(ct.byref(end), 0))
                    try:
                        d.check(d.event_record(start, P(s0)))
                        t0 = time.perf_counter_ns()
                        execute(name)
                        d.check(d.event_record(end, P(s0)))
                        d.check(d.event_sync(end))
                        wall = (time.perf_counter_ns() - t0) / 1e6
                        elapsed = ct.c_float()
                        d.check(d.event_elapsed(ct.byref(elapsed), start, end))
                        timings[name].append(dict(completion_wall_ms=wall, submission_stream_interval_ms=elapsed.value))
                    finally:
                        d.check(d.event_destroy(start))
                        d.check(d.event_destroy(end))
                    with pool.read(s1) as views:
                        np.testing.assert_array_equal(read(views[0])[:, 2], [1, 1] + [0] * 14)
            for name in ('_collect', '_retire', '_reclaim'):
                package = getattr(pool, name).package
                bindings['measurement' + name] = dict(binding=package.binding_digest,
                                                      protocol=read_heap_contract(package.arena_ir))
        packet = dict(schema=1, backend=args.backend, chip=options['chip'], host=platform.node(),
                      kernel=platform.release(), compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                      recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      model=explore(), missing_reclamation_barrier=explore(unsafe_reuse=True),
                      proofs=proofs, artifacts=bindings, measurements=timings,
                      promotion_eligible=False, performance_scope='single process; stream interval includes host submission gaps',
                      envelope='exclusive stream epoch; one graph writer; final remark exclusive; no racing sweep')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2) + '\n')
        print(json.dumps(dict(output=str(args.output), proofs=proofs)))
    finally:
        d.check(d.sync())
        memory.close()
        for stream in streams:
            d.check(destroy(stream))


if __name__ == '__main__':
    main()
