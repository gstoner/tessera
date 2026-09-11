#!/usr/bin/env python3
"""Owning-device proof of per-object pins and incremental update marking."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_pool_resident_ssd import Memory  # noqa: E402
from tessera.compiler.resident_incremental_pool import ResidentIncrementalPool  # noqa: E402
from tessera.compiler.incremental_heap_model import explore_graph  # noqa: E402
from tessera.compiler.heap_barrier_contract import read_heap_contract  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['nvidia', 'rocm'], required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d, P = Device(args.backend), ct.c_void_p
    memory = Memory(d)
    def bind(cu, hip, signature):
        fn = getattr(d.lib, cu if d.cuda else hip)
        fn.argtypes, fn.restype = signature, ct.c_int
        return fn
    create = bind('cuStreamCreate', 'hipStreamCreateWithFlags', [ct.POINTER(P), ct.c_uint])
    destroy = bind('cuStreamDestroy_v2', 'hipStreamDestroy', [P])
    sync = bind('cuStreamSynchronize', 'hipStreamSynchronize', [P])
    copy = bind('cuMemcpyDtoDAsync_v2', 'hipMemcpyDtoDAsync', [P, P, ct.c_size_t, P])
    streams = [P(), P()]
    for s in streams:
        d.check(create(ct.byref(s), 1))
    s0, s1 = [s.value for s in streams]
    options = dict(compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
                   backend=args.backend, chip='sm_120' if d.cuda else 'gfx1151')
    def state(pool):
        with pool.read(s0) as views:
            d.check(sync(P(s0)))
            return memory.get(views[0])
    def status(pool):
        with pool.read(s0) as views:
            d.check(sync(P(s0)))
            return memory.get(views[-1])
    try:
        payload = memory.put(np.full(8, 7, np.int8))
        new_payload = memory.put(np.full(8, 9, np.int8))
        copied = memory.put(np.zeros(8, np.int8))
        roots_data = np.array([1, 0, 0, 0], np.int64)
        edges_data = np.tile(np.array([-1, 0], np.int64), (4, 1))
        roots, edges = memory.put(roots_data), memory.put(edges_data)
        with ResidentIncrementalPool(4, 8, 1, stream=s0, **options) as pool:
            for _ in range(4):
                pool.allocate(s0, payload, 8).wait()
            pool.set_graph(s0, roots, edges).wait()
            pool.begin_mark(s0)
            pool.mark_step(s0, 1).wait()  # object 0 is black
            edges_data[0] = [1, 1]
            memory.write(edges, edges_data)
            pool.set_graph(s0, roots, edges).wait()  # insertion shades object 1
            assert status(pool)[0] == 0
            before = state(pool)
            try:
                pool.finish_mark(s0)
            except ValueError as exc:
                assert 'incomplete' in str(exc)
            else:
                raise AssertionError('final retirement ignored pending grey work')
            np.testing.assert_array_equal(state(pool), before)
            with pool.read_object(s1, 3, 1) as view:
                d.check(copy(P(copied.__cuda_array_interface__['data'][0]),
                             P(view.__cuda_array_interface__['data'][0]), 8, P(s1)))
                pool.mark_step(s0, 1).wait()
                pool.finish_mark(s0)
                np.testing.assert_array_equal(state(pool)[:, 2], [1, 1, 2, 2])
                pool.reclaim_retired(s0).wait()
                np.testing.assert_array_equal(state(pool)[:, 2], [1, 1, 0, 2])
                # Unrelated slot 2 can be reused while object 3 is still admitted.
                pool.allocate(s0, new_payload, 8).wait()
                np.testing.assert_array_equal(state(pool)[:, 2], [1, 1, 1, 2])
                try:
                    with pool.read_object(s0, 3, 1):
                        pass
                except ValueError as exc:
                    assert 'retired' in str(exc)
                else:
                    raise AssertionError('new reader acquired a retired object')
            pool.wait()  # explicit reader completion; admission/cleanup are not fully async
            pool.reclaim_retired(s0).wait()
            pool.allocate(s0, new_payload, 8).wait()
            assert state(pool)[3, 0] == 2
            np.testing.assert_array_equal(memory.get(copied), np.full(8, 7, np.int8))
            try:
                with pool.read_object(s1, 3, 1):
                    pass
            except ValueError as exc:
                assert 'stale' in str(exc)
            else:
                raise AssertionError('old generation reacquired reused payload')
            with pool.read_object(s1, 3, 2) as view:
                d.check(sync(P(s1)))
                np.testing.assert_array_equal(memory.get(view), np.full(8, 9, np.int8))
            # A fresh cycle with no roots eventually collects all floating garbage.
            memory.write(roots, np.zeros(4, np.int64))
            pool.set_graph(s0, roots, edges).wait()
            pool.begin_mark(s0)
            pool.finish_mark(s0)
            pool.wait()
            pool.reclaim_retired(s0).wait()
            np.testing.assert_array_equal(state(pool)[:, 2], [0] * 4)
            artifacts = {mode: dict(binding=b.package.binding_digest, protocol=read_heap_contract(b.package.arena_ir))
                         for mode, b in pool._incremental_bindings.items()}
        packet = dict(schema=1, backend=args.backend, chip=options['chip'], host=platform.node(), kernel=platform.release(),
                      compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                      recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      model=explore_graph(), missing_barrier=explore_graph(omit_barrier=True),
                      missing_pins=explore_graph(omit_pins=True), artifacts=artifacts,
                      proofs=['black-to-white edge insertion produces consumed grey work',
                              'incomplete marking refuses retirement without mutation',
                              'admitted payload reader survives logical retirement',
                              'unrelated unpinned slot reuses during active object lease',
                              'retired and stale generations reject new readers',
                              'last completed reader permits reuse; next cycle collects floating garbage'],
                      promotion_eligible=False, envelope='single metadata writer; immutable payload; synchronous pin admission; no measured hardware overlap')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2) + '\n')
        print(json.dumps(dict(output=str(args.output), model=packet['model'], proofs=packet['proofs'])))
    finally:
        d.check(d.sync())
        memory.close()
        for stream in streams:
            d.check(destroy(stream))


if __name__ == '__main__':
    main()
