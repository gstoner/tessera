#!/usr/bin/env python3
"""Gated metadata inventory and process-death recovery on the owning device."""
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
from tessera.compiler.resident_gated_pool import ResidentGatedPool
from tessera.compiler.native_isolated_heap import HEALTH_PROBE, IsolatedHeapPool  # noqa: E402
from tessera.compiler.heap_writer_model import explore_writers  # noqa: E402
from tessera.compiler.heap_barrier_contract import read_heap_contract  # noqa: E402


def until(predicate):
    deadline = time.monotonic() + 30
    while not predicate():
        if time.monotonic() > deadline:
            raise RuntimeError('heap receipt did not complete within recorder deadline')


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
    copy = bind('cuMemcpyDtoDAsync_v2', 'hipMemcpyDtoDAsync', [P, P, ct.c_size_t, P])
    streams = [P(), P()]
    for stream in streams:
        d.check(create(ct.byref(stream), 1))
    s0, s1 = [s.value for s in streams]
    options = dict(compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
                   backend=args.backend, chip='sm_120' if d.cuda else 'gfx1151')
    try:
        payload = memory.put(np.full(8, 7, np.int8))
        copied = memory.put(np.zeros(8, np.int8))
        with ResidentGatedPool(4, 8, 1, stream=s0, **options) as pool:
            pool.allocate(s0, payload, 8).wait()
            pool.prepare_readers(2)
            saved_status = pool._status_values
            def forbidden(*args):
                raise AssertionError('asynchronous receipt path used shared synchronous status')
            pool._status_values = forbidden
            for _ in range(8):
                request = pool.begin_read_object(s1, 0, 1)
                until(request.poll)
                with request as view:
                    assert view.__cuda_array_interface__['shape'] == (8,)
                    d.check(copy(P(copied.__cuda_array_interface__['data'][0]),
                                 P(view.__cuda_array_interface__['data'][0]), 8, P(s1)))
                until(lambda: pool.poll_object_readers(s0))
            request = pool.begin_read_object(s1, 0, 1)
            request.cancel()
            until(lambda: pool.poll_object_readers(s0))
            stale = pool.begin_read_object(s1, 0, 999)
            try:
                until(stale.poll)
            except ValueError as exc:
                assert 'stale' in str(exc)
            else:
                raise AssertionError('stale generation admitted')
            assert not pool._object_uncertain
            assert len(pool._heap_receipts) == len(pool._free_receipts) == 2
            pool._status_values = saved_status
            pool.wait()
            np.testing.assert_array_equal(memory.get(copied), np.full(8, 7, np.int8))
            pool.begin_mark(s0)
            pool.mark_step(s0, 4).wait()
            allocation = pool.allocate(s0, payload, 8)
            assert pool._status_values(allocation)[0] == 0
            try:
                pool.finish_mark(s0)
            except ValueError as exc:
                assert 'incomplete' in str(exc)
            else:
                raise AssertionError('mark allocation was not published grey')
            pool.mark_step(s0, 4).wait()
            pool.finish_mark(s0)
            pool.wait()
            np.testing.assert_array_equal(pool.inspect_metadata(s0)[0][:, 2], [1, 1, 0, 0])
            roots = memory.put(np.zeros(4, np.int64))
            edges = memory.put(np.tile(np.array([-1, 0], np.int64), (4, 1)))
            pool.set_graph(s0, roots, edges).wait()
            pool.begin_mark(s0)
            pool.set_graph(s0, roots, edges).wait()
            until(pool.finish_mark_async(s0).poll)
            pool.reclaim_retired(s0).wait()
            np.testing.assert_array_equal(pool.inspect_metadata(s0)[0][:, 2], [0] * 4)
            artifacts = {mode: dict(binding=b.package.binding_digest,
                                   protocol=read_heap_contract(b.package.arena_ir))
                         for mode, b in pool._incremental_bindings.items()}
        isolated = IsolatedHeapPool(4, 8, 1, **options)
        isolated.submit('allocate', np.full(8, 3, np.int8))
        def result(owner):
            output = owner.poll()
            if output is not None:
                results.append(output)
                return True
            return False
        results = []
        until(lambda: result(isolated))
        assert results[-1][0] == 0
        isolated.submit('inspect')
        until(lambda: result(isolated))
        assert results[-1][0][0, 2] == 1
        isolated.submit('close')
        until(lambda: result(isolated))
        assert results[-1] == 'close_receipt'
        isolated.recover_async()
        until(isolated.poll_recovery)
        assert isolated.process.exitcode is not None
        # Deterministic stopped-worker fault, not an actual driver-hang claim.
        import os
        import signal
        stalled = IsolatedHeapPool(4, 8, 1, **options)
        stalled.timeout = .05
        stalled.lease.timeout_seconds = .2
        os.kill(stalled.process.pid, signal.SIGSTOP)
        stalled.submit('close')
        try:
            until(lambda: result(stalled))
        except TimeoutError:
            pass
        else:
            raise AssertionError('stopped heap worker was treated as closed')
        assert not stalled.closed
        # No replacement before the predecessor's death is confirmed.
        try:
            stalled.replacement()
        except ValueError as exc:
            assert 'confirmed' in str(exc)
        else:
            raise AssertionError('replacement admitted before confirmed death')
        stalled.recover_async()
        until(stalled.poll_recovery)
        assert stalled.process.exitcode is not None
        # A replacement is a fresh worker that passed its own in-process device
        # probe (allocate / inspect / pinned readback / empty-graph mark /
        # reclaim) before it was admitted; it then serves bounded commands.
        fresh = stalled.replacement()
        assert fresh is not stalled and fresh.process.pid != stalled.process.pid
        fresh.submit('allocate', np.full(8, 5, np.int8))
        until(lambda: result(fresh))
        assert results[-1][0] == 0
        fresh.submit('inspect')
        until(lambda: result(fresh))
        assert results[-1][0][0, 2] == 1 and results[-1][0][0, 1] == 8
        fresh.submit('close')
        until(lambda: result(fresh))
        assert results[-1] == 'close_receipt'
        fresh.recover_async()
        until(fresh.poll_recovery)
        assert fresh.process.exitcode is not None
        packet = dict(schema=1, backend=args.backend, chip=options['chip'], host=platform.node(),
                      compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                      recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      health=HEALTH_PROBE, artifacts=artifacts, writer_model=explore_writers(),
                      split_writer_counterexample=explore_writers(split_reservation=True),
                      proofs=['private polled admission and unpin receipts reused across eight scopes',
                              'cancelled admission releases pin after completion',
                              'stale admission refuses without poisoning pool',
                              'allocation during marking publishes grey root before retirement',
                              'all admitted metadata kernels carry the owned gate',
                              'metadata copies remain valid after reclamation',
                              'normal and stopped-process teardown require confirmed worker death',
                              'a worker is admitted only after its in-process device probe verifies allocation, readback and reclamation',
                              'replacement refuses before confirmed predecessor death and admits a freshly probed worker after it'],
                      promotion_eligible=False, measured_overlap=False,
                      envelope='gated owner retains stream epochs; isolated bounded host commands; stopped-worker fault is not a driver hang; the startup probe proves the workload on this device ordinal now, not global driver health')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2) + '\n')
        print(json.dumps(dict(output=str(args.output), proofs=packet['proofs'])))
    finally:
        d.check(d.sync())
        memory.close()
        for stream in streams:
            d.check(destroy(stream))


if __name__ == '__main__':
    main()
