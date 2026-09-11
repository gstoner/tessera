#!/usr/bin/env python3
"""Bounded native atomic publication/retirement and asynchronous finalization proof."""
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
from tessera.compiler.resident_incremental_pool import ResidentIncrementalPool  # noqa: E402
from tessera.compiler.gpu_heap_collection import materialize_pool  # noqa: E402
from tessera.compiler.heap_writer_model import explore_writers, explore_retirement  # noqa: E402
from tessera.compiler.heap_barrier_contract import read_heap_contract  # noqa: E402


def until(predicate):
    deadline = time.monotonic() + 30
    while not predicate():
        if time.monotonic() > deadline:
            raise RuntimeError('heap receipt did not complete within recorder deadline')


class AsyncClosePool(ResidentIncrementalPool):
    def __exit__(self, *exc):
        ticket = self.close_async()
        until(ticket.poll)


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
        with AsyncClosePool(4, 8, 1, stream=s0, **options) as pool:
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
                until(pool.finish_mark_async(s0).poll)
            except ValueError as exc:
                assert 'incomplete' in str(exc)
            else:
                raise AssertionError('mark allocation was not published grey')
            pool.mark_step(s0, 4).wait()
            until(pool.finish_mark_async(s0).poll)
            pool.wait()
            with pool.read(s0) as views:
                d.check(d.sync())
                np.testing.assert_array_equal(memory.get(views[0])[:, 2], [1, 1, 0, 0])
            # Direct native callers share one gate and separate result buffers.
            gate = memory.put(np.ones(1, np.int64))
            private = memory.put(np.zeros(3, np.int64))
            roots = memory.put(np.array([1, 1, 0, 0], np.int64))
            edges = memory.put(np.tile(np.array([-1, 0], np.int64), (4, 1)))
            unrooted = memory.put(np.array([1, 0, 0, 0], np.int64))
            pool.set_graph(s0, unrooted, edges).wait()
            pool.begin_mark(s0)
            pool.mark_step(s0, 1).wait()  # Slot 1 is white and eligible for retirement.
            atomic = {mode: materialize_pool(4, 8, mode, payload_dtype='int8', references=1,
                                            **options).bind()
                      for mode in ('atomic_graph_incremental', 'atomic_retire_marked')}
            writer, retire = atomic.values()
            writer_args = (pool._state, pool._roots, pool._edges, roots, edges, pool._marks, private, gate, 1)
            retire_args = (pool._state, pool._roots, pool._edges, pool._marks, pool._status, gate, 1)
            writer.submit(s0, *writer_args).ticket.wait()
            assert memory.get(private)[0] == 3  # No spin and no write while held.
            retire.submit(s1, *retire_args).ticket.wait()
            assert memory.get(pool._status)[0] == 3
            memory.write(gate, np.zeros(1, np.int64))
            # No stream dependency between these submissions; overlap is not measured.
            w = writer.submit(s0, *writer_args)
            r = retire.submit(s1, *retire_args)
            w.ticket.wait()
            r.ticket.wait()
            assert memory.get(private)[0] in (0, 2, 3)
            assert memory.get(pool._status)[0] in (0, 2, 3)
            assert memory.get(gate)[0] == 0
            if memory.get(pool._roots)[1]:
                assert memory.get(pool._state)[1, 2] == 1
                assert memory.get(pool._marks)[1] != 0
            atomic_artifacts = {mode: dict(binding=b.package.binding_digest,
                                           protocol=read_heap_contract(b.package.arena_ir))
                                for mode, b in atomic.items()}
            for binding in atomic.values():
                binding.close()
            artifacts = {mode: dict(binding=b.package.binding_digest,
                                   protocol=read_heap_contract(b.package.arena_ir))
                         for mode, b in pool._incremental_bindings.items()}
        packet = dict(schema=1, backend=args.backend, chip=options['chip'], host=platform.node(),
                      compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                      recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      retirement_model=explore_retirement(), missing_gate=explore_retirement(omit_gate=True),
                      artifacts=artifacts, atomic_artifacts=atomic_artifacts, writer_model=explore_writers(),
                      split_writer_counterexample=explore_writers(split_reservation=True),
                      proofs=['private polled admission and unpin receipts reused across eight scopes',
                              'cancelled admission releases pin after completion',
                              'stale admission refuses without poisoning pool',
                              'allocation during marking publishes grey root before retirement',
                              'atomic graph writer and retirement share try-lock; busy refuses without spinning',
                              'private finalization polled; pool close runs in bounded context-owning worker'],
                      promotion_eligible=False, measured_overlap=False,
                      envelope='atomic kernels require all competing metadata users to share gate; resident pool remains epoch serialized; worker driver operations may block')
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
