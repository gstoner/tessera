#!/usr/bin/env python3
"""Owning-device recovery close after injected pin/unpin receipt-copy failure."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_pool_resident_ssd import Memory  # noqa: E402
from tessera.compiler.resident_incremental_pool import ResidentIncrementalPool  # noqa: E402
from tessera.compiler.resident_object_pool import _UNCERTAIN_POOLS  # noqa: E402


def until(predicate):
    deadline = time.monotonic() + 30
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError('snapshot recovery test deadline')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['nvidia', 'rocm'])
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d = Device(args.backend)
    memory = Memory(d)
    stream = ct.c_void_p()
    create = getattr(d.lib, 'cuStreamCreate' if d.cuda else 'hipStreamCreateWithFlags')
    create.argtypes, create.restype = [ct.POINTER(ct.c_void_p), ct.c_uint], ct.c_int
    destroy = getattr(d.lib, 'cuStreamDestroy_v2' if d.cuda else 'hipStreamDestroy')
    destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int
    d.check(create(ct.byref(stream), 1))
    try:
        payload = memory.put(np.ones(8, np.int8))
        proofs = []
        for phase in ('pin', 'unpin'):
            pool = ResidentIncrementalPool(4, 8, 1, stream=stream.value, compiler=args.compiler,
                llvm_bin=Path('/usr/lib/llvm-23/bin'), backend=args.backend,
                chip='sm_120' if d.cuda else 'gfx1151')
            pool.allocate(stream.value, payload, 8).wait()
            snapshot = pool.snapshot(stream.value)
            pool.prepare_readers(1)
            receipt = pool._free_receipts[-1]
            def fail(*args):
                raise RuntimeError('injected receipt-copy failure')
            if phase == 'unpin':
                reader = pool.begin_read_object(stream.value, 0, 1)
                until(reader.poll)
                with reader:
                    pass
                until(reader.completion.poll)
            receipt.enqueue_copy = fail
            try:
                if phase == 'pin':
                    pool.begin_read_object(stream.value, 0, 1)
                else:
                    pool.poll_object_readers(stream.value)
            except RuntimeError as error:
                assert 'injected' in str(error)
            else:
                raise AssertionError('receipt failure did not propagate')
            assert pool in _UNCERTAIN_POOLS and snapshot.buffers
            try:
                snapshot.read(stream.value)
            except RuntimeError:
                pass
            else:
                raise AssertionError('poisoned parent admitted a snapshot read')
            pool.close()
            assert pool.closed and snapshot.closed and not pool.buffers and not snapshot.buffers
            assert pool not in _UNCERTAIN_POOLS and not pool._recovery_close and not snapshot._closing
            proofs.append(phase + ' receipt failure: parent and snapshot released after completion')
        files = ['python/tessera/compiler/resident_pool_snapshot.py',
                 'python/tessera/compiler/resident_incremental_pool.py']
        packet = dict(backend=args.backend, chip='sm_120' if d.cuda else 'gfx1151', proofs=proofs,
                      compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                      recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
                      promotion_eligible=False, fault='injected receipt-copy exception; not a driver hang')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2) + '\n')
        print(json.dumps(packet))
    finally:
        d.check(d.sync())
        memory.close()
        d.check(destroy(stream))


if __name__ == '__main__':
    main()
