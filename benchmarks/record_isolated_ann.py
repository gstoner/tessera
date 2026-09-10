#!/usr/bin/env python3
"""Owning-device ANN binding and process-failure recovery (not driver fault injection)."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_native_ann_execution import source  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_ann import prepare_native_ann  # noqa: E402
from tessera.compiler.native_ann_gpu import materialize_native_ann_gpu  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia', 'rocm'), required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    Device(args.backend)
    os.environ['TESSERA_OPT'] = str(args.compiler.resolve())
    pair = prepare_native_ann(source(16, 8, 'square'), allow_reassociation=True)
    physical = materialize_native_ann_gpu(pair, compiler=args.compiler.resolve(),
        llvm_bin=Path('/usr/lib/llvm-23/bin'), backend=args.backend,
        chip='sm_120' if args.backend == 'nvidia' else 'gfx1151',
        fuse_elementwise=True, parallel_rows=True)
    value = np.linspace(-1, 1, 128, dtype=np.float32).reshape(16, 8)
    with physical.bind(input_bound=1, absolute_budget=.001) as direct:
        direct.verify([value, np.zeros_like(value)])
        expected = direct.run(value)
    with physical.bind_isolated(input_bound=1, absolute_budget=.001) as isolated:
        np.testing.assert_array_equal(isolated.run(value), expected)
    failed = physical.bind_isolated(input_bound=1, absolute_budget=.001)
    pid = failed._process.pid
    failed.timeout = .25
    failed.lease.timeout_seconds = .25
    start = time.monotonic()
    try:
        os.kill(pid, signal.SIGSTOP)
        try:
            failed.run(value)
        except TimeoutError:
            pass
        else:
            raise AssertionError('stopped worker returned a result')
        assert failed.failed and failed._pending is not None
        ticket = failed.recover_async()
        assert ticket is not None
        deadline = time.monotonic() + 10
        polls = 0
        while not failed.poll_recovery():
            polls += 1
            if time.monotonic() > deadline:
                raise TimeoutError('process teardown remains unconfirmed')
            time.sleep(.001)
        assert failed.lease.reusable and failed.closed
    finally:
        if not failed.closed:
            failed._poison()
            failed.recover()
    duration = time.monotonic() - start
    failed.timeout = 30.0
    with failed.replacement() as replacement:
        assert replacement._process.pid != pid
        np.testing.assert_array_equal(replacement.run(value), expected)
    names = ['native_isolated_ann.py', 'native_ann_gpu.py', 'native_driver_isolation.py']
    args.output.write_text(json.dumps(dict(schema=1, backend=args.backend,
        chip=physical.original.chip, original=physical.original.binding_digest,
        transformed=physical.transformed.binding_digest, numerical_verified=True,
        fault='SIGSTOP idle owning process before request; not an injected GPU driver failure',
        process_death_confirmed=True, replacement_verified=True, recovery_seconds=duration,
        production_promoted=False, asynchronous_recovery=True, numerical_health_admitted=True, incomplete_polls=polls, sources={name:hashlib.sha256((ROOT/'python/tessera/compiler'/name).read_bytes()).hexdigest() for name in names}), indent=2)+'\n')
    print(args.backend, 'isolated native execution and replacement passed')


if __name__ == '__main__':
    main()
