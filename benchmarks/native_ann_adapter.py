"""Bounded ANN package benchmark shared by SuperBench and DLOP.

Receipts observe the actual bound driver's launch entry point. They are API
receipts, not profiler kernel counts, and only completed, checked calls publish.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import platform
import statistics
import time

import numpy as np


@contextmanager
def launch_receipt(native):
    original = native._launch
    receipt = dict(attempted=0, accepted=0, completed=False)

    def observed(*args):
        receipt['attempted'] += 1
        status = original(*args)
        if status == 0:
            receipt['accepted'] += 1
        return status

    native._launch = observed
    try:
        yield receipt
        receipt['completed'] = True
    finally:
        native._launch = original


def measure(backend, compiler, *, rows=16, width=8, activation='relu', repeat=5):
    if type(repeat) is not int or repeat < 1:
        raise ValueError('repeat must be positive')
    from benchmarks.record_device_ring_protocol import Device
    from benchmarks.record_native_ann_execution import source
    from tessera.compiler.native_ann import prepare_native_ann
    from tessera.compiler.native_ann_gpu import materialize_native_ann_gpu

    Device(backend)
    compiler = Path(compiler).resolve()
    logical = prepare_native_ann(source(rows, width, activation), allow_reassociation=True)
    pair = materialize_native_ann_gpu(
        logical, compiler=compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
        backend=backend, chip='sm_120' if backend == 'nvidia' else 'gfx1151',
        fuse_elementwise=True, parallel_rows=True)
    value = np.linspace(-1, 1, rows * width, dtype=np.float32).reshape(rows, width)
    results = []
    with pair.bind(input_bound=1.0, absolute_budget=.001) as runner:
        runner.verify([value, np.zeros_like(value), -value])
        for variant, package in enumerate((pair.original, pair.transformed)):
            runner.run(value, transformed=bool(variant))
            samples, receipts = [], []
            for _ in range(repeat):
                start = time.perf_counter_ns()
                with launch_receipt(runner.bindings[variant]._bound) as receipt:
                    runner.run(value, transformed=bool(variant))
                samples.append((time.perf_counter_ns() - start) / 1e6)
                if receipt['accepted'] < 1 or receipt['accepted'] != receipt['attempted']:
                    raise RuntimeError('native ANN call did not produce successful launch receipts')
                receipts.append(receipt)
            results.append(dict(variant='transformed' if variant else 'original',
                                artifact=package.binding_digest, samples_ms=samples,
                                latency_ms=statistics.median(samples), receipts=receipts))
        runner.verify([value, -value])
    return dict(schema=1, ok=True, workload='two_affine_' + activation,
                shape=[rows, width], backend=backend, runtime_status='executable',
                execution_kind='native_gpu', pair=logical.digest,
                compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                host_kernel=platform.release(), numerical_verified=True,
                timing_domain='instrumented_warm_package_h2d_dispatch_d2h_host_wall',
                dispatch_count_source='observed_driver_launch_api',
                profiler_kernel_count=None, promotion_eligible=False,
                latency_ms=results[0]['latency_ms'], variants=results)
