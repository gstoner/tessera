#!/usr/bin/env python3
"""Real traced reduction/JVP/VJP packages; export Apple products on compiler host."""
import argparse
import ctypes as ct
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
import tessera as ts  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_storage_pair import NativeStoragePair  # noqa: E402
from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage  # noqa: E402


@ts.jit(autodiff='forward')
def sum_pair(x):
    return ts.ops.reduce(x, op='sum', axis=0)


@ts.jit(autodiff='forward')
def mean_pair(x):
    return ts.ops.reduce(x, op='mean', axis=0)


@ts.jit(autodiff='reverse')
def sum_vjp(x):
    return ts.ops.reduce(x, op='sum', axis=0)


@ts.jit(autodiff='reverse')
def mean_vjp(x):
    return ts.ops.reduce(x, op='mean', axis=0)


@ts.jit(autodiff='forward')
def tanh_pair(x):
    return ts.ops.tanh(x)


@ts.jit(autodiff='reverse')
def square_vjp(x):
    return ts.ops.mul(x, x)


@ts.jit(autodiff='reverse')
def tanh_vjp(x):
    return ts.ops.tanh(x)


def inputs_oracles(family, width):
    x = (np.arange(width, dtype=np.float32) % 31 - 15) / 8
    d = (np.arange(width, dtype=np.float32) % 7 - 3) / 8
    if family in ('sum_vjp', 'mean_vjp'):
        d = np.array([-0.375], np.float32)
        op = np.sum if family == 'sum_vjp' else np.mean
        y = np.array([op(x)], np.float32)
        dy = np.full(width, d[0] / (width if family == 'mean_vjp' else 1), np.float32)
    elif family in ('sum', 'mean'):
        op = np.sum if family == 'sum' else np.mean
        y, dy = np.array([op(x)], np.float32), np.array([op(d)], np.float32)
    elif family == 'square_vjp':
        y, dy = x*x, (d*x + d*x)
    else:
        y = np.tanh(x.astype(np.float64))
        dy = d*(1-y*y)
    return x, d, y, dy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['nvidia', 'rocm', 'apple'], required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    device = None if a.backend == 'apple' else Device(a.backend)
    rows = []
    if device is None:
        a.output.mkdir(parents=True, exist_ok=True)
    for family, fn in [('sum', sum_pair), ('mean', mean_pair), ('tanh_jvp', tanh_pair), ('square_vjp', square_vjp), ('tanh_vjp', tanh_vjp), ('sum_vjp', sum_vjp), ('mean_vjp', mean_vjp)]:
        for width in [32, 64, 256]:
            x, d, expected, derivative = inputs_oracles(family, width)
            package = fn.compile_native_storage_pair(x, compiler=a.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
                backend=a.backend, chip='sm_120' if a.backend == 'nvidia' else 'gfx1151')
            if device is None:
                (a.output / f'{family}-{width}.json').write_text(json.dumps(dict(**asdict(package), digest=package.digest)))
                continue
            package = NativeGPUStoragePackage.from_json(package.to_json(), expected_digest=package.binding_digest)
            pair = NativeStoragePair(package)
            arrays = [x, d, np.zeros(expected.shape, np.float32), np.zeros(derivative.shape, np.float32)]
            pointers, tensors = [], []
            try:
                for arr in arrays:
                    ptr = ct.c_void_p()
                    device.check(device.alloc(ct.byref(ptr), arr.nbytes))
                    pointers.append(ptr)
                    device.check(device.htod(ptr, arr.ctypes.data, arr.nbytes) if device.cuda else device.copy(ptr, arr.ctypes.data, arr.nbytes, 1))
                    tensors.append(SimpleNamespace(__cuda_array_interface__=dict(version=3, shape=arr.shape, typestr=arr.dtype.str, data=(ptr.value, False))))
                assert pair(*tensors, width) == tuple(tensors[-2:])
                for arr, ptr in zip(arrays[-2:], pointers[-2:]):
                    device.check(device.dtoh(arr.ctypes.data, ptr, arr.nbytes) if device.cuda else device.copy(arr.ctypes.data, ptr, arr.nbytes, 2))
                np.testing.assert_allclose(arrays[-2], expected, atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(arrays[-1], derivative, atol=2e-6, rtol=2e-6)
                rows.append(dict(family=family, width=width, mode=pair.contract['mode'], oracle='atol=rtol=2e-6', package_digest=package.binding_digest))
            finally:
                pair.close()
                for ptr in pointers:
                    device.check(device.free(ptr))
    if device:
        a.output.write_text(json.dumps(dict(backend=a.backend, rows=rows,
            compiler_sha256=hashlib.sha256(a.compiler.read_bytes()).hexdigest(),
            recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
        print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
