#!/usr/bin/env python3
"""Prove persistent snapshot/repeated-backward/nested-frame ownership on device."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_storage_pair import square_vjp, tanh_vjp, sum_vjp, mean_vjp, inputs_oracles  # noqa: E402
from tessera.compiler.native_storage_pair import NativeStoragePair  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia','rocm'), required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = Device(args.backend)
    rows = []
    for family, fn in [('square_vjp',square_vjp),('tanh_vjp',tanh_vjp),('sum_vjp',sum_vjp),('mean_vjp',mean_vjp)]:
        for width in (32,64,256):
            x, d, y, derivative = inputs_oracles(family,width)
            package = fn.compile_native_storage_pair(x, compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'), backend=args.backend, chip='sm_120' if device.cuda else 'gfx1151')
            pointers = []
            pair = NativeStoragePair(package)
            def upload(value):
                pointer = ct.c_void_p()
                device.check(device.alloc(ct.byref(pointer), value.nbytes))
                pointers.append(pointer)
                device.check(device.htod(pointer,value.ctypes.data,value.nbytes) if device.cuda else device.copy(pointer,value.ctypes.data,value.nbytes,1))
                return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(pointer.value,False)))
            def download(value):
                result = np.empty(value.shape,np.float32)
                device.check(device.dtoh(result.ctypes.data,value.pointer,result.nbytes) if device.cuda else device.copy(result.ctypes.data,value.pointer,result.nbytes,2))
                return result
            try:
                source, cotangent = upload(x), upload(d)
                frame = pair.capture(source)
                np.testing.assert_allclose(download(frame.primal),y,atol=2e-6,rtol=2e-6)
                changed = np.full_like(x,19)
                device.check(device.htod(pointers[0],changed.ctypes.data,changed.nbytes) if device.cuda else device.copy(pointers[0],changed.ctypes.data,changed.nbytes,1))
                first = frame.backward(cotangent)
                second = frame.backward(upload(d*2))
                np.testing.assert_allclose(download(first),derivative,atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(download(second),derivative*2,atol=2e-6,rtol=2e-6)
                child = frame.child(pair,source)
                child_result = child.backward(cotangent)
                child_expected = (d * 38 if family == 'square_vjp' else
                                  d * (1 - np.tanh(changed.astype(np.float64))**2) if family == 'tanh_vjp' else
                                  np.full(width, d[0] / (width if family == 'mean_vjp' else 1)))
                np.testing.assert_allclose(download(child_result),child_expected,atol=2e-6,rtol=2e-6)
                frame.close()
                assert child.closed
                try:
                    first.__cuda_array_interface__
                except ValueError:
                    pass
                else:
                    raise AssertionError('closed frame exposed a freed allocation')
                try:
                    frame.backward(cotangent)
                except ValueError:
                    pass
                else:
                    raise AssertionError('closed frame accepted backward')
                rows.append(dict(family=family,width=width,package_digest=package.binding_digest, snapshot_after_input_mutation=True,repeated_backward=True,nested_lifetime=True))
            finally:
                pair.close()
                for pointer in pointers:
                    device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(backend=args.backend,rows=rows,compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),implementation_sha256={name:hashlib.sha256((ROOT/'python/tessera/compiler'/name).read_bytes()).hexdigest() for name in ('native_device_tape.py','native_storage_pair.py')},higher_order_ad=False,control_flow_tape=False),indent=2)+'\n')
    print(len(rows),'persistent device tape cases passed')


if __name__ == '__main__':
    main()
