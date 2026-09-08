#!/usr/bin/env python3
"""Owning-device checked async tickets; injected status failure is explicit."""
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
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia', 'rocm'), required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = Device(args.backend)
    chip = 'sm_120' if device.cuda else 'gfx1151'
    source = '''module { func.func @square(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
      %y = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
      return %y : tensor<4xf32> } }'''
    pair = materialize_persistent_tape(source, compiler=args.compiler,
        llvm_bin=Path('/usr/lib/llvm-23/bin'), backend=args.backend, chip=chip, checked_status=True)
    pointers, streams = [], []
    def copy_in(pointer, array):
        device.check(device.htod(pointer, array.ctypes.data, array.nbytes) if device.cuda
                     else device.copy(pointer, array.ctypes.data, array.nbytes, 1))
    def upload(array):
        pointer = ct.c_void_p()
        device.check(device.alloc(ct.byref(pointer), array.nbytes)); pointers.append(pointer)
        copy_in(pointer, array)
        return SimpleNamespace(__cuda_array_interface__=dict(version=3, shape=array.shape,
            typestr=array.dtype.str, data=(pointer.value, False)))
    def download(view):
        array = np.empty(4, np.float32)
        pointer = view.__cuda_array_interface__['data'][0]
        device.check(device.dtoh(array.ctypes.data, pointer, array.nbytes) if device.cuda
                     else device.copy(array.ctypes.data, pointer, array.nbytes, 2))
        return array
    create = getattr(device.lib, 'cuStreamCreate' if device.cuda else 'hipStreamCreateWithFlags')
    create.argtypes, create.restype = [ct.POINTER(ct.c_void_p), ct.c_uint], ct.c_int
    destroy = getattr(device.lib, 'cuStreamDestroy_v2' if device.cuda else 'hipStreamDestroy')
    destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int
    rows = []
    try:
        for _ in range(2):
            stream = ct.c_void_p(); device.check(create(ct.byref(stream), 1)); streams.append(stream)
        with pair.capture(upload(np.full(4, 3, np.float32))) as frame:
            generations = [frame.backward_async(stream.value, upload(np.full(4, i+1, np.float32)))
                           for i, stream in enumerate(streams)]
            assert generations[0]._status_buffer.pointer.value != generations[1]._status_buffer.pointer.value
            for i, generation in enumerate(generations):
                try:
                    generation.outputs
                except ValueError as error:
                    assert 'successful wait or poll' in str(error)
                else:
                    raise AssertionError('unchecked output was exposed')
                np.testing.assert_allclose(download(generation.wait()[0]), np.full(4, 6*(i+1), np.float32))
                rows.append(dict(case='independent_checked_generation', derivative=6*(i+1)))
            for generation in generations:
                generation.release()
            failed = frame.backward_async(streams[0].value, upload(np.ones(4, np.float32)))
            failed.submission.wait()
            # Fault injection tests the reader boundary, not a natural kernel guard.
            copy_in(failed._status_buffer.pointer, np.ones(1, np.int64))
            try:
                failed.wait()
            except RuntimeError as error:
                assert 'guard failed' in str(error)
            else:
                raise AssertionError('failed status exposed derivative')
            failed.release()
            rows.append(dict(case='injected_status_failure', output_refused=True, released=True))
    finally:
        for pointer in pointers:
            device.check(device.free(pointer))
        for stream in streams:
            device.check(destroy(stream))
    files = ['python/tessera/compiler/native_persistent_tape.py', 'python/tessera/compiler/native_gpu_tensor.py',
             'src/transforms/lib/NativeTapeToGPUPass.cpp']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(backend=args.backend, chip=chip, rows=rows,
        promotion_eligible=False, overlap_measured=False,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    print('Independent checked generations and injected status refusal passed')


if __name__ == '__main__':
    main()
