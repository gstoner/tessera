#!/usr/bin/env python3
"""Owning-device mixed storage and SAVE/HYBRID/recompute execution proof."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_persistent_split_tape import source  # noqa: E402
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def mixed_source():
    return '''module {
      func.func @mixed(%x: tensor<4xf32>, %w: tensor<4xf64>) -> (tensor<4xf32>, tensor<4xf64>)
          attributes {tessera.autodiff = "reverse"} {
        %a = "tessera.mul"(%x, %x) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %b = "tessera.mul"(%w, %w) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
        return %a, %b : tensor<4xf32>, tensor<4xf64>
      }
    }'''


def checkpoint_source(policy):
    text = source(outer=3).replace('checkpoint_indices = array<i64: 1>',
                                  'checkpoint_indices = array<i64: 1, 2>')
    if policy == 'hybrid':
        text = text.replace('array<i64: 1, 2>', 'array<i64: 1>')
    return text.replace('checkpoint_policy = "save"', f'checkpoint_policy = "{policy}"')



def counted_while_source():
    return '''module {
      func.func @counted(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32>
          attributes {tessera.autodiff = "reverse"} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %three = arith.constant 3 : index
        %count, %out = "scf.while"(%zero, %x) ({
        ^bb0(%i: index, %state: tensor<4xf32>):
          %continue = arith.cmpi slt, %i, %three : index
          scf.condition(%continue) %i, %state : index, tensor<4xf32>
        }, {
        ^bb0(%i: index, %state: tensor<4xf32>):
          %next = "tessera.mul"(%state, %w) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %next_i = arith.addi %i, %one : index
          scf.yield %next_i, %next : index, tensor<4xf32>
        }) {tessera.autodiff.checkpoint_policy = "save", tessera.autodiff.max_iters = 3 : i64} :
          (index, tensor<4xf32>) -> (index, tensor<4xf32>)
        return %out : tensor<4xf32>
      }
    }'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia', 'rocm'), required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = Device(args.backend)
    rows = []
    for policy in ('mixed', 'save', 'hybrid', 'recompute_all', 'counted_while'):
        text = mixed_source() if policy == 'mixed' else counted_while_source() if policy == 'counted_while' else checkpoint_source(policy)
        exponent = 3 if policy == 'counted_while' else 9
        pair = materialize_persistent_tape(text, compiler=args.compiler,
            llvm_bin='/usr/lib/llvm-23/bin', backend=args.backend,
            chip='sm_120' if device.cuda else 'gfx1151')
        pointers = []
        frame = None
        def upload(value):
            pointer = ct.c_void_p()
            device.check(device.alloc(ct.byref(pointer), value.nbytes))
            pointers.append(pointer)
            device.check(device.htod(pointer, value.ctypes.data, value.nbytes) if device.cuda else
                         device.copy(pointer, value.ctypes.data, value.nbytes, 1))
            return SimpleNamespace(__cuda_array_interface__=dict(version=3, shape=value.shape,
                typestr=value.dtype.str, data=(pointer.value, False)))
        def download(value):
            view = value.__cuda_array_interface__
            output = np.empty(view['shape'], np.dtype(view['typestr']))
            pointer = ct.c_void_p(view['data'][0])
            device.check(device.dtoh(output.ctypes.data, pointer, output.nbytes) if device.cuda else
                         device.copy(output.ctypes.data, pointer, output.nbytes, 2))
            return output
        try:
            x = np.linspace(.1, .7, 4, dtype=np.float32)
            w = np.linspace(.8, 1.1, 4, dtype=np.float64 if policy == 'mixed' else np.float32)
            frame = pair.capture(upload(x), upload(w))
            retained = sum(b.nbytes for b in frame.buffers)
            expected = (x*x, w*w) if policy == 'mixed' else (x*w**exponent,)
            for value, ref in zip(frame.primals, expected, strict=True):
                np.testing.assert_allclose(download(value), ref,
                    rtol=1e-12 if ref.dtype == np.float64 else 1e-5,
                    atol=1e-12 if ref.dtype == np.float64 else 1e-6)
            saved = [download(value).copy() for value in frame.residuals]
            for factor in (1, 2):
                cotangents = [upload(np.full_like(ref, factor)) for ref in expected]
                derivatives = frame.backward(*cotangents)
                refs = (2*factor*x, 2*factor*w) if policy == 'mixed' else (factor*w**exponent, factor*exponent*x*w**(exponent-1))
                for value, ref in zip(derivatives, refs, strict=True):
                    actual = download(value)
                    assert actual.dtype == ref.dtype
                    np.testing.assert_allclose(actual, ref,
                        rtol=1e-12 if ref.dtype == np.float64 else 1e-5,
                        atol=1e-12 if ref.dtype == np.float64 else 1e-6)
                for value, old in zip(frame.residuals, saved, strict=True):
                    np.testing.assert_array_equal(download(value), old)
            rows.append(dict(policy=policy, source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                lineage=pair.lineage_digest, forward=pair.forward.binding_digest,
                backward=pair.backward.binding_digest, retained_bytes=retained,
                residual_bytes=sum(v.nbytes for v in saved), repeated_backward=True))
        finally:
            if frame is not None:
                frame.close()
            for pointer in pointers:
                device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(backend=args.backend, chip='sm_120' if device.cuda else 'gfx1151',
        rows=rows, compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'src/transforms/lib/NativeTapeToGPUPass.cpp', 'src/transforms/lib/AutodiffPairedPass.cpp',
            'python/tessera/compiler/native_persistent_tape.py',
            'python/tessera/compiler/native_device_tape.py')},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        promotion_eligible=False), indent=2)+'\n')
    print(len(rows), 'mixed/checkpoint device cases passed')


if __name__ == '__main__':
    main()
