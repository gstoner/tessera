#!/usr/bin/env python3
"""Exact-device tensor/JIT binding of architecture-owned dynamic producers."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_gpu_storage import build_native_gpu_storage  # noqa: E402
from tessera.compiler.native_gpu_tensor import TensorSpec, IndexSpec  # noqa: E402
from tessera.compiler.jit import JitFn  # noqa: E402
from tessera.compiler.graph_ir import GraphIRModule  # noqa: E402
from tessera.compiler.constraints import ConstraintSolver  # noqa: E402
from tessera.compiler.effects import Effect  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replacement-tokens', action='store_true')
    parser.add_argument('--generated', action='store_true')
    parser.add_argument('--arbiter', action='store_true')
    parser.add_argument('--streams', action='store_true')
    parser.add_argument('--backend', required=True, choices=['nvidia', 'rocm'])
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.replacement_tokens and args.backend != 'nvidia':
        parser.error('replacement-token fixture is NVIDIA-only')
    mode = ('token_replacement' if args.replacement_tokens else 'async_forwarded') if args.backend == 'nvidia' else 'rocm_prefetch_device'
    fixture = ROOT / f'tests/tessera-ir/phase3/tile_dynamic_gpu_{mode}.mlir'
    source = fixture.read_text()
    if args.generated:
        from tessera.compiler.native_storage_contract import attach_tensor_contract
        source = attach_tensor_contract(source,
            (TensorSpec('input', 'float32', (32, 'n') if args.backend == 'nvidia' else ('rounds', 32, 'n')),
             TensorSpec('output', 'float32', (32, 'n'), True), IndexSpec('n'), IndexSpec('rounds')),
            grid=(32, 1, 1), block=('n', 1, 1))
    package = build_native_gpu_storage(source, compiler=args.compiler,
        llvm_bin=Path('/usr/lib/llvm-23/bin'), backend=args.backend,
        chip='sm_120' if args.backend == 'nvidia' else 'gfx1151')
    (args.artifacts / 'package.json').write_text(package.to_json())
    (args.artifacts / 'image.bin').write_bytes(package.image)
    def kernel(input, output, n, rounds):
        raise AssertionError('native JIT must not execute or trace the Python body')
    device = Device(args.backend)
    cases = []
    for width, rounds in ((32, 1), (64, 7), (128, 17), (256, 33)):
        blocks = 32
        shape = (blocks, width) if device.cuda else (rounds, blocks, width)
        inputs = (np.arange(np.prod(shape), dtype=np.float32) % 127).reshape(shape)
        output = np.zeros((blocks, width), dtype=np.float32)
        src, dst = ct.c_void_p(), ct.c_void_p()
        jit = JitFn(kernel, GraphIRModule(), Effect.memory, ConstraintSolver())
        if args.generated:
            jit.bind_native_storage(package)
        else:
            jit.bind_native_storage(package, (TensorSpec('input', 'float32', (blocks, 'n') if device.cuda else ('rounds', blocks, 'n')),
                TensorSpec('output', 'float32', (blocks, 'n'), True), IndexSpec('n'), IndexSpec('rounds')),
                grid=(blocks, 1, 1), block=('n', 1, 1))
        def tensor(ptr, array):
            return SimpleNamespace(__cuda_array_interface__=dict(version=3, shape=array.shape,
                typestr=array.dtype.str, data=(ptr.value, False), strides=None))
        try:
            device.check(device.alloc(ct.byref(src), inputs.nbytes))
            device.check(device.alloc(ct.byref(dst), output.nbytes))
            device.check(device.htod(src, inputs.ctypes.data, inputs.nbytes) if device.cuda
                         else device.copy(src, inputs.ctypes.data, inputs.nbytes, 1))
            result = tensor(dst, output)
            if args.arbiter:
                expected_probe = (np.roll(inputs, -1, axis=-1) * rounds if device.cuda
                                  else (np.roll(inputs, -1, axis=-1) * 2 + 1).sum(axis=0))
                def oracle(candidate, region, **_):
                    candidate.run(region, tensor(src, inputs), result, width, rounds)
                    device.check(device.dtoh(output.ctypes.data, dst, output.nbytes) if device.cuda
                                 else device.copy(output.ctypes.data, dst, output.nbytes, 2))
                    return bool(np.array_equal(output, expected_probe))
                jit.enable_native_storage_arbiter(oracle)
                assert jit(tensor(src, inputs), result, width, rounds) is result
            if args.streams:
                driver = device.lib
                create = getattr(driver, 'cuStreamCreate' if device.cuda else 'hipStreamCreateWithFlags')
                destroy = getattr(driver, 'cuStreamDestroy_v2' if device.cuda else 'hipStreamDestroy')
                create.argtypes, create.restype = [ct.POINTER(ct.c_void_p), ct.c_uint], ct.c_int
                destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int
                streams = [ct.c_void_p(), ct.c_void_p()]
                try:
                    for stream in streams:
                        device.check(create(ct.byref(stream), 1))
                    first = jit.submit_native_storage(streams[0].value, tensor(src, inputs), result, width, rounds)
                    second = jit.submit_native_storage(streams[1].value, tensor(src, inputs), result, width, rounds)
                    assert second.wait() is result
                    first.wait()
                finally:
                    jit.close_native_storage()
                    for stream in streams:
                        if stream:
                            device.check(destroy(stream))
            else:
                assert jit(tensor(src, inputs), result, width, rounds) is result
            device.check(device.dtoh(output.ctypes.data, dst, output.nbytes) if device.cuda
                         else device.copy(output.ctypes.data, dst, output.nbytes, 2))
            expected = (np.roll(inputs, -1, axis=-1) * rounds if device.cuda
                        else (np.roll(inputs, -1, axis=-1) * 2 + 1).sum(axis=0))
            np.testing.assert_array_equal(output, expected)
            cases.append(dict(width=width, rounds=rounds, blocks=blocks, oracle='exact',
                              tensor_binding=jit._native_storage_call.binding_digest))
        finally:
            jit.close_native_storage()
            if src:
                device.check(device.free(src))
            if dst:
                device.check(device.free(dst))
    identity = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']
        if device.cuda else ['/opt/rocm/bin/rocminfo'], text=True, timeout=15)
    args.output.write_text(json.dumps(dict(backend=args.backend, device_identity=identity,
        sync_key='IR-NATIVE-FOUNDATION-1', status='explicit tensor/JIT package execution proven',
        timing_claim=None, generated_descriptors=args.generated, arbiter=args.arbiter, stream_submissions=args.streams, producer=mode, cases=cases, binding_digest=package.binding_digest,
        runtime_sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'python/tessera/compiler/jit.py', 'python/tessera/compiler/native_gpu_storage.py',
            'python/tessera/compiler/native_gpu_tensor.py', 'python/tessera/compiler/native_storage_contract.py',
            'python/tessera/compiler/emit/native_storage_candidate.py')},
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        image_sha256=hashlib.sha256(package.image).hexdigest(), compiler_sha256=package.compiler_digest), indent=2) + '\n')
    print(f'{args.backend}: {len(cases)} exact tensor/JIT cases')


if __name__ == '__main__':
    main()
