#!/usr/bin/env python3
"""Execute serialized native kernel/sizer packages on each owning device."""
from __future__ import annotations
import argparse
import ctypes as ct
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_gpu_storage import build_native_gpu_storage, NativeGPUStoragePackage  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['nvidia', 'rocm'])
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    device = Device(args.backend)
    cases = []
    for mode in (('nested', 'async') if args.backend == 'nvidia' else ('nested',)):
        fixture = ROOT / f'tests/tessera-ir/phase3/tile_dynamic_gpu_{mode}_device.mlir'
        source = fixture.read_text()
        package = build_native_gpu_storage(source, compiler=args.compiler,
            llvm_bin=Path('/usr/lib/llvm-23/bin'), backend=args.backend,
            chip='sm_120' if args.backend == 'nvidia' else 'gfx1151')
        serialized = package.to_json()
        (args.artifacts / f'{mode}.json').write_text(serialized)
        (args.artifacts / f'{mode}.bin').write_bytes(package.image)
        restored = NativeGPUStoragePackage.from_json(serialized, expected_digest=package.binding_digest)
        try:
            replace(restored, host_library=restored.host_library + b'corrupt').bind()
        except ValueError:
            pass
        else:
            raise AssertionError('modified sizing companion was accepted')
        rejected = []
        if mode == 'async':
            for name, wait in [('missing_wait', ''), ('partial_wait', 'nvgpu.device_async_wait %group {numGroups = 1 : i32}'),
                               ('wrong_group', '%unrelated = nvgpu.device_async_create_group\n        nvgpu.device_async_wait %unrelated')]:
                negative = source.replace('nvgpu.device_async_wait %group', wait)
                result = subprocess.run([str(args.compiler), '--allow-unregistered-dialect', '--tessera-tile-buffer-reuse',
                                         '--tessera-tile-buffer-arena'], input=negative, text=True, capture_output=True, timeout=30)
                if result.returncode == 0 or 'dynamic GPU arena requires uniform structured kernel regions' not in result.stderr:
                    raise AssertionError(f'{name} was not rejected for its incomplete lifetime: {result.stderr}')
                rejected.append(name)
        with restored.bind() as bound:
            for width, rounds in ((32, 1), (64, 7), (128, 17), (256, 33)):
                blocks = 32
                inputs = np.arange(blocks * width, dtype=np.float32) % 127
                output = np.zeros_like(inputs)
                src, dst = ct.c_void_p(), ct.c_void_p()
                try:
                    device.check(device.alloc(ct.byref(src), inputs.nbytes))
                    device.check(device.alloc(ct.byref(dst), output.nbytes))
                    device.check(device.htod(src, inputs.ctypes.data, inputs.nbytes) if device.cuda
                                 else device.copy(src, inputs.ctypes.data, inputs.nbytes, 1))
                    values = (src.value, dst.value, width, rounds) if mode == 'async' else (dst.value, width, rounds)
                    count = bound.launch(values, grid=(blocks, 1, 1), block=(width, 1, 1))
                    assert count == width * 4
                    device.check(device.dtoh(output.ctypes.data, dst, output.nbytes) if device.cuda
                                 else device.copy(output.ctypes.data, dst, output.nbytes, 2))
                    expected = (np.roll(inputs.reshape(blocks, width), -1, axis=1).ravel() * rounds if mode == 'async'
                                else np.tile(rounds * ((np.arange(width) + 1) % width) + rounds * (rounds - 1) // 2, blocks).astype(np.float32))
                    np.testing.assert_array_equal(output, expected)
                    oversized = values[:-2] + (1 << 30, rounds)
                    try:
                        bound.launch(oversized, grid=(blocks, 1, 1), block=(width, 1, 1))
                    except ValueError as error:
                        assert 'sizing companion rejected' in str(error)
                    else:
                        raise AssertionError('native sizing failure reached the GPU')
                    cases.append({'mode': mode, 'width': width, 'rounds': rounds, 'blocks': blocks,
                        'dynamic_bytes': count, 'oracle': 'exact', 'binding_digest': package.binding_digest,
                        'fixture_sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'image_sha256': hashlib.sha256(package.image).hexdigest(),
                        'host_library_sha256': hashlib.sha256(package.host_library).hexdigest(),
                        'rejected_protocols': rejected, 'oversized_launch': 'rejected before dispatch'})
                finally:
                    if src:
                        device.check(device.free(src))
                    if dst:
                        device.check(device.free(dst))
    identity = subprocess.check_output(
        ['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']
        if args.backend == 'nvidia' else ['/opt/rocm/bin/rocminfo'], text=True, timeout=15)
    report = {'backend': args.backend, 'device_identity': identity.strip(), 'compiler_sha256': hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
              'recorder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'status': 'raw native package execution proven', 'sync_key': 'IR-NATIVE-FOUNDATION-1',
              'timing_claim': None, 'cases': cases}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'cases': len(cases)}))


if __name__ == '__main__':
    main()
