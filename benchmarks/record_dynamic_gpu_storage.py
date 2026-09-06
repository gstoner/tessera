#!/usr/bin/env python3
"""Exact-device dynamic shared-storage experiment using native MLIR host sizing.

Compile the arena once with a symbolic extent and compare with per-extent static
arenas. The compiler's emitted native host function supplies launch bytes; Python
only drives the launch and numerical oracle. This is an experiment, not a product
package or asynchronous-overlap claim. Run on the owning CUDA/ROCm host.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.record_device_ring_protocol import Device, compile_image  # noqa: E402

FIXTURE = ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_arena.mlir'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], source: str | None = None) -> str:
    return subprocess.check_output(command, input=source, text=True, stderr=subprocess.PIPE, timeout=120)


def extract(source: str, pattern: str) -> str:
    # Only extract top-level blocks from canonical compiler output. Requiring
    # exactly one match avoids accidentally measuring a different entry.
    matches = re.findall(pattern, source, re.MULTILINE | re.DOTALL)
    if len(matches) != 1:
        raise ValueError('expected exactly one emitted native artifact')
    return 'module {\n' + matches[0] + '\n}\n'


def prepare(compiler: Path, mlir_opt: Path, backend: str, directory: Path, width: int | None):
    directory.mkdir(parents=True, exist_ok=True)
    source = FIXTURE.read_text()
    if width is not None:
        source = source.replace('memref.alloca(%n)', 'memref.alloca()').replace('memref<?xf32>', f'memref<{width}xf32>')
    emitted = run([str(compiler), '--allow-unregistered-dialect', '--tessera-tile-buffer-reuse',
                   '--tessera-tile-buffer-arena', '--canonicalize'], source)
    (directory / 'source.mlir').write_text(source)
    (directory / 'arena.mlir').write_text(emitted)
    device_ir = extract(emitted, r'^  gpu.module .*?^  }')
    if width is None and 'gpu.dynamic_shared_memory' not in device_ir:
        raise ValueError('compiler did not materialize dynamic shared storage')
    image = compile_image(mlir_opt, backend, device_ir, directory / 'device')
    if width is not None:
        return image, None
    host_ir = extract(emitted, r'^  func.func @__tessera_shared_bytes_.*?^  }')
    (directory / 'sizer.mlir').write_text(host_ir)
    llvm_ir = run([str(mlir_opt), '--convert-to-llvm', '--reconcile-unrealized-casts'], host_ir)
    (directory / 'sizer-llvm.mlir').write_text(llvm_ir)
    translated = run([str(mlir_opt.parent / 'mlir-translate'), '--mlir-to-llvmir'], llvm_ir)
    (directory / 'sizer.ll').write_text(translated)
    library = directory / 'sizer.so'
    run([str(mlir_opt.parent / 'clang'), '-shared', '-fPIC', '-O2', str(directory / 'sizer.ll'), '-o', str(library)])
    return image, library


def load_sizer(path: Path):
    library = ct.CDLL(str(path.resolve()))
    function = library.__tessera_shared_bytes_dynamic_scratch
    function.argtypes = [ct.c_void_p, ct.c_int64, ct.c_int64]
    function.restype = ct.c_int64
    return function


def reject_invalid_sizes(path: Path) -> list[int]:
    rejected = []
    for size in (-1, 1 << 30, (1 << 63) - 1):
        command = [sys.executable, str(Path(__file__).resolve()), '--check-size', str(path), str(size)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or result.stdout.strip() != '-1':
            raise AssertionError(f'native sizer failed to reject {size}: {result.returncode} {result.stderr}')
        rejected.append(size)
    return rejected


def measure(device: Device, image: bytes, width: int, rounds: int, blocks: int,
            shared_bytes: int, samples: int, reps: int) -> dict:
    P = ct.c_void_p
    dst, module, function, start, end = (P() for _ in range(5))
    expected = np.tile(rounds * ((np.arange(width) + 1) % width) + rounds * (rounds - 1) // 2, blocks).astype(np.float32)
    output = np.empty_like(expected)
    blob = ct.create_string_buffer(image)
    try:
        device.check(device.alloc(ct.byref(dst), output.nbytes))
        device.check(device.load(ct.byref(module), ct.cast(blob, P)))
        device.check(device.function(ct.byref(function), module, b'scratch'))
        n, r = ct.c_int64(width), ct.c_int64(rounds)
        params = (P * 3)(ct.cast(ct.byref(dst), P), ct.cast(ct.byref(n), P), ct.cast(ct.byref(r), P))

        def launch():
            device.check(device.launch(function, blocks, 1, 1, width, 1, 1, shared_bytes, None, params, None))

        def check_output():
            device.check(device.dtoh(output.ctypes.data, dst, output.nbytes) if device.cuda
                         else device.copy(output.ctypes.data, dst, output.nbytes, 2))
            np.testing.assert_array_equal(output, expected)

        resources = {}
        for name, attribute in (('static_shared_bytes', 1), ('local_bytes', 3), ('registers', 4)):
            value = ct.c_int()
            device.check(device.attribute(ct.byref(value), attribute, function))
            resources[name] = value.value
        active = ct.c_int()
        device.check(device.occupancy(ct.byref(active), function, width, shared_bytes))
        resources['active_blocks_per_multiprocessor'] = active.value
        for _ in range(3):
            launch()
        device.check(device.sync())
        check_output()
        device.check(device.event_create(ct.byref(start), 0))
        device.check(device.event_create(ct.byref(end), 0))
        times = []
        for _ in range(samples):
            device.check(device.event_record(start, None))
            for _ in range(reps):
                launch()
            device.check(device.event_record(end, None))
            device.check(device.event_sync(end))
            ms = ct.c_float()
            device.check(device.event_elapsed(ct.byref(ms), start, end))
            times.append(ms.value / reps)
            check_output()
        return {'oracle': 'exact', 'event_ms': times, 'median_event_ms': statistics.median(times),
                'dynamic_shared_bytes': shared_bytes, 'resources': resources,
                'image_sha256': hashlib.sha256(image).hexdigest()}
    finally:
        for handle, destroy in ((start, device.event_destroy), (end, device.event_destroy),
                                (module, device.unload), (dst, device.free)):
            if handle:
                device.check(destroy(handle))


def main():
    # A fresh child calls the compiled companion directly, proving that invalid
    # sizes return native failure rather than relying on a Python bounds check.
    if len(sys.argv) == 4 and sys.argv[1] == '--check-size':
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        print(load_sizer(Path(sys.argv[2]))(None, int(sys.argv[3]), 17), flush=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['nvidia', 'rocm'], required=True)
    parser.add_argument('--tessera-opt', type=Path, required=True)
    parser.add_argument('--mlir-opt', type=Path, default=Path('/usr/lib/llvm-23/bin/mlir-opt'))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--reps', type=int, default=20)
    args = parser.parse_args()
    if args.samples < 3 or args.reps < 1:
        parser.error('at least three samples and one repetition required')
    args.artifacts = args.artifacts.resolve()
    dynamic, library = prepare(args.tessera_opt, args.mlir_opt, args.backend, args.artifacts / 'dynamic', None)
    sizer = load_sizer(library)
    rejected = reject_invalid_sizes(library)
    device = Device(args.backend)
    cases = []
    for width in (32, 64, 128, 256):
        byte_count = sizer(None, width, 17)
        if byte_count != 4 * width:
            raise AssertionError('emitted native sizer disagrees with the fixture storage oracle')
        static, _ = prepare(args.tessera_opt, args.mlir_opt, args.backend, args.artifacts / f'static-{width}', width)
        record = {'threads': width, 'rounds': 17, 'blocks': 256}
        # Alternate first variant across cases to reduce systematic order bias.
        variants = [('dynamic', dynamic, byte_count), ('static', static, 0)]
        if width in (64, 256):
            variants.reverse()
        for name, image, count in variants:
            record[name] = measure(device, image, width, 17, 256, count, args.samples, args.reps)
        record['dynamic_over_static'] = record['dynamic']['median_event_ms'] / record['static']['median_event_ms']
        cases.append(record)
    identity = run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']) if args.backend == 'nvidia' else run(['/opt/rocm/bin/rocminfo'])
    report = {'schema_version': 1, 'status': 'experiment', 'sync_key': 'IR-NATIVE-FOUNDATION-1',
              'backend': args.backend, 'device_identity': identity.strip(),
              'compiler_sha256': digest(args.tessera_opt), 'compiler_path': str(args.tessera_opt.resolve()),
              'mlir_opt_sha256': digest(args.mlir_opt), 'sizer_sha256': digest(library),
              'fixture_sha256': digest(FIXTURE), 'recorder_sha256': digest(Path(__file__)),
              'native_rejected_sizes': rejected, 'samples': args.samples, 'reps': args.reps,
              'timing': 'CUDA/HIP device events; each sample averages reps launches',
              'claim': 'dynamic storage and native host sizing proof; no product integration or async-overlap claim',
              'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'ratios': [r['dynamic_over_static'] for r in cases]}))


if __name__ == '__main__':
    main()
