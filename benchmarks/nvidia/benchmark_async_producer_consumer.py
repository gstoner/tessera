#!/usr/bin/env python3
"""Measure native SM120 GEMM prefetch overlap against an immediate-wait ablation.

Experimental images only: neither image is installed in a selector or ledger.
Inputs stay resident; interleaved CUDA-event samples exclude compilation/copies.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.record_device_ring_protocol import Device
from tessera.compiler import nvidia_native as native


def serialize_prefetch(lowered: str) -> str:
    """Fail closed if the native two-site (prime/prefetch) protocol changes."""
    lines = lowered.splitlines()
    commits = [i for i, line in enumerate(lines) if 'nvvm.cp.async.commit.group' in line]
    if len(commits) != 2:
        raise ValueError('expected exactly prime and loop-prefetch commit sites')
    pending = [i for i in commits if 'nvvm.cp.async.wait.group 0' not in lines[i + 1]]
    if pending != [commits[1]]:
        raise ValueError('expected only loop prefetch to defer its wait')
    i = pending[0]
    lines.insert(i + 1, lines[i][:len(lines[i]) - len(lines[i].lstrip())] + 'nvvm.cp.async.wait.group 0')
    return '\n'.join(lines) + '\n'


def compile_images(directory: Path):
    tools = {name: native._tool(name) for name in ('tessera-nvidia-opt', 'mlir-opt', 'mlir-translate', 'llc', 'ptxas')}
    if any(path is None for path in tools.values()):
        raise RuntimeError(f'missing native compiler tools: {tools}')
    fixture = ROOT / 'src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/sm120_macro_cta_matmul.mlir'
    source = '\n'.join(line for line in fixture.read_text().splitlines() if not line.startswith('//'))
    # Fixture identity is experimental, not a production schedule attestation.
    lowered = native._run([tools['tessera-nvidia-opt'], '--tessera-lower-to-nvidia-sm120'], source.encode()).decode()
    result = {}
    for name, text in [('async', lowered), ('serialized', serialize_prefetch(lowered))]:
        out = directory / name
        out.mkdir(parents=True, exist_ok=True)
        (out / 'input.mlir').write_text(source)
        (out / 'lowered.mlir').write_text(text)
        llvm_mlir = native._run([tools['mlir-opt'], *native._TILE_TO_PTX_MLIR_PASSES,
                                '--convert-math-to-llvm', '--convert-arith-to-llvm',
                                '--convert-cf-to-llvm', '--reconcile-unrealized-casts'], text.encode())
        llvm = native._run([tools['mlir-translate'], '--mlir-to-llvmir'], llvm_mlir)
        (out / 'kernel.ll').write_bytes(llvm)
        ptx = native._run([tools['llc'], '-mtriple=nvptx64-nvidia-cuda', '-mcpu=sm_120a', '-O3'], llvm)
        (out / 'kernel.ptx').write_bytes(ptx)
        assembled = subprocess.run([tools['ptxas'], '-arch=sm_120a', '-v', str(out / 'kernel.ptx'),
                                    '-o', str(out / 'kernel.cubin')], capture_output=True, check=True)
        (out / 'ptxas.txt').write_bytes(assembled.stderr)
        result[name] = (out / 'kernel.cubin').read_bytes()
    return result, {name: {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for name, path in tools.items()}


class Gemm:
    def __init__(self, device, image, shape):
        self.device = device
        self.m, self.k, self.n = shape
        rng = np.random.default_rng(73)
        self.a = (rng.standard_normal((self.m, self.k)) * .25).astype(np.float16)
        self.b = np.asfortranarray((rng.standard_normal((self.k, self.n)) * .25).astype(np.float16))
        self.output = np.empty((self.m, self.n), dtype=np.float32)
        self.buffers = []
        self.module, self.fn = ct.c_void_p(), ct.c_void_p()
        device.check(device.load(ct.byref(self.module), ct.cast(ct.create_string_buffer(image), ct.c_void_p)))
        device.check(device.function(ct.byref(self.fn), self.module, b'macro'))
        for array in (self.a, self.b, self.output):
            ptr = ct.c_void_p()
            device.check(device.alloc(ct.byref(ptr), array.nbytes))
            self.buffers.append(ptr)
            device.check(device.htod(ptr, ct.c_void_p(array.ctypes.data), array.nbytes))
        self.values = self.buffers + [ct.c_int64(self.m), ct.c_int64(self.n), ct.c_int64(self.k)]
        self.args = (ct.c_void_p * 6)(*[ct.cast(ct.pointer(v), ct.c_void_p) for v in self.values])

    def launch(self):
        self.device.check(self.device.launch(self.fn, (self.n + 31)//32, (self.m + 31)//32, 1,
                                             128, 1, 1, 0, None, self.args, None))

    def validate(self):
        self.launch()
        self.device.check(self.device.sync())
        self.device.check(self.device.dtoh(ct.c_void_p(self.output.ctypes.data), self.buffers[2], self.output.nbytes))
        expected = self.a.astype(np.float32) @ self.b.astype(np.float32)
        np.testing.assert_allclose(self.output, expected, atol=2e-4, rtol=2e-4)
        return float(np.max(np.abs(self.output - expected)))

    def measure(self, repeats):
        d = self.device
        start, end = ct.c_void_p(), ct.c_void_p()
        d.check(d.event_create(ct.byref(start), 0))
        d.check(d.event_create(ct.byref(end), 0))
        try:
            d.check(d.event_record(start, None))
            for _ in range(repeats):
                self.launch()
            d.check(d.event_record(end, None))
            d.check(d.event_sync(end))
            elapsed = ct.c_float()
            d.check(d.event_elapsed(ct.byref(elapsed), start, end))
            return elapsed.value / repeats
        finally:
            d.check(d.event_destroy(start))
            d.check(d.event_destroy(end))

    def close(self):
        for ptr in self.buffers:
            self.device.check(self.device.free(ptr))
        self.device.check(self.device.unload(self.module))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=['async', 'serialized'])
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error('samples and repeats must be positive')
    images, tools = compile_images(args.artifacts)
    device = Device('nvidia')
    rows = []
    shapes = [(1024, 1024, 1024)] if args.profile else [(1, 8, 1), (33, 24, 65), (257, 520, 257),
                                                                      (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    for shape in shapes:
        kernels = {name: Gemm(device, image, shape) for name, image in images.items()
                   if args.profile is None or name == args.profile}
        try:
            errors = {name: kernel.validate() for name, kernel in kernels.items()}
            for kernel in kernels.values():
                for _ in range(3):
                    kernel.launch()
            device.check(device.sync())
            samples = {name: [] for name in kernels}
            for trial in range(1 if args.profile else args.samples):
                names = list(kernels)
                if trial % 2:
                    names.reverse()
                for name in names:
                    samples[name].append(kernels[name].measure(1 if args.profile else args.repeats))
            row = {'shape_mkn': shape, 'max_abs_error': errors, 'cuda_event_ms': samples,
                   'median_ms': {name: statistics.median(values) for name, values in samples.items()}}
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            for kernel in kernels.values():
                kernel.close()
    packet = {'schema': 'native_async_gemm_ablation_v1', 'experimental_only': True,
              'timing': 'resident CUDA events; alternating variant order; profiler runs excluded',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'device': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip(),
              'images_sha256': {name: hashlib.sha256(image).hexdigest() for name, image in images.items()},
              'tools': tools, 'profile': args.profile, 'repeats': 1 if args.profile else args.repeats, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + '\n')


if __name__ == '__main__':
    main()
