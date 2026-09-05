#!/usr/bin/env python3
"""Native MLIR cooperative ring experiment; no selector or async-overlap claim.

The same input/output math compares direct streaming with ring depths 1/2/4.
Each slot carries a generation tag, CTA publish and CTA release. Neighbor-lane
consumption makes synchronization necessary. GPU compilation is native MLIR ->
LLVM -> NVPTX/AMDGPU. This is a protocol experiment, not a Tessera product route.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = 128


def module(depth: int) -> str:
    slots = max(depth, 1) * THREADS
    head = f'''module {{
  gpu.module @ring_module {{
    llvm.mlir.global private @slots() {{addr_space = 3 : i32, alignment = 16 : i64}} : !llvm.array<{slots} x f32>
    llvm.mlir.global private @epochs() {{addr_space = 3 : i32, alignment = 8 : i64}} : !llvm.array<{max(depth, 1)} x i64>
    gpu.func @ring(%input: !llvm.ptr<1>, %output: !llvm.ptr<1>, %rounds: i64) kernel {{
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %t = arith.index_cast %tid : index to i64
      %b = arith.index_cast %bid : index to i64
      %zero = arith.constant 0 : i64
      %one = arith.constant 1 : i64
      %width = arith.constant {THREADS} : i64
      %depth = arith.constant {max(depth, 1)} : i64
      %two = arith.constant 2.0 : f32
      %bias = arith.constant 1.0 : f32
      %bad = arith.constant -999.0 : f32
      %neighbor0 = arith.addi %t, %one : i64
      %neighbor = arith.remui %neighbor0, %width : i64
      %rows = arith.muli %b, %rounds : i64
      %base = arith.muli %rows, %width : i64
      %leader = arith.cmpi eq, %t, %zero : i64
      %shared = llvm.mlir.addressof @slots : !llvm.ptr<3>
      %epochs = llvm.mlir.addressof @epochs : !llvm.ptr<3>
'''
    def publish(g: str, slot: str, label: str) -> str:
        return f'''        %{label}row = arith.muli {g}, %width : i64
        %{label}base = arith.addi %base, %{label}row : i64
        %{label}index = arith.addi %{label}base, %t : i64
        %{label}src = llvm.getelementptr %input[%{label}index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
        %{label}value = llvm.load %{label}src : !llvm.ptr<1> -> f32
        %{label}offset = arith.muli {slot}, %width : i64
        %{label}lane = arith.addi %{label}offset, %t : i64
        %{label}dst = llvm.getelementptr %shared[0, %{label}lane] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{slots} x f32>
        llvm.store %{label}value, %{label}dst : f32, !llvm.ptr<3>
        scf.if %leader {{
          %{label}epoch = llvm.getelementptr %epochs[0, {slot}] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{max(depth, 1)} x i64>
          llvm.store {g}, %{label}epoch : i64, !llvm.ptr<3>
        }}
'''
    if depth:
        head += '''      scf.for %prefill = %zero to %depth step %one : i64 {
        %valid = arith.cmpi ult, %prefill, %rounds : i64
        scf.if %valid {
'''+publish('%prefill', '%prefill', 'p')+'''        }
      }
      gpu.barrier
'''
    head += '''      scf.for %generation = %zero to %rounds step %one : i64 {
        %row = arith.muli %generation, %width : i64
        %rowbase = arith.addi %base, %row : i64
        %outindex = arith.addi %rowbase, %t : i64
        %dst = llvm.getelementptr %output[%outindex] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
'''
    if depth:
        head += f'''        %slot = arith.remui %generation, %depth : i64
        %offset = arith.muli %slot, %width : i64
        %lane = arith.addi %offset, %neighbor : i64
        %src = llvm.getelementptr %shared[0, %lane] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{slots} x f32>
        %epochptr = llvm.getelementptr %epochs[0, %slot] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<{depth} x i64>
        %epoch = llvm.load %epochptr : !llvm.ptr<3> -> i64
        %correct = arith.cmpi eq, %epoch, %generation : i64
'''
    else:
        head += '''        %inindex = arith.addi %rowbase, %neighbor : i64
        %src = llvm.getelementptr %input[%inindex] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
'''
    space = 3 if depth else 1
    head += f'''        %value = llvm.load %src : !llvm.ptr<{space}> -> f32
        %scaled = arith.mulf %value, %two : f32
        %result = arith.addf %scaled, %bias : f32
'''
    if depth:
        head += '        %checked = arith.select %correct, %result, %bad : f32\n'
    head += f'        llvm.store %{"checked" if depth else "result"}, %dst : f32, !llvm.ptr<1>\n'
    if depth:
        head += '''        gpu.barrier
        %future = arith.addi %generation, %depth : i64
        %more = arith.cmpi ult, %future, %rounds : i64
        scf.if %more {
'''+publish('%future', '%slot', 'n')+'''        }
        gpu.barrier
'''
    return head+'''      }
      gpu.return
    }
  }
}
'''


def compile_image(tool: Path, backend: str, source: str, directory: Path):
    from benchmarks.rocm.benchmark_rocm_gemm_pipeline_vs_direct import _extract_hsaco
    target = 'nvvm' if backend == 'nvidia' else 'rocdl'
    chip = 'sm_120' if backend == 'nvidia' else 'gfx1151'
    pipeline = ('builtin.module(gpu.module(convert-scf-to-cf,'
                f'convert-gpu-to-{target},reconcile-unrealized-casts),'
                f'{target}-attach-target{{chip={chip}}},gpu-module-to-binary)')
    result = subprocess.run([str(tool), f'--pass-pipeline={pipeline}'], input=source,
                            text=True, capture_output=True, timeout=120)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'input.mlir').write_text(source)
    (directory/'compiler.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    (directory/'output.mlir').write_text(result.stdout)
    if 'bin = "' in result.stdout:
        image = _extract_hsaco(result.stdout)
    else:
        # NVVM uses the positional gpu.object assembly form. Exactly one
        # object is emitted; the final MLIR string literal is its binary.
        literals = re.findall(r'"((?:\\.|[^"\\])*)"', result.stdout)
        if result.stdout.count('#gpu.object<') != 1 or not literals:
            raise RuntimeError('expected one serialized native GPU object')
        image = _extract_hsaco('bin = "'+literals[-1]+'"')
    (directory/'kernel.bin').write_bytes(image)
    return image


class Device:
    def __init__(self, backend):
        self.cuda = backend == 'nvidia'
        self.lib = ct.CDLL('libcuda.so.1' if self.cuda else 'libamdhip64.so')
        def bind(name, args, cuda_name=None):
            fn = getattr(self.lib, cuda_name if self.cuda else name)
            fn.argtypes, fn.restype = args, ct.c_int
            return fn
        P, U, S = ct.c_void_p, ct.c_uint, ct.c_size_t
        self.alloc = bind('hipMalloc', [ct.POINTER(P), S], 'cuMemAlloc_v2')
        self.free = bind('hipFree', [P], 'cuMemFree_v2')
        self.load = bind('hipModuleLoadData', [ct.POINTER(P), P], 'cuModuleLoadData')
        self.unload = bind('hipModuleUnload', [P], 'cuModuleUnload')
        self.function = bind('hipModuleGetFunction', [ct.POINTER(P), P, ct.c_char_p], 'cuModuleGetFunction')
        self.launch = bind('hipModuleLaunchKernel', [P]+[U]*7+[P, ct.POINTER(P), ct.POINTER(P)], 'cuLaunchKernel')
        self.attribute = bind('hipFuncGetAttribute', [ct.POINTER(ct.c_int), ct.c_int, P], 'cuFuncGetAttribute')
        self.occupancy = bind('hipModuleOccupancyMaxActiveBlocksPerMultiprocessor',
                              [ct.POINTER(ct.c_int), P, ct.c_int, S], 'cuOccupancyMaxActiveBlocksPerMultiprocessor')
        self.sync = bind('hipDeviceSynchronize', [], 'cuCtxSynchronize')
        self.event_create = bind('hipEventCreateWithFlags', [ct.POINTER(P), U], 'cuEventCreate')
        self.event_record = bind('hipEventRecord', [P, P], 'cuEventRecord')
        self.event_sync = bind('hipEventSynchronize', [P], 'cuEventSynchronize')
        self.event_elapsed = bind('hipEventElapsedTime', [ct.POINTER(ct.c_float), P, P], 'cuEventElapsedTime')
        self.event_destroy = bind('hipEventDestroy', [P], 'cuEventDestroy_v2')
        self.check(bind('hipInit', [U], 'cuInit')(0))
        if self.cuda:
            self.context = P()
            create = getattr(self.lib, 'cuCtxCreate_v2')
            create.argtypes = [ct.POINTER(P), U, ct.c_int]
            self.check(create(ct.byref(self.context), 0, 0))
            self.htod = bind('', [P, P, S], 'cuMemcpyHtoD_v2')
            self.dtoh = bind('', [P, P, S], 'cuMemcpyDtoH_v2')
        else:
            self.copy = bind('hipMemcpy', [P, P, S, ct.c_int])

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'GPU API status {code}')

    def run(self, image, rounds, blocks, reps):
        P = ct.c_void_p
        values = np.arange(blocks*rounds*THREADS, dtype=np.float32) % 127
        expected = np.roll(values.reshape(blocks, rounds, THREADS), -1, axis=2).reshape(-1)*2+1
        output = np.zeros_like(values)
        src, dst, mod, fn, start, end = (P() for _ in range(6))
        storage = ct.create_string_buffer(image)
        try:
            self.check(self.alloc(ct.byref(src), values.nbytes))
            self.check(self.alloc(ct.byref(dst), values.nbytes))
            self.check(self.htod(src, values.ctypes.data, values.nbytes) if self.cuda
                       else self.copy(src, values.ctypes.data, values.nbytes, 1))
            self.check(self.load(ct.byref(mod), ct.cast(storage, P)))
            self.check(self.function(ct.byref(fn), mod, b'ring'))
            resources = {}
            for key, attribute in (('shared_bytes', 1), ('local_bytes', 3), ('registers', 4)):
                value = ct.c_int()
                self.check(self.attribute(ct.byref(value), attribute, fn))
                resources[key] = value.value
            active = ct.c_int()
            self.check(self.occupancy(ct.byref(active), fn, THREADS, 0))
            resources['active_blocks_per_multiprocessor'] = active.value
            count = ct.c_int64(rounds)
            args = (P*3)(ct.cast(ct.byref(src), P), ct.cast(ct.byref(dst), P), ct.cast(ct.byref(count), P))
            def launch():
                self.check(self.launch(fn, blocks, 1, 1, THREADS, 1, 1, 0, None, args, None))
            for _ in range(3):
                launch()
            self.check(self.sync())
            self.check(self.dtoh(output.ctypes.data, dst, output.nbytes) if self.cuda
                       else self.copy(output.ctypes.data, dst, output.nbytes, 2))
            np.testing.assert_array_equal(output, expected)
            self.check(self.event_create(ct.byref(start), 0))
            self.check(self.event_create(ct.byref(end), 0))
            wall = time.perf_counter()
            self.check(self.event_record(start, None))
            for _ in range(reps):
                launch()
            self.check(self.event_record(end, None))
            self.check(self.event_sync(end))
            ms = ct.c_float()
            self.check(self.event_elapsed(ct.byref(ms), start, end))
            host_ms = (time.perf_counter()-wall)*1000/reps
            self.check(self.dtoh(output.ctypes.data, dst, output.nbytes) if self.cuda
                       else self.copy(output.ctypes.data, dst, output.nbytes, 2))
            np.testing.assert_array_equal(output, expected)
            return {'event_ms': ms.value/reps, 'host_ms': host_ms, 'resources': resources}
        finally:
            for handle, destroy in ((start, self.event_destroy), (end, self.event_destroy),
                                    (mod, self.unload), (src, self.free), (dst, self.free)):
                if handle:
                    self.check(destroy(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['nvidia', 'rocm'], required=True)
    parser.add_argument('--mlir-opt', type=Path, default=Path('/usr/lib/llvm-23/bin/mlir-opt'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile-depth', type=int, choices=[0, 1, 2, 4])
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--reps', type=int, default=20)
    args = parser.parse_args()
    if args.samples < 3 or args.reps < 1:
        parser.error('at least 3 samples and 1 repetition required')
    device = Device(args.backend)
    if args.profile_depth is not None:
        image = (args.artifacts/str(args.profile_depth)/'kernel.bin').read_bytes()
        result = device.run(image, 257, 64, 1)
        print(json.dumps({'profile_only': True, 'depth': args.profile_depth, 'result': result}))
        return
    if args.output is None:
        parser.error('--output is required for an unprofiled measurement')
    images = {d: compile_image(args.mlir_opt, args.backend, module(d), args.artifacts/str(d))
              for d in (0, 1, 2, 4)}
    faulty_source = module(2).replace('llvm.store %future, %nepoch', 'llvm.store %zero, %nepoch')
    faulty = compile_image(args.mlir_opt, args.backend, faulty_source, args.artifacts/'stale-generation')
    try:
        device.run(faulty, 37, 4, 1)
    except AssertionError:
        stale_generation_rejected = True
    else:
        raise RuntimeError('generation oracle accepted an intentionally stale ring slot')
    rows = {d: {'depth': d, 'oracle_rounds': [], 'samples': [],
                'image_sha256': hashlib.sha256(image).hexdigest(),
                'source_sha256': hashlib.sha256(module(d).encode()).hexdigest(),
                'declared_shared_bytes': d*(THREADS*4+8),
                'barriers_per_round': 2 if d else 0}
            for d, image in images.items()}
    for d, image in images.items():
        for rounds in sorted({1, max(1, d-1), max(1, d), d+1, 37, 257}):
            device.run(image, rounds, 4, 1)
            rows[d]['oracle_rounds'].append(rounds)
    for trial in range(args.samples):
        order = list(images)
        if trial % 2:
            order.reverse()
        for d in order:
            rows[d]['samples'].append(device.run(images[d], 257, 64, args.reps))
    for row in rows.values():
        row['event_median_ms'] = statistics.median(x['event_ms'] for x in row['samples'])
    packet = {'schema': 'tessera.native-ring-protocol.v1', 'backend': args.backend,
              'sync_key': 'IR-NATIVE-FOUNDATION-1', 'threads': THREADS,
              'timing_shape': {'rounds': 257, 'blocks': 64}, 'repetitions': args.reps,
              'stale_generation_rejected': stale_generation_rejected,
              'device': subprocess.check_output(
                  ['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']
                  if args.backend == 'nvidia' else ['/opt/rocm/bin/rocminfo'], text=True).strip(),
              'clock': 'CUDA events' if args.backend == 'nvidia' else 'HIP events',
              'compiler_sha256': hashlib.sha256(args.mlir_opt.read_bytes()).hexdigest(),
              'compiler_version': subprocess.check_output([str(args.mlir_opt), '--version'], text=True).strip(),
              'scope': 'native MLIR cooperative protocol experiment; no asynchronous overlap or production-route promotion',
              'rows': list(rows.values())}
    args.output.write_text(json.dumps(packet, indent=2)+'\n')
    print(f'Wrote {args.output}: all generation and numerical oracles passed')


if __name__ == '__main__':
    main()
