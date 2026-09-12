#!/usr/bin/env python3
"""Exact basic-arithmetic probes through MLIR/LLVM and native CUDA/HIP images.

This bypasses general frontend capture deliberately: a result proves the stated
scalar/vector backend primitive, not public dtype admission or matrix support.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import ml_dtypes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tessera.compiler.native_gpu_storage import build_native_gpu_storage  # noqa: E402

DTYPES = {
    'bool': ('i1', np.bool_), 'complex64': ('f32', np.complex64),
    'complex128': ('f64', np.complex128),
    'fp64': ('f64', np.float64), 'fp32': ('f32', np.float32),
    'fp16': ('f16', np.float16), 'bf16': ('bf16', ml_dtypes.bfloat16),
    'fp8_e4m3': ('f8E4M3FN', ml_dtypes.float8_e4m3fn),
    'fp8_e5m2': ('f8E5M2', ml_dtypes.float8_e5m2),
    **{f'{sign}int{bits}': (f'i{bits}', getattr(np, f'{sign}int{bits}'))
       for sign in ('', 'u') for bits in (8, 16, 32, 64)},
}


def emit(dtype, lanes=1):
    if dtype.startswith('complex'):
        return emit_complex(dtype, lanes)
    ty = DTYPES[dtype][0]
    ty = ty if lanes == 1 else f'vector<{lanes}x{ty}>'
    floating = dtype.startswith('fp') or dtype == 'bf16'
    fp8 = dtype.startswith('fp8')
    boolean = dtype == 'bool'
    memory_ty = ('i8' if lanes == 1 else f'vector<{lanes}xi8>') if fp8 or boolean else ty
    compute_ty = ('f32' if lanes == 1 else f'vector<{lanes}xf32>') if fp8 else ty
    suffix = 'f' if floating else 'i'
    lines = ['module {', 'gpu.module @dtype_probe {',
             'gpu.func @arithmetic(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %add: !llvm.ptr<1>, %sub: !llvm.ptr<1>, %mul: !llvm.ptr<1>, %div: !llvm.ptr<1>, %scratch: index) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {',
             '%marker = memref.alloca(%scratch) : memref<?xf32>',
             '"tile.alloc_shared"(%marker) : (memref<?xf32>) -> ()',
             '%bid = gpu.block_id x', '%i = arith.index_cast %bid : index to i64']
    for name in ('a', 'b', 'add', 'sub', 'mul', 'div'):
        lines.append(f'%{name}p = llvm.getelementptr %{name}[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {memory_ty}')
    for name, ptr in (('x','ap'), ('y','bp')):
        if boolean:
            zero = '0' if lanes == 1 else 'dense<0>'
            lines += [f'%{name}bits = llvm.load %{ptr} : !llvm.ptr<1> -> {memory_ty}',
                      f'%{name}zero = arith.constant {zero} : {memory_ty}',
                      f'%{name} = arith.cmpi ne, %{name}bits, %{name}zero : {memory_ty}']
        elif fp8:
            lines += [f'%{name}bits = llvm.load %{ptr} : !llvm.ptr<1> -> {memory_ty}',
                      f'%{name}low = arith.bitcast %{name}bits : {memory_ty} to {ty}',
                      f'%{name} = arith.extf %{name}low : {ty} to {compute_ty}']
        else:
            lines += [f'%{name} = llvm.load %{ptr} : !llvm.ptr<1> -> {ty}']
    for op in ('add', 'sub', 'mul', 'div'):
        instruction = op + suffix if op != 'div' or floating else ('divui' if dtype.startswith('uint') else 'divsi')
        instruction = {'add':'ori', 'sub':'xori', 'mul':'andi', 'div':'cmpi eq,'}[op] if boolean else instruction
        lines += [f'%{op}v = arith.{instruction} %x, %y : {compute_ty}']
        if boolean:
            lines += [f'%{op}bits = arith.extui %{op}v : {ty} to {memory_ty}',
                      f'llvm.store %{op}bits, %{op}p : {memory_ty}, !llvm.ptr<1>']
        elif fp8:
            lines += [f'%{op}low = arith.truncf %{op}v : {compute_ty} to {ty}',
                      f'%{op}bits = arith.bitcast %{op}low : {ty} to {memory_ty}',
                      f'llvm.store %{op}bits, %{op}p : {memory_ty}, !llvm.ptr<1>']
        else:
            lines += [f'llvm.store %{op}v, %{op}p : {ty}, !llvm.ptr<1>']
    return '\n'.join(lines + ['gpu.return', '}', '}', '}'])


def emit_complex(dtype, lanes):
    # Interleaved real/imaginary storage; bounded finite division probes. This
    # explicitly tests component lowering, not a general stable complex divide.
    scalar = DTYPES[dtype][0]
    lines = ['module {', 'gpu.module @dtype_probe {',
        'gpu.func @arithmetic(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %add: !llvm.ptr<1>, %sub: !llvm.ptr<1>, %mul: !llvm.ptr<1>, %div: !llvm.ptr<1>, %scratch: index) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {',
        '%marker = memref.alloca(%scratch) : memref<?xf32>',
        '"tile.alloc_shared"(%marker) : (memref<?xf32>) -> ()',
        '%bid = gpu.block_id x', '%i = arith.index_cast %bid : index to i64',
        f'%width = arith.constant {lanes*2} : i64', '%base = arith.muli %i, %width : i64']
    for lane in range(lanes):
        pre = f'l{lane}'
        for component in range(2):
            lines += [f'%{pre}c{component} = arith.constant {lane*2+component} : i64',
                      f'%{pre}i{component} = arith.addi %base, %{pre}c{component} : i64']
            for arg in ('a','b','add','sub','mul','div'):
                lines += [f'%{pre}{arg}p{component} = llvm.getelementptr %{arg}[%{pre}i{component}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {scalar}']
                if arg in ('a','b'):
                    lines += [f'%{pre}{arg}{component} = llvm.load %{pre}{arg}p{component} : !llvm.ptr<1> -> {scalar}']
        def op(name, instruction, x, y):
            lines.append(f'%{pre}{name} = arith.{instruction} %{pre}{x}, %{pre}{y} : {scalar}')
        for component in range(2):
            op(f'add{component}', 'addf', f'a{component}', f'b{component}')
            op(f'sub{component}', 'subf', f'a{component}', f'b{component}')
        for name, x, y in [('rr','a0','b0'),('ii','a1','b1'),('ri','a0','b1'),('ir','a1','b0'),('br','b0','b0'),('bi','b1','b1')]:
            op(name,'mulf',x,y)
        op('mul0','subf','rr','ii'); op('mul1','addf','ri','ir')
        op('den','addf','br','bi'); op('nr','addf','rr','ii'); op('ni','subf','ir','ri')
        op('div0','divf','nr','den'); op('div1','divf','ni','den')
        for operation in ('add','sub','mul','div'):
            for component in range(2):
                lines.append(f'llvm.store %{pre}{operation}{component}, %{pre}{operation}p{component} : {scalar}, !llvm.ptr<1>')
    return '\n'.join(lines + ['gpu.return','}','}','}'])


def samples(dtype, lanes):
    npdtype = DTYPES[dtype][1]
    if dtype == 'bool':
        return np.array([False, False, True, True] * 4), np.array([False, True, False, True] * 4)
    if dtype.startswith('complex'):
        return np.array([0, 1+2j, -3+4j, -1-2j, 0.5-0.25j, 16+8j, -8j, 3] * 2, dtype=npdtype), np.array([1, 2-1j, 1+2j, -2j, .5, 2, 4j, -1] * 2, dtype=npdtype)
    floating = dtype.startswith('fp') or dtype == 'bf16'
    if dtype.startswith('fp8'):
        # Exhaustive storage inputs, including every subnormal and NaN encoding.
        values = np.arange(256, dtype=np.uint8).view(npdtype)
        return np.repeat(values, 256), np.tile(values, 256)
    if floating:
        info = np.finfo(npdtype) if dtype in ('fp16','fp32','fp64') else ml_dtypes.finfo(npdtype)
        a = [0., -0., 1., -1., 1., 1., info.max, -info.max, info.tiny,
             info.smallest_subnormal, np.inf, -np.inf, np.nan, 3., -3., 0.5]
        b = [-0., -0., info.eps / 2, info.eps / 2, info.eps, -info.eps / 2,
             info.max, info.max, 0.5, 1., -np.inf, 2., 1., 0.5, -0.5, 2.]
    else:
        info = np.iinfo(npdtype)
        a = [info.min, info.max, info.max, 0, 1, 3, info.min + 1, info.max - 1] * 2
        b = [1, 1, 3, 1, info.max, 2, info.max, 2] * 2
    return np.asarray(a, dtype=npdtype), np.asarray(b, dtype=npdtype)


def oracle(a, b, op):
    if a.dtype.kind == 'b':
        return {'add':np.logical_or,'sub':np.logical_xor,'mul':np.logical_and,'div':np.equal}[op](a,b)
    if a.dtype.kind == 'c':
        return {'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide}[op](a,b)
    if a.dtype.kind in 'iu':
        bits = a.dtype.itemsize * 8
        f = {'add': lambda x,y:x+y, 'sub': lambda x,y:x-y, 'mul': lambda x,y:x*y, 'div': lambda x,y: (abs(x)//abs(y)) * (-1 if (x<0) != (y<0) else 1)}[op]
        values = [f(int(x), int(y)) % (1 << bits) for x,y in zip(a,b,strict=True)]
        return np.asarray(values, dtype=f'uint{bits}').view(a.dtype)
    # Every fp16/bf16/fp32 input is exact in f64. Individual add/sub/mul
    # round once to storage; no fused multiply-add or relaxed math is requested.
    with np.errstate(all='ignore'):
        x, y = a.astype(np.float64), b.astype(np.float64)
        return getattr(np, {'add':'add', 'sub':'subtract', 'mul':'multiply','div':'divide'}[op])(x,y).astype(a.dtype)


def check(actual, expected):
    if actual.dtype.kind in 'iubc':
        return int(np.count_nonzero(actual != expected))
    x, y = actual.astype(np.float64), expected.astype(np.float64)
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    same &= ~((x == 0) & (y == 0) & (np.signbit(x) != np.signbit(y)))
    return int(np.count_nonzero(~same))


def run(args):
    cuda = args.backend == 'nvidia'
    toolkit = args.toolkit.resolve(strict=True) if args.toolkit else None
    if cuda and toolkit is None:
        raise ValueError('NVIDIA evidence requires an explicit --toolkit')
    tool = args.disassembler or (toolkit / 'bin/cuobjdump' if cuda else args.llvm_bin / 'llvm-objdump')
    provenance = {'disassembler': str(tool.resolve()),
                  'disassembler_sha256': hashlib.sha256(tool.read_bytes()).hexdigest()}
    if toolkit is not None:
        provenance['toolkit'] = str(toolkit)
    if cuda:
        provenance['version'] = json.loads((toolkit / 'version.json').read_text())
        provenance['ptxas_sha256'] = hashlib.sha256((toolkit / 'bin/ptxas').read_bytes()).hexdigest()
    driver = ct.CDLL('libcuda.so.1' if cuda else 'libamdhip64.so')
    P, S = ct.c_void_p, ct.c_size_t
    def bind(cu, hip, types):
        fn = getattr(driver, cu if cuda else hip)
        fn.argtypes, fn.restype = types, ct.c_int
        return fn
    def checked(status):
        if status:
            raise RuntimeError(f'driver status {status}')
    context = P()
    if cuda:
        checked(bind('cuInit', '', [ct.c_uint])(0))
        checked(bind('cuDevicePrimaryCtxRetain', '', [ct.POINTER(P), ct.c_int])(ct.byref(context), 0))
        checked(bind('cuCtxSetCurrent', '', [P])(context))
    else:
        checked(bind('', 'hipSetDevice', [ct.c_int])(0))
    alloc = bind('cuMemAlloc_v2', 'hipMalloc', [ct.POINTER(P), S])
    free = bind('cuMemFree_v2', 'hipFree', [P])
    upload = bind('cuMemcpyHtoD_v2', 'hipMemcpyHtoD', [P, P, S])
    download = bind('cuMemcpyDtoH_v2', 'hipMemcpyDtoH', [P, P, S])
    rows = []
    for dtype in args.dtypes.split(','):
        for lanes in (1, 2):
            a, b = samples(dtype, lanes)
            row = {'dtype': dtype, 'lanes': lanes, 'operations': ['or','xor','and','equal'] if dtype == 'bool' else ['add','sub','mul','div'], 'lane_mode': 'unrolled_components' if dtype.startswith('complex') else 'scalar' if lanes == 1 else 'vector', 'frontend': 'explicit MLIR primitive', 'promotion_eligible': False}
            source = emit(dtype, lanes)
            row['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            try:
                if dtype.startswith('fp8'):
                    row['arithmetic_mode'] = 'byte_storage_f32_compute_round_to_fp8'
                    row['input_pairs'] = len(a)
                package = build_native_gpu_storage(source, compiler=args.compiler, llvm_bin=args.llvm_bin,
                                                   backend=args.backend, chip='sm_120' if cuda else 'gfx1151', toolkit=toolkit)
            except subprocess.CalledProcessError as error:
                row.update(state='compile_failed', reason=str(error.stderr)[:2200])
                rows.append(row)
                continue
            row['binding_digest'] = package.binding_digest
            row['image_sha256'] = hashlib.sha256(package.image).hexdigest()
            stem = f'{dtype}_{lanes}'
            args.artifacts.mkdir(parents=True, exist_ok=True)
            (args.artifacts / (stem + '.mlir')).write_text(source)
            (args.artifacts / (stem + '.image')).write_bytes(package.image)
            image_path = args.artifacts / (stem + '.image')
            assembly = subprocess.check_output([str(tool), '--dump-sass' if cuda else '-d', str(image_path)], text=True)
            (args.artifacts / (stem + '.asm')).write_text(assembly)
            row['assembly_sha256'] = hashlib.sha256(assembly.encode()).hexdigest()
            row['instruction_mnemonics'] = sorted(set(re.findall(
                r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s' if cuda
                else r'\b((?:v_|s_|global_|flat_|buffer_)[a-z0-9_]+)\b', assembly)))
            if not row['instruction_mnemonics']:
                raise ValueError('generated image has no disassembled instruction witnesses')
            bound = package.bind()
            pointers = []
            try:
                for value in (a,b,a,a,a,a):
                    ptr = P()
                    checked(alloc(ct.byref(ptr), value.nbytes))
                    pointers.append(ptr)
                    checked(upload(ptr, P(value.ctypes.data), value.nbytes))
                start = time.perf_counter_ns()
                bound.launch(tuple(p.value for p in pointers) + (1,), grid=(len(a)//lanes,1,1), block=(1,1,1))
                row['synchronous_launch_wall_ns'] = time.perf_counter_ns() - start
                errors = {}
                for op, ptr in zip(('add','sub','mul','div'), pointers[2:], strict=True):
                    out = np.empty_like(a)
                    checked(download(P(out.ctypes.data), ptr, out.nbytes))
                    errors[op] = check(out, oracle(a,b,op))
                row.update(state='passed' if not any(errors.values()) else 'numerical_failure', mismatches=errors)
            finally:
                # An uncertain device result must not be followed by premature
                # frees. close synchronizes; failure leaves process-owned memory.
                bound.close()
                for ptr in pointers:
                    checked(free(ptr))
            rows.append(row)
    if cuda:
        checked(bind('cuDevicePrimaryCtxRelease_v2', '', [ct.c_int])(0))
    return {'backend': args.backend, 'chip': 'sm_120' if cuda else 'gfx1151',
            'toolchain': provenance,
            'compiler_sha256': hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
            'scope': 'Basic MLIR scalar/vector arithmetic; no public frontend or matrix closure; no performance promotion', 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia','rocm'), required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--dtypes', default=','.join(DTYPES))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--disassembler', type=Path)
    parser.add_argument('--toolkit', type=Path, help='Explicit GPU serialization toolkit; required for NVIDIA evidence')
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if any(row['state'] != 'passed' for row in result['rows']):
        raise SystemExit(1)
