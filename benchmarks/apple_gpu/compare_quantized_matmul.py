"""Correctness-gated MPP comparisons from identical quantized operand values.

Each low-precision route has its own matched FP16 baseline. Packing and host
submission costs remain separate from device-counter intervals. No promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import ml_dtypes
import numpy as np



def pack_values(values, fmt):
    dtype = {'fp8_e4m3': ml_dtypes.float8_e4m3fn,
             'fp8_e5m2': ml_dtypes.float8_e5m2,
             'fp4_e2m1': ml_dtypes.float4_e2m1fn}[fmt]
    quantized = values.astype(dtype)
    codes = quantized.view(np.uint8)
    if fmt == 'fp4_e2m1':
        codes = codes[:, ::2] | (codes[:, 1::2] << 4)
    return np.ascontiguousarray(codes), quantized.astype(np.float16)


def verify(result, a, b):
    # Bound accumulation separately from input quantization. All reference
    # operands are exactly representable in fp16 and float64.
    a, b = a.astype(np.float64), b.astype(np.float64)
    reference = a @ b
    k = a.shape[1]
    u = np.finfo(np.float32).eps / 2
    bound = (k * u / (1 - k * u)) * (np.abs(a) @ np.abs(b))
    error = np.abs(result.astype(np.float64) - reference)
    if not np.all(np.isfinite(result)) or not np.all(error <= bound + 1e-7):
        raise ValueError('matmul correctness gate failed; timing refused')
    return {'max_abs_error': float(error.max()), 'max_abs_bound': float(bound.max()),
            'reference': 'float64_from_quantized_values', 'correctness_passed': True}


def main():
    from benchmarks.apple_gpu.benchmark_lowp_matmul2d import Timer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shapes', default='256,1024,2048')
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from tessera import runtime as rt
    if platform.system() != 'Darwin' or not rt.apple_gpu_mtl4_matmul2d_lowp_available():
        raise RuntimeError('owning Metal4.1 runtime required')
    if args.reps < 3:
        raise ValueError('at least three repetitions required')
    timer = Timer(rt)
    rows = []
    rng = np.random.default_rng(15)
    for n in map(int, args.shapes.split(',')):
        if n <= 0 or n > 4096 or n % 256:
            raise ValueError('shapes must be multiples of 256 through 4096')
        a, b = (rng.uniform(-2, 2, (n, n)).astype(np.float32) for _ in range(2))
        for fmt in ('fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1'):
            start = time.perf_counter_ns()
            ac, ah = pack_values(a, fmt)
            bc, bh = pack_values(b, fmt)
            pack_ms = (time.perf_counter_ns() - start) / 1e6
            low = rt.apple_gpu_mtl4_matmul2d_lowp(ac, bc, np, fmt=fmt, M=n, N=n, K=n)
            high, ran = rt.apple_gpu_mtl4_matmul2d_f16(ah, bh, np)
            if not ran:
                raise RuntimeError('matched fp16 route declined')
            proofs = [verify(low, ah, bh), verify(high, ah, bh)]
            timings = []
            for route in ('lowp', 'fp16'):
                def run():
                    if route == 'lowp':
                        rt.apple_gpu_mtl4_matmul2d_lowp(ac, bc, np, fmt=fmt, M=n, N=n, K=n)
                    else:
                        _, ok = rt.apple_gpu_mtl4_matmul2d_f16(ah, bh, np)
                        if not ok:
                            raise RuntimeError('fp16 route declined during timing')
                    ns, source = timer.device_ns()
                    if ns is None or source != 'metal4_timestamp_heap':
                        raise RuntimeError('device-counter evidence missing')
                    return ns
                timings.append(timer.run(run, reps=args.reps))
            rows.append({'shape': [n, n, n], 'dtype': fmt, 'proofs': proofs,
                         'packed_bytes': ac.nbytes + bc.nbytes,
                         'fp16_bytes': ah.nbytes + bh.nbytes,
                         'pack_and_reference_conversion_ms': pack_ms,
                         'input_sha256': hashlib.sha256(ac.tobytes() + bc.tobytes()).hexdigest(),
                         'lowp': timings[0], 'matched_fp16': timings[1],
                         'device_speed_ratio': timings[1]['device_ms_median'] / timings[0]['device_ms_median']})
            print(n, fmt, rows[-1]['device_speed_ratio'], flush=True)
    lib = Path(os.environ['TESSERA_APPLE_GPU_RUNTIME_LIB'])
    report = {'device': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
              'os': platform.mac_ver()[0], 'compiler': subprocess.check_output(['xcrun', 'metal', '--version'], text=True),
              'runtime_sha256': hashlib.sha256(lib.read_bytes()).hexdigest(), 'seed': 15,
              'promotion_eligible': False, 'rows': rows}
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
