#!/usr/bin/env python3
"""Public resident SSD VJP versus an independent float64 finite-difference oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]


def reference(values):
    x, decay, b, c, initial = [np.asarray(v, dtype=np.float64) for v in values]
    state = initial.copy()
    out = []
    for t in range(len(x)):
        state = decay[t, :, None, None] * state + b[t, :, :, None] * x[t, :, None, :]
        out.append(np.einsum('hn,hnp->hp', c[t], state))
    return np.array(out)


def finite_vjp(values, seed, epsilon=1e-5):
    values = [np.array(v, dtype=np.float64, copy=True) for v in values]
    result = []
    for value in values:
        grad = np.empty_like(value)
        for index in np.ndindex(value.shape):
            old = value[index]
            value[index] = old + epsilon
            plus = np.sum(reference(values) * seed)
            value[index] = old - epsilon
            minus = np.sum(reference(values) * seed)
            value[index] = old
            grad[index] = (plus - minus) / (2 * epsilon)
        result.append(grad)
    return result


def main():
    from benchmarks.record_device_ring_protocol import Device
    from benchmarks.record_pool_resident_ssd import Memory
    from tessera.compiler.scheduled_ssd import lower_scheduled_ssd
    from tessera.compiler.resident_ssd import ResidentSSDProgram
    from tessera.control import vjp
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', required=True, choices=['nvidia', 'rocm'])
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    device = Device(args.backend)
    rows = []
    for T, H, N, P, chunk in ((1, 1, 1, 1, 1), (3, 2, 2, 2, 2), (5, 1, 3, 2, 2)):
        rng = np.random.default_rng(746 + T)
        values = [rng.uniform(-.4, .4, shape).astype(np.float32)
                  for shape in [(T, H, P), (T, H), (T, H, N), (T, H, N), (H, N, P)]]
        memory = Memory(device)
        try:
            inputs = [memory.put(v) for v in values]
            logical = lower_scheduled_ssd(T, H, N, P, chunk, compiler=args.compiler)
            with ResidentSSDProgram(logical, compiler=args.compiler, llvm_bin=Path('/usr/lib/llvm-23/bin'),
                                    backend=args.backend, chip='sm_120' if device.cuda else 'gfx1151') as program:
                for seed_name, seed in [('ones', np.ones_like(values[0])),
                                        ('signed', rng.uniform(-1, 1, values[0].shape).astype(np.float32))]:
                    result, pullback = vjp(program, *inputs)
                    with pullback:
                        np.testing.assert_allclose(memory.get(result.frame.value), reference(values), rtol=2e-5, atol=2e-6)
                        gradients = pullback(memory.put(seed))
                        expected = finite_vjp(values, seed)
                        errors = []
                        for got, want in zip(gradients, expected, strict=True):
                            actual = memory.get(got)
                            np.testing.assert_allclose(actual, want, rtol=2e-4, atol=2e-6)
                            errors.append(float(np.max(np.abs(actual - want))))
                    rows.append(dict(shape=[T, H, N, P], chunk=chunk, seed=seed_name,
                                     max_abs_errors=errors, forward=program.forward.package.binding_digest,
                                     reverse=program.reverse.package.binding_digest))
        finally:
            memory.close()
    args.output.write_text(json.dumps(dict(schema=1, backend=args.backend, host_kernel=platform.release(),
        api='tessera.control.vjp', execution_kind='native_gpu', oracle='independent_float64_central_difference',
        differentiated_inputs=['x', 'decay', 'b', 'c', 'initial'], rows=rows, promotion_eligible=False,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main()
