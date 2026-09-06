#!/usr/bin/env python3
"""Owning Metal proof of compiler AD pairs and explicit cross-queue fences."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
import tessera as ts  # noqa: E402
from tessera.runtime import DeviceTensor  # noqa: E402
from tessera.compiler.apple_native_arena import AppleNativeArena, AppleArenaPackage, AppleArenaQueue, build_apple_arena_package  # noqa: E402
from benchmarks.record_storage_pair import inputs_oracles  # noqa: E402


@ts.jit
def dispatch(arg0, arg1, primal, derivative, n):
    return primal, derivative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not DeviceTensor.is_metal():
        raise RuntimeError('owning Metal device unavailable')
    expected_exports = {f'{family}-{width}.json' for family in (
        'sum', 'mean', 'tanh_jvp', 'square_vjp', 'tanh_vjp', 'sum_vjp', 'mean_vjp')
        for width in (32, 64, 256)}
    if {p.name for p in a.exports.glob('*.json')} != expected_exports:
        raise ValueError('Apple pair export set is incomplete or unexpected')
    rows, queue_rows = [], []
    for path in sorted(a.exports.glob('*.json')):
        data = json.loads(path.read_text())
        digest = data.pop('digest')
        artifact = AppleNativeArena(**data)
        assert artifact.digest == digest
        package = build_apple_arena_package(artifact)
        package = AppleArenaPackage.from_json(package.to_json(), expected_digest=package.binding_digest)
        family, width = path.stem.rsplit('-', 1)
        width = int(width)
        x, d, expected, derivative = inputs_oracles(family, width)
        arrays = [x, d, np.zeros(expected.shape, np.float32), np.zeros(derivative.shape, np.float32)]
        tensors = [DeviceTensor.from_numpy(v) for v in arrays]
        if any(t is None for t in tensors):
            raise RuntimeError('Metal allocation failed')
        try:
            dispatch.bind_native_storage_pair(package)
            assert dispatch(*tensors, width) == tuple(tensors[-2:])
            np.testing.assert_allclose(tensors[-2].numpy(), expected, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(tensors[-1].numpy(), derivative, atol=2e-6, rtol=2e-6)
            rows.append(dict(family=family, width=width, package_digest=package.binding_digest,
                             compiler_digest=artifact.compiler_digest, oracle='atol=rtol=2e-6'))
            if family == 'square_vjp':
                producer, consumer = AppleArenaQueue(package), AppleArenaQueue(package)
                try:
                    with package.bind(queue=consumer) as bound:
                        for value in [0, 63, 0, 62]:
                            fence = producer.fill_bytes(tensors[0], value)
                            try:
                                consumer.wait_for(fence)
                                bound.launch((*tensors, width), grid=(1, 1, 1), block=(width, 1, 1))
                                filled = np.frombuffer(bytes([value])*4, np.float32)[0]
                                np.testing.assert_allclose(tensors[-2].numpy(), filled*filled, atol=2e-6, rtol=2e-6)
                                np.testing.assert_allclose(tensors[-1].numpy(), 2*d*filled, atol=2e-6, rtol=2e-6)
                                queue_rows.append(dict(width=width, byte=value, protocol='producer blit -> fresh shared event -> consumer wait -> VJP'))
                            finally:
                                fence.close()
                            try:
                                consumer.wait_for(fence)
                            except RuntimeError:
                                pass
                            else:
                                raise AssertionError('closed fence accepted')
                finally:
                    producer.close()
                    consumer.close()
        finally:
            dispatch.close_native_storage()
            for tensor in tensors:
                tensor.free()
    a.output.write_text(json.dumps(dict(backend='apple', host=platform.platform(), rows=rows,
        queue_rows=queue_rows, selector_promotion=False, performance_claim=False,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'benchmarks/record_apple_storage_pair.py', 'python/tessera/compiler/apple_arena_bridge.mm',
            'python/tessera/compiler/apple_native_arena.py', 'python/tessera/compiler/native_storage_pair.py']}), indent=2)+'\n')
    print(f'{len(rows)} paired cases, {len(queue_rows)} cross-queue generations passed')


if __name__ == '__main__':
    main()
