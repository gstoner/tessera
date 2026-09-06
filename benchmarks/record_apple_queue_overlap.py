#!/usr/bin/env python3
"""Matched serial/concurrent Metal command intervals, with independent outputs.

GPU timestamp interval intersection is queue residency evidence, not proof that
instructions issue simultaneously. No automatic selection follows this recorder.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from tessera.compiler.apple_native_arena import AppleNativeArena, AppleArenaQueue, build_apple_arena_package
from tessera.runtime import DeviceTensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=9)
    a = p.parse_args()
    data = json.loads(a.artifact.read_text())
    digest = data.pop('digest')
    artifact = AppleNativeArena(**data)
    if artifact.digest != digest or not DeviceTensor.is_metal():
        raise RuntimeError('native artifact or owning Metal device unavailable')
    package = build_apple_arena_package(artifact)
    queues = [AppleArenaQueue(package) for _ in range(2)]
    bindings = [package.bind(queue=q) for q in queues]
    outputs = [DeviceTensor.empty((32, 256), np.float32) for _ in range(2)]
    rows = []
    try:
        for rounds in [128, 512, 2048]:
            # Same shader, buffers, geometry and work for both schedules.
            def launch(i, gate=None):
                if gate is not None:
                    gate.wait()
                bindings[i].launch((outputs[i], 256, rounds), grid=(32, 1, 1), block=(256, 1, 1))
                return bindings[i].last_device_interval()
            for i in range(2):
                launch(i)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for sample in range(a.samples):
                    for mode in (['serial', 'concurrent'] if sample % 2 == 0 else ['concurrent', 'serial']):
                        start = time.perf_counter()
                        if mode == 'serial':
                            intervals = [launch(0), launch(1)]
                        else:
                            gate = threading.Barrier(2)
                            futures = [pool.submit(launch, i, gate) for i in range(2)]
                            intervals = [f.result() for f in futures]
                        wall = time.perf_counter() - start
                        expected = np.tile(rounds*((np.arange(256)+1)%256) + rounds*(rounds-1)/2, (32, 1))
                        for output in outputs:
                            np.testing.assert_array_equal(output.numpy(), expected)
                        intersection = max(0, min(e for _, e in intervals)-max(s for s, _ in intervals))
                        span = max(e for _, e in intervals)-min(s for s, _ in intervals)
                        rows.append(dict(rounds=rounds, sample=sample, mode=mode, gpu_intervals_s=intervals,
                                         gpu_span_s=span, interval_intersection_s=intersection, wall_s=wall))
    finally:
        for binding in bindings:
            binding.close()
        for queue in queues:
            queue.close()
        for output in outputs:
            output.free()
    root = Path(__file__).resolve().parents[1]
    a.output.write_text(json.dumps(dict(rows=rows, package_digest=package.binding_digest,
        compiler_digest=artifact.compiler_digest, device=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in [
            'benchmarks/record_apple_queue_overlap.py', 'python/tessera/compiler/apple_arena_bridge.mm',
            'python/tessera/compiler/apple_native_arena.py']},
        interpretation='command timestamp overlap only; no instruction concurrency or selector claim'), indent=2)+'\n')
    print(f'{len(rows)} matched measurements passed correctness')


if __name__ == '__main__':
    main()
