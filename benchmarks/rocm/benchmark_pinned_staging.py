"""Paired completed-call staging comparison; the compiler image is unchanged."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_captured_movement import paged_full
from benchmarks.rocm.benchmark_native_movement import device_identity
from benchmarks.rocm.benchmark_strided_paged_kv import SOURCES, storage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(architecture, candidate_source, rounds, repeats):
    root = Path(__file__).resolve().parents[2]
    # The owning snapshot supplies unchanged Python/IR tools. Its native runtime
    # is separately compiled from the explicit candidate-source bytes below.
    if (root/'python/tessera/runtime.py').is_file() is False:
        root = Path.cwd()
    identity = device_identity(rt._load_hip_for_launch(), architecture)
    cases = []
    previous = os.environ.get('TESSERA_ROCM_MOVEMENT_PINNED_STAGING')
    try:
        os.environ['TESSERA_ROCM_MOVEMENT_PINNED_STAGING'] = '0'
        for label, shape, logical in (('small', (4, 256, 3, 8), 4),
                                      ('large', (32, 16, 8, 128), 64)):
            rng = np.random.default_rng(120_518)
            dense = rng.normal(size=shape).astype(np.float32)
            table = rng.integers(0, shape[0], logical, dtype=np.int32)
            expected = dense[table].reshape(1024, shape[2], shape[3])
            for layout in ('compact', 'padded', 'permuted', 'fortran'):
                x = storage(dense, layout)
                fn = ts.jit(target='rocm_'+architecture, native_required=True)(paged_full)
                np.testing.assert_array_equal(fn(x, table), expected)
                owner = fn.prepare_native_movement(x, table)
                owner.upload((x, table))
                descriptor, image = fn.compile_bundle.launch_descriptor, fn.compile_bundle.native_image
                cases.append((fn, owner, x, table.copy(), expected, dict(
                    profile=label, layout=layout, shape=list(shape), image_digest=image.image_digest,
                    image_cache_key=image.cache_key, abi_id=descriptor.abi_id,
                    public_window_ms={'pageable': [], 'pinned': []},
                    device_event_ms=[], correctness='passed_before_and_after_timing')))
        def forbidden(*args, **kwargs):
            raise AssertionError('warm compiler invocation')
        with ExitStack() as guards:
            for name in ('subprocess.run', 'subprocess.Popen', 'subprocess.check_output'):
                guards.enter_context(patch(name, forbidden))
            for fn, _, x, table, expected, _ in cases:
                for policy in ('0', '1'):
                    os.environ['TESSERA_ROCM_MOVEMENT_PINNED_STAGING'] = policy
                    for _ in range(4):
                        np.testing.assert_array_equal(fn(x, table), expected)
            for round_index in range(rounds):
                ordered = cases[round_index % len(cases):]+cases[:round_index % len(cases)]
                for fn, owner, x, table, expected, data in ordered:
                    arms = [('pageable', '0'), ('pinned', '1')]
                    if round_index % 2:
                        arms.reverse()
                    for label, setting in arms:
                        os.environ['TESSERA_ROCM_MOVEMENT_PINNED_STAGING'] = setting
                        start = time.perf_counter_ns()
                        for _ in range(repeats):
                            output = fn(x, table)
                        window = (time.perf_counter_ns()-start)/1e6
                        data['public_window_ms'][label].append(window)
                        np.testing.assert_array_equal(output, expected)
                    _, receipt = owner.execute(download=False)
                    data['device_event_ms'].append(receipt['kernel_elapsed_ms'])
            for fn, owner, x, table, expected, _ in cases:
                retained = fn(x, table)
                original = retained.copy()
                x[...] *= np.float32(-0.75)
                table[:] = table[::-1]
                changed = x[table].reshape(expected.shape)
                for setting in ('0', '1'):
                    os.environ['TESSERA_ROCM_MOVEMENT_PINNED_STAGING'] = setting
                    np.testing.assert_array_equal(fn(x, table), changed)
                    np.testing.assert_array_equal(retained, original)
                owner.upload((x, table))
                np.testing.assert_array_equal(owner.execute()[0], changed)
        rows = []
        for _, _, _, _, _, data in cases:
            data['public_median_ms'] = {k: median(v)/repeats for k, v in data['public_window_ms'].items()}
            data['paired_pinned_over_pageable'] = [p/c for p, c in zip(
                data['public_window_ms']['pinned'], data['public_window_ms']['pageable'], strict=True)]
            data['paired_ratio_median'] = median(data['paired_pinned_over_pageable'])
            data['device_event_median_ms'] = median(data['device_event_ms'])
            rows.append(data)
        sources = {path: sha(root/path) for path in SOURCES}
        sources['src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp'] = sha(candidate_source)
        sources['benchmarks/rocm/benchmark_pinned_staging.py'] = sha(__file__)
        for path in ('native_nvfp4_runtime.cpp', 'native_program_runtime.cpp'):
            name = 'src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/'+path
            sources[name] = sha(root/name)
        return dict(schema='tessera.rocm.pinned_staging.v1', architecture=architecture, device=identity,
                    source_hashes=sources, compiler_sha256=sha(os.environ['TESSERA_OPT']),
                    target_compiler_sha256=sha(os.environ['TESSERA_ROCM_OPT']),
                    runtime_sha256=sha(os.environ['TESSERA_ROCM_NATIVE_MOVEMENT_LIB']),
                    candidate_source=str(candidate_source), rounds=rounds, repeats=repeats,
                    comparison='same candidate runtime, compiler images and device arenas; pageable vs pinned native-owned staging',
                    public_scope='completed public JIT including transfers, excluding compilation',
                    event_scope='one resident kernel excluding host transfers; diagnostic scope separate from public calls',
                    default_staging_policy='best_effort_pinned', compiler_route_changed=False,
                    physical_schedule_promotion=False, cases=rows)
    finally:
        for _, owner, *_ in cases:
            owner.close()
        if previous is None:
            os.environ.pop('TESSERA_ROCM_MOVEMENT_PINNED_STAGING', None)
        else:
            os.environ['TESSERA_ROCM_MOVEMENT_PINNED_STAGING'] = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', required=True, choices=('gfx1151', 'gfx1201'))
    parser.add_argument('--candidate-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=128)
    args = parser.parse_args()
    if args.rounds <= 0 or args.repeats <= 0:
        raise ValueError('positive rounds/repeats required')
    packet = record(args.architecture, args.candidate_source, args.rounds, args.repeats)
    args.output.write_text(json.dumps(packet, indent=2)+'\n')
    for case in packet['cases']:
        print(args.architecture, case['profile'], case['layout'], case['public_median_ms'], case['paired_ratio_median'])


if __name__ == '__main__':
    main()
