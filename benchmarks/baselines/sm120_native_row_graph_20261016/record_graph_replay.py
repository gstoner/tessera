"""Matched native row owner direct/graph replay; run without other GPU jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.prepared_attention_vjp import clear_prepared, prepared
from tests.device.nvidia.test_native_row_graph_replay import graph_info
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert rt._nvidia_device_name() == 'sm_120'
    gpu = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,compute_cap,driver_version', '--format=csv,noheader'], text=True).strip()
    root = Path.cwd()
    paths = [Path(__file__), Path(os.environ['TESSERA_NVIDIA_PTX_LAUNCH_LIB']),
             Path(os.environ['TESSERA_OPT']), Path(os.environ['TESSERA_NVIDIA_OPT'])]
    paths += [root / name for name in (
        'python/tessera/compiler/prepared_attention_vjp.py',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp',
        'src/transforms/lib/AttentionFamilyPasses.cpp',
        'src/transforms/lib/NativeAttentionProjection.h',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp')]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity = hashes()
    rows = []
    os.environ['TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION'] = 'prepass'
    for storage in ('float16', 'bfloat16'):
        dtype = np.float16 if storage == 'float16' else ml_dtypes.bfloat16
        for dims in ((1, 2, 1, 3, 5, 4, 3), (1, 4, 2, 1, 129, 32, 24),
                     (1, 4, 2, 7, 129, 32, 24), (1, 4, 2, 32, 257, 32, 24)):
            b, hq, hkv, sq, sk, d, dv = dims
            rng = np.random.default_rng(20261016)
            values = {name: rng.normal(0, .3, shape).astype(dtype) for name, shape in {
                'q': (b, hq, sq, d), 'k': (b, hkv, sk, d), 'v': (b, hkv, sk, dv)}.items()}
            values['bias'] = rng.normal(0, .1, (1, hq, 1, sk)).astype(np.float32)
            do = rng.normal(0, .2, (b, hq, sq, dv)).astype(dtype)
            seed = rng.normal(0, .2, (b, hq, sq)).astype(np.float32)
            _, oracle = reference(values['q'], values['k'], values['v'], do, values['bias'], seed)
            expected = tuple(oracle[i] for i in (3, 2, 1, 0))
            fn = ts.jit(target='nvidia_sm120', autodiff='reverse', wrt=('bias', 'v', 'k', 'q'))(function(True, True))
            try:
                os.environ['TESSERA_NVIDIA_ATTENTION_ROW_GRAPH'] = '0'
                initial = fn.native_backward(**values, out_cotangents=(do, seed))
                artifact = fn.native_backward_runtime_artifact()
                owner = prepared(artifact.metadata)
                ordered = tuple(values[name] for name in fn.arg_names) + (do, seed)
                members = owner.program.members
                proof = dict(program_digest=artifact.metadata['program_digest'],
                    members=[dict(image_digest=m.image.image_digest, entry=m.descriptor.entry_symbol,
                                  schedule_digest=m.descriptor.provenance['schedule_digest'],
                                  tile_ir_digest=m.descriptor.provenance['tile_ir_digest']) for m in members])
                def check(outputs):
                    for a, e in zip(outputs, expected, strict=True):
                        compare(a, e)
                    for a, e in zip(outputs, initial, strict=True):
                        np.testing.assert_array_equal(a, e)
                check(initial)
                for policy in ('0', '1'):
                    os.environ['TESSERA_NVIDIA_ATTENTION_ROW_GRAPH'] = policy
                    for _ in range(3):
                        check(fn.native_backward(**values, out_cotangents=(do, seed)))
                samples = {domain: {policy: [] for policy in ('0', '1')}
                           for domain in ('public', 'prepared')}
                for batch in range(8):
                    order = ('0', '1') if batch % 2 == 0 else ('1', '0')
                    for domain in samples:
                        for policy in order:
                            os.environ['TESSERA_NVIDIA_ATTENTION_ROW_GRAPH'] = policy
                            for rep in range(10):
                                start = time.perf_counter_ns()
                                if domain == 'public':
                                    outputs = fn.native_backward(**values, out_cotangents=(do, seed))
                                else:
                                    outputs = owner.invoke(artifact.metadata, ordered)
                                wall = (time.perf_counter_ns() - start) / 1e6
                                stage = owner.last_device_ms
                                check(outputs)
                                samples[domain][policy].append(dict(batch=batch, rep=rep, order=order,
                                    completed_call_ms=wall, forward_ms=stage[0], row_ms=stage[1],
                                    backward_ms=stage[2], device_window_ms=stage[3]))
                info = graph_info(owner)
                assert info[:2] == (1, 0) and info[4] == 3
                assert info[6:] == (1, 163, 0, 0), info
                medians = {domain: {policy: {key: median(s[key] for s in arm) for key in (
                    'completed_call_ms', 'device_window_ms', 'forward_ms', 'row_ms', 'backward_ms')}
                    for policy, arm in arms.items()} for domain, arms in samples.items()}
                row = dict(storage=storage, dims=dims, proof=proof, graph_info=info, samples=samples,
                           medians=medians, ratios_direct_over_graph={domain: {
                               key: m['0'][key] / m['1'][key] for key in ('completed_call_ms', 'device_window_ms')}
                               for domain, m in medians.items()})
                rows.append(row)
                print(storage, dims, row['ratios_direct_over_graph'], flush=True)
                args.output.with_suffix('.partial.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
            finally:
                fn.close_native_storage()
                clear_prepared()
    assert hashes() == identity
    args.output.write_text(json.dumps(dict(schema='tessera.sm120.row_graph_ab.v1', device=gpu,
        identity=identity, benchmark_complete=True, rows=rows,
        methodology=dict(batches=8, repetitions_per_arm=80, warmup_per_arm=3,
                         alternating='AB/BA', same_owner_and_images=True,
                         correctness='independent FP64 and bitwise direct before and outside every sample',
                         domains='public native_backward and prepared invoke wall; separate native member events and forward-to-backward device window; host graph copies lie outside device interval',
                         graph_policy='explicit opt-in; no automatic promotion')), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
