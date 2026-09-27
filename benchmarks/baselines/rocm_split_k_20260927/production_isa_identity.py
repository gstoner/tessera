"""Does the post-rule production package compile to the same instruction stream
the sweep timed for that slice count?"""
import json
import sys
from pathlib import Path

sys.path[:0] = ['.', 'python']
from benchmarks.rocm.record_split_k_router_gate import _graph_module
from benchmarks.rocm.record_split_k_sweep import _isa_sha256
from tessera.compiler import rocm_native, scheduled_matmul
from tessera.compiler.llvm_tools import llvm_bin_dir

sweep = json.loads(Path('benchmarks/baselines/rocm_split_k_20260927/sweep_20ms/sweep.json').read_text())
timed = {(g['dtype'], tuple(g['shape']), v['split_k']): v.get('isa_sha256')
         for g in sweep['groups'] for v in g['variants']}
same = diff = 0
for g in sweep['groups']:
    m, n, k = g['shape']
    art = scheduled_matmul.lower_scheduled_matmul(_graph_module(m, n, k, g['dtype']), target='rocm_gfx1201')
    pkg = rocm_native.package_scheduled_matmul(art, pipeline_name='tessera-lower-to-rocm')
    isa = _isa_sha256(pkg.image.payload, llvm_bin_dir())
    want = timed.get((g['dtype'], tuple(g['shape']), int(art.split_k)))
    ok = isa == want
    same += ok
    diff += not ok
    print(g['dtype'], 'x'.join(map(str, g['shape'])), 'S=', art.split_k,
          'k_unroll=', pkg.descriptor.provenance['k_unroll'], 'same_isa' if ok else 'DIFFERENT')
print('same', same, 'different', diff)
