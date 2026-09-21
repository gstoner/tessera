#!/usr/bin/env python3
"""Project dtype contracts without upgrading declarations into execution proof."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tessera.dtype import canonical_dtypes, planned_gated_dtypes  # noqa: E402
from tessera.compiler.nvidia_dtype_contract import SM120_DTYPE_CONTRACTS  # noqa: E402
from tessera.compiler.rocm_isa_contract import AMD_DTYPE_CONTRACTS  # noqa: E402
from tessera.compiler.rocm_exact_device_proofs import GFX1201_PUBLIC_PROOFS  # noqa: E402
from tessera.compiler.x86_dtype_contract import X86_DTYPE_CONTRACTS  # noqa: E402


def record():
    """Include vocabulary with no backend contract, rather than silently omit it."""
    canonical = canonical_dtypes()
    sources = {
        'nvidia_sm120': ('nvidia_dtype_contract.py', SM120_DTYPE_CONTRACTS),
        'x86_zen5_avx512': ('x86_dtype_contract.py', X86_DTYPE_CONTRACTS),
        **{f'rocm_gfx{arch}': ('rocm_isa_contract.py', tuple(rows.values()))
           for arch, rows in AMD_DTYPE_CONTRACTS.items()},
        'apple_gpu': (None, ()),
    }
    rows = []
    gfx1201_matmul = next(
        proof for proof in GFX1201_PUBLIC_PROOFS
        if proof.target == 'rocm_gfx1201' and proof.op_name == 'tessera.matmul'
    )
    for dtype in sorted(canonical | planned_gated_dtypes()):
        targets = {}
        for target, (source, contracts) in sources.items():
            matches = [asdict(c) for c in contracts if c.storage == dtype]
            targets[target] = {
                'source': 'python/tessera/compiler/' + source if source else None,
                'contracts': matches,
                'coverage': 'declared_contract' if matches else 'no_layered_contract',
                'execution_proof': 'requires_operation_artifact_and_exact_device_packet',
            }
            if target == 'rocm_gfx1201' and dtype in gfx1201_matmul.dtypes:
                targets[target]['exact_device_operations'] = {
                    'tessera.matmul': {
                        'compiler_path': gfx1201_matmul.compiler_path,
                        'executor_id': gfx1201_matmul.executor_id,
                        'fixture': gfx1201_matmul.numerical_fixture,
                        'proof_build': gfx1201_matmul.proof_build,
                    }
                }
        rows.append({'dtype': dtype, 'vocabulary': 'canonical' if dtype in canonical else 'planned_gated',
                     'targets': targets})
    return {'schema': 2, 'scope': 'Scalar/vector and matrix contract projection. Exact-device operation fields are '
            'joined only from the machine-readable proof registry; a declaration alone is not code-generation or device proof. '
            'Apple operation-specific capabilities are not a layered dtype contract. '
            'Zen 5 vector GEMM and VNNI do not establish AMX or systolic-array support. '
            'TF32 is a fp32 math mode, not a storage dtype.', 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(record(), indent=2) + '\n')
