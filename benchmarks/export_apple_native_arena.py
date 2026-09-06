#!/usr/bin/env python3
"""Export compiler-owned MSL and LLVM sizing for an owning-Mac device probe."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tessera.compiler.apple_native_arena import materialize_apple_arena


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--llvm-bin', type=Path, default=Path('/usr/lib/llvm-23/bin'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--typed', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root/'tests/tessera-ir/phase3/tile_dynamic_gpu_nested_device.mlir').read_text()
    if args.typed:
        from tessera.compiler.native_storage_contract import attach_tensor_contract
        from tessera.compiler.native_gpu_tensor import TensorSpec, IndexSpec
        source = attach_tensor_contract(source, (TensorSpec('output', 'fp32', (32, 'n'), True),
            IndexSpec('n', 1, 256), IndexSpec('rounds', 0, 33)), grid=(32, 1, 1), block=('n', 1, 1))
    artifact = materialize_apple_arena(source, compiler=args.compiler, llvm_bin=args.llvm_bin)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'kernel.metal').write_text(artifact.msl)
    (args.output/'sizer.ll').write_text(artifact.host_llvm_ir)
    (args.output/'artifact.json').write_text(json.dumps(dict(**asdict(artifact), digest=artifact.digest),indent=2)+'\n')


if __name__ == '__main__':
    main()
