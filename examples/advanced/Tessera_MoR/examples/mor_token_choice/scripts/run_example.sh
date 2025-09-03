#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

echo "[MoR] Example run (adjust commands to your tessera toolchain)"

# Example flow:
# tessera-compile "$ROOT/mor_token_choice.tsr" -o "$ROOT/mlir/out.mlir"
# tessera-opt "$ROOT/mlir/out.mlir" \
#   -tessera-mor-route-assign -tessera-mor-kv-plan -tessera-mor-depth-batching \
#   -canonicalize -cse -o "$ROOT/mlir/out.lowered.mlir"
# tessera-run "$ROOT/mlir/out.lowered.mlir" --print-summary
echo "[MoR] Done."
