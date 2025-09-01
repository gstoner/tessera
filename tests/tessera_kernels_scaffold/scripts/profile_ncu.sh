#!/usr/bin/env bash
set -euo pipefail
mkdir -p out
ncu -o out/ncu_fwd --set full --target-processes all --nvtx base --nvtx-include "tessera.flashattn_fwd_tiled" \  python -m pytest -q tests/functional/test_flashattn_tiled_vs_torch.py::test_tiled_forward_matches_torch
ncu -o out/ncu_bwd --set full --target-processes all --nvtx base --nvtx-include "tessera.flashattn_bwd_tiled" \  python -m pytest -q tests/functional/test_flashattn_tiled_vs_torch.py::test_tiled_backward_matches_torch
