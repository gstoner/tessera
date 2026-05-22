# Nsight Compute Recipes for Tessera Attention Kernels

## Quick start
```bash
# Forward tiled
ncu -o out/ncu_fwd --set full --target-processes all \  --nvtx base --nvtx-include "tessera.flashattn_fwd_tiled" \  python -m pytest -q tests/functional/test_flashattn_tiled_vs_torch.py::test_tiled_forward_matches_torch

# Backward tiled
ncu -o out/ncu_bwd --set full --target-processes all \  --nvtx base --nvtx-include "tessera.flashattn_bwd_tiled" \  python -m pytest -q tests/functional/test_flashattn_tiled_vs_torch.py::test_tiled_backward_matches_torch
```

## Sections to inspect
- **Memory Workload Analysis**: global read/write throughput, L2 hit %, dram transactions.
- **Launch Statistics**: eligible warps per cycle, occupancy, block limiters (smem/regs).
- **Scheduler Stats**: warp issue efficiency, stall reasons (long scoreboard, memory throttling).
- **Source Counters**: correlate with NVTX ranges `LDS:KV`, `QK^T:max`, `softmax+PV`, `bwd:commit dK/dV`.
