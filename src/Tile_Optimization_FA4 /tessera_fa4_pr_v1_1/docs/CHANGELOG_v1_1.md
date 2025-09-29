# v1.1 Notes

- Added verifier implementations:
  - `schedule.warp`: role non-empty, count>0
  - `schedule.pipe`: buffering keys ⊆ {K,V,S,O}, values > 0
  - `schedule.policy`: kind ∈ {grid,persistent}; if persistent => max_cta_per_sm == 1
  - `tile.mma.tcgen05`: cta_group ∈ [1,4]
  - `numerics.softmax`: exp ∈ {poly3,native}, rescale_threshold > 0
- NVPTX inline-asm stub for `tcgen05` (symbol `__tessera_tcgen05_stub`), with TODO to guard on `sm_100`.
- E2E example showing forward attention and LSE carry ops.
