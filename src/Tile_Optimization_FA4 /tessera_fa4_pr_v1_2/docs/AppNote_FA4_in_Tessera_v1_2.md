# Application Note: FA‑4 Style Warp Specialization & TMEM in Tessera (v1.2)

**Audience:** Tessera compiler/runtime engineers and kernel authors targeting NVIDIA Blackwell (SM_100).  
**Goal:** Encode FlashAttention‑4 style pipeline controls as first‑class IR so we can tune, verify, and lower consistently across backends.

## 1) What you get
- Warp‑specialized pipeline with explicit roles (`load`, `mma`, `softmax`, `correction`, `epilogue`) and a persistent scheduler policy.
- Numerics policy for fast softmax (`exp=poly3`, `rescale_threshold`).
- Blackwell TMEM memory space + `tcgen05.mma` op with `cta_group` control.
- LSE carry ops: `tessera.attn.lse.save/load` to bridge fwd/bwd.
- Lowering that emits async tokens and (for SM_100) a PTX inline‑asm hook for `tcgen05`.

## 2) How to write the IR
```mlir
tessera.schedule @fa4 {
  %w_load  = tessera.schedule.warp "load", 1
  %w_mma   = tessera.schedule.warp "mma", 1
  %w_smx   = tessera.schedule.warp "softmax", 8
  %w_corr  = tessera.schedule.warp "correction", 4
  %w_epi   = tessera.schedule.warp "epilogue", 2
  tessera.schedule.pipe %w_load, %w_mma, %w_smx, %w_corr, %w_epi { buffering = {K=3,V=3,S=2,O=2} }
  tessera.schedule.policy "persistent", 1, "static"
}
tessera.numerics.softmax "poly3", 2.0e-3
%acc = "tessera.tile.mma.tcgen05"(%q, %k -> %buf, 2)
%lse = "tessera.attn.lse.save"(%acc)
```

**Rules enforced by verifiers**
- Roles must be unique; counts must be > 0.
- Buffering keys ⊆ {K,V,S,O}, values > 0.
- `policy "persistent"` ⇒ `max_cta_per_sm == 1`.
- `tcgen05.mma`: `cta_group ∈ [1,4]`.
- `softmax`: `exp ∈ {poly3,native}`, `rescale_threshold > 0`.

## 3) How it lowers
- `schedule.*` → `async.execute` regions per role + `async.await` edges per pipe; persistent policy attaches launch metadata (`ctas_per_sm=1`) and creates a tile‑queue token.
- `numerics.softmax` → chooses polynomial exp + thresholded correction in softmax regions.
- `tile.mma.tcgen05` → for `sm_100` target: emits an inline‑asm PTX stub (replace with true PTX when ready). Other targets: gracefully bypass (or fall back to WMMA).
- `attn.lse.*` → materializes an LSE tensor for reuse in backward.

## 4) Tuning knobs (autotuner schema v2)
- `warp_counts.softmax` ∈ {4,8,12}; `warp_counts.correction` ∈ {2,4,6}
- `buffer_depths.K/V/S/O` ∈ {…}
- `scheduler` ∈ {grid, persistent}
- `rescale_threshold` ∈ {1e‑3, 2e‑3, 4e‑3}

## 5) Practical guidance
- Start with `softmax=8, correction=4, epilogue=2` for long‑seq; shrink for short‑seq.
- Tune `rescale_threshold` jointly with the model’s logit scaling (RoPE/ALiBi can shift optimal values).
- For Blackwell, prefer `cta_group=2` on common head sizes; test 1/2/4 for register pressure vs overlap.

## 6) Next work
- Replace PTX stub with real `tcgen05.mma` + operands/constraints guarded on `sm_100`.
- Add shape‑checked verifiers for `lse.save` (rows) and `lse.load`.
- Persistent tile‑queue as a first‑class type with producer/consumer ops for better analysis.
