<<<MERGE_START: Tessera_TPU_FlashAttention_Lowering>>>
# FlashAttention Lowering for TPU (StableHLO)

We lower `tessera.flash_attn(Q,K,V, [mask], scale, dropout_p, causal)` to a TPU-friendly form.

## v1 (this starter)
- Emit `stablehlo.custom_call` named `tessera.flash_attention` with attrs:
  - `tessera.scale: f32`
  - `tessera.dropout_p: f32`
  - `tessera.causal: i1`
  - `tessera.has_mask: i1`
- XLA backends can fuse this into an on-chip attention kernel (softmax, masking, dropout included).

## v2 (next step)
- Expand into StableHLO primitives:
  1. `scores = dot_general(Q, K^T) * scale`
  2. `scores += mask` (or add -INF where masked)
  3. `P = softmax(scores)` via `reduce(max)` + `exp` and `reduce(sum)`
  4. (dropout) `mask = rng_uniform(0,1) < (1-p)`, `P *= mask / (1-p)`
  5. `O = dot_general(P, V)`
- The XLA HLO pipeline will fuse steps to keep values on chip.

<<<MERGE_END: Tessera_TPU_FlashAttention_Lowering>>>
