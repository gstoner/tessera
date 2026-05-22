
# Power/Retention in Tessera (v0.2)

This revision aligns the Tessera port with **Vidrial**'s kernel organization model and adds a **Retention** variant:

- **Vidrial patterns**: separate static config from dynamic data, rigid naming (slab/tile/fragment), CUTE layouts,
  SMEM pipeline with staged double/triple buffering, optional LDSM loads, vectorized gmem copies, MMA fragments.
- **Power Attention**: symmetric, linear-cost with tunable state `M`, optional window, causal masking.
- **Power Retention**: chunked algorithm, optional gating `log_G`, training/inference paths; inference keeps a fixed-size
  **state** and **sum_of_keys** analogous to KV cache but O(1) memory in sequence length.

## Dialect

```
%o = power.attn %q, %k, %v {state=256, window=0, causal} : (...) -> (...)
%o2, %state2, %sum2 = power.retention %q, %k, %v, %log_g
    {deg=2, chunk=128, switch_over=1024, causal} : (...) -> (..., ..., ...)
```

## Lowering

- **Tile IR**: builds two phases:
  1) state update sweep across S (pipelines data movement Q/K/V → SMEM → registers),
  2) projection from state to output per token (optionally via MMA fragments).
- **Target IR**: selects a tuned `Cfg` and launches a fused kernel (`power` or `retention` flavor).

## Inference semantics (Retention)

- Maintains `(state, sum_of_keys)` of shape `[B, H, D, Dh]` and `[B, H, D]` respectively.
- `switch_over_seq_len` dictates when to convert accumulated K/V into `(state, sum_of_keys)`.
- Next-token calls query from current `(state, sum_of_keys)`; update happens only when needed.

## Dtypes & tensor cores

- BF16/F16 compute with FP32 accumulate; optional FP8 storage.
- Hopper/Blackwell: prefer WGMMA + `cp.async`/TMA; MI300: MFMA with ds_read/write vectorization.

## Tuning

- Token tile 128/256, 2–3 pipeline stages, cooperative_groups for sync, vectorized 128-bit gmem.
- Vidrial-style `Cfg` static specialization enables parallel JIT-style config sweeps (offline via CMake options).

