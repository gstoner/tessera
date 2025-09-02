<!-- MERGE_BEGIN -->
# Tessera Target IR → x86 (AVX‑512 / AMX) Mapping

This document outlines a reference lowering of common Tessera Target‑IR tile ops onto Intel® x86 intrinsics.

| Tessera IR Op (concept) | AVX‑512 mapping | AMX mapping | Notes |
|---|---|---|---|
| `tile.load(tile, ptr, stride)` | `vmov*` / `_mm512_load*` (vector) | `_tile_loadd(t, base, stride)` | AMX stride is in **bytes**; configure rows/cols in tileconfig. |
| `tile.store(tile, ptr, stride)` | `vmov*` / `_mm512_store*` | `_tile_stored(t, base, stride)` |  |
| `tile.zero(t)` | `_mm512_setzero_*` | `_tile_zero(t)` |  |
| `mma.bf16.acc.fp32` | `_mm512_dpbf16_ps` | `_tile_dpbf16ps(TC, A, B)` | Accumulates FP32 from BF16 operands. |
| `mma.s8.acc.s32` | `_mm512_dpbusd_epi32` / VNNI | `_tile_dpbssd(TC, A, B)` | VNNI favors unsigned×signed; choose variant per data type. |
| `reduce.add.fp32` | `_mm512_reduce_*` + `_mm512_add_ps` loop | Accumulate into TC tile then store | For AMX reduce, store to C and do horizontal reduce. |
| `barrier.tile` | `/* no‑op at scalar level */` | `/* tile ops are synchronous */` | On CPUs, explicit barriers usually unnecessary. |
| `config.amx(bf16|int8, shape…)` | `/* N/A */` | `_tile_loadconfig(&cfg)` | See runtime helper to build and load tile config. |

**Tile shapes** (suggested defaults):

- **BF16**: `TC` 16×64, `A` 16×64, `B` 64×16 (bytes/row: `cols * sizeof(bf16)`)
- **INT8**: `TC` 16×64, `A` 16×64, `B` 64×16 (bytes/row: `cols * sizeof(int8)`)

> Implementation detail: Tessera’s `mma` op should carry element types and accumulation type so the lowering can select `_mm512_dpbf16_ps` vs `_tile_dpbf16ps`, etc.

<!-- MERGE_END -->


## Epilogues

| Tessera IR (epilogue) | AVX‑512 mapping | AMX mapping | Notes |
|---|---|---|---|
| `epilogue.bias` | `_mm512_add_ps` (vector) or scalar loop | Store TC then apply vector add | Bias is typically per‑N column |
| `epilogue.bias_gelu` | bias add + tanh‑GELU | Store TC then vector/scalar GELU | Provided here as scalar GELU for portability |
