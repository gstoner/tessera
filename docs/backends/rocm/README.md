---
classification: Backend architecture reference
authority: ROCm reader entry point
last_updated: 2026-09-21
---

# ROCm Backend

This page is the reader-facing entry point for Tessera's AMD ROCm/HIP target.
It separates the compiler and runtime architecture from the generated evidence
and from the historical bring-up record.

## Current evidence

| Exact target | Product families | Tessera support boundary |
|---|---|---|
| `gfx1201` | Radeon RX 9070 series; Radeon AI PRO R9700/R9700S/R9600D | Exact-device RX 9070 XT proof exists for every registered content-addressed family plugin and bounded native runtime ABIs. Public exact-target capability/execution rows project the proved `matmul`, `flash_attn`, and `softmax` subset. D=128 linear-attention load batching has numerical, HSACO-resource, and paired timing evidence. Exact MXFP4 W4A8 scalar and FP8-WMMA packages have bit-exact K32-scale proof and recorded ISA/resources; the named packed generic carrier now materializes that proved ABI, while the public dtype and folded policy remain unpromoted. |
| `gfx1200` | Radeon RX 9060 and RX 9050 series | ISA, dtype, feature, and compile-target modeling only. No promoted executable family, measured topology default, numerical device fixture, or performance claim. |

`gfx1200` and `gfx1201` share RDNA 4 ISA features, but they are distinct exact
targets. Compiler acceptance or device evidence for one never promotes the
other. The aliases `gfx12`, `rdna4`, and `rx9000` are intentionally rejected
because they do not select an exact target.

- [Runtime execution matrix](../../audit/generated/runtime_execution_matrix.md)
  is current public operation/executor registry truth. Only the bounded
  scheduled-package subset with a public operation/path/fixture join is
  projected; the broader internal family proof does not widen it.
- [ROCm target map](../../audit/generated/rocm_target_map.md) is the generated
  per-op exact-target view.
- [WMMA fragment layout](wmma-fragment-layout.md) is the normative lane /
  accumulator / operand / int4-nibble contract the generators are written
  against, with the device evidence for each claim.
- [ROCm kernel inventory](kernel-inventory.md) explains MFMA/WMMA contracts;
  it is not a mutable status ledger.

## Architecture and decisions

[ROCm audit](../../audit/backend/rocm/ROCM_AUDIT.md) owns target-specific
decisions and current deltas. [Strix Halo execution plan](../../audit/backend/rocm/STRIX_HALO_EXECUTION_PLAN.md)
records active execution work; older material remains in the audit archive.
