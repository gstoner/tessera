// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=parent_view input=tile output=target arch=gfx1151})' %s | FileCheck %s --check-prefix=VIEW1151
// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=parent_view input=tile output=target arch=gfx1201})' %s | FileCheck %s --check-prefix=VIEW1201
// RUN: not %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=parent_view input=tile output=target arch=gfx1200})' %s 2>&1 | FileCheck %s --check-prefix=VIEW1200
// RUN: not %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=block_quant input=tile output=target arch=gfx1200})' %s 2>&1 | FileCheck %s --check-prefix=QUANT1200
//
// The C++ mirror of `rocm_pipeline.promoted_families` for the two
// architecture-owned quantized-MLP members. gfx1151 joined gfx1201 on
// Princess-Luna device proof (GFX1151-QUANTIZED-MLP-20261030 slice 4); any
// other arch still refuses. The block_quant body check (one schema-4 row
// region) is exercised by the packagers; here it is the arch refusal.

module {}

// VIEW1151: tessera.pipeline.family = "parent_view"
// VIEW1201: tessera.pipeline.family = "parent_view"
// VIEW1200: parent_view requires owning gfx1201/gfx1151 native Graph -> Schedule -> Tile entry
// QUANT1200: block_quant requires owning gfx1201/gfx1151 native Graph -> Schedule -> Tile schema 4
