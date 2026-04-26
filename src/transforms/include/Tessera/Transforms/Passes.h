
#pragma once
#include "mlir/Pass/Pass.h"

namespace tessera {

// Phase 1 passes
std::unique_ptr<mlir::Pass> createCanonicalizeTesseraIRPass();
std::unique_ptr<mlir::Pass> createVerifyTesseraIRPass();
std::unique_ptr<mlir::Pass> createMigrateTesseraIRPass();

// Phase 2 lowering chain
//
// Pipeline order:
//   tessera-distribution-lowering
//   → tessera-effect-annotation
//   → tessera-tiling
//   → tessera-tile-to-x86
//
// Run the whole chain with: -tessera-lower-to-x86

// DistributionLoweringPass — converts tessera.shard argument attributes into
// schedule.mesh.define + schedule.mesh.region ops wrapping the function body.
// Options:
//   --mesh-axes  comma-separated axis names (e.g. "dp,tp")
//   --mesh-sizes comma-separated axis sizes (e.g. "4,4")
std::unique_ptr<mlir::Pass> createDistributionLoweringPass();

// EffectAnnotationPass — infers side-effect class from the function body and
// attaches tessera.effect = "pure"|"random"|"memory"|"io" to each func.func.
// Signals failure if a func annotated "pure" contains random or memory ops.
std::unique_ptr<mlir::Pass> createEffectAnnotationPass();

// TilingPass — tiles tessera.matmul ops (inside schedule.mesh.region bodies)
// into scf.for loop nests over M and N tiles using tensor.extract/insert_slice.
// Options:
//   --tile-m  tile size along M dimension (default 16)
//   --tile-n  tile size along N dimension (default 16)
std::unique_ptr<mlir::Pass> createTilingPass();

// TileToX86Pass — replaces tiled tessera.matmul ops (static bf16 operands)
// with calls to the tessera_x86_backend C functions via func.call with raw
// i64 pointer arguments.  Handles fused_epilogue bias/gelu variants.
// Options:
//   --prefer-amx  prefer AMX over AVX-512 when both are available (default true)
std::unique_ptr<mlir::Pass> createTileToX86Pass();

void registerTesseraPasses();

} // namespace tessera
