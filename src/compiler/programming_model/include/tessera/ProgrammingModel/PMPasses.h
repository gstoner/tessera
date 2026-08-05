//===- PMPasses.h — Programming Model v1.1 pass interfaces ----------------===//
//
// Library-owned declarations for the Programming Model v1.1 passes. These were
// previously defined inside the `tools/tessera-opt/` driver source, which made
// them reachable only through that translation unit's pipeline builders
// (W0.6). They now live in a library so other components can construct them
// directly and lit fixtures can target each pass in isolation.
//
// MATURITY — read before depending on these:
//
//   createPMV11VerifierPass()      REAL. Structurally verifies schedule.* /
//                                  cache.* / tile.* ops (mbarrier scope and
//                                  ordering, pipeline micro-batch counts,
//                                  async-copy stages, knob arity).
//
//   createGraphToSchedulePass()    REAL FOR ONE BOUNDED SLICE. Creates a
//                                  content-addressed schedule.matmul SSA edge
//                                  for static rank-2 x86 f32 and ROCm f16/f32
//                                  Graph matmul. Other contracts fail closed.
//
//   createScheduleToTilePass()     REAL FOR THE SAME SLICE. Atomically consumes
//                                  schedule.matmul and its Graph producer,
//                                  materializes A/B/D/M/N/K, and emits one
//                                  typed launch-level tile.matmul_kernel.
//
// Decision #29 (a declaration must have a consumer) and Decision #31 (one
// implementation per boundary) both bear on the two skeletons: they register
// under production-sounding pass names, so their descriptions and this header
// must state plainly that they do not lower. Do not cite either as evidence
// that a Graph->Schedule or Schedule->Tile boundary is implemented in C++.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_PROGRAMMING_MODEL_PMPASSES_H
#define TESSERA_PROGRAMMING_MODEL_PMPASSES_H

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;
class DialectRegistry;
}  // namespace mlir

namespace tessera {

/// Verifies all Programming Model v1.1 ops. This is a real verifier.
std::unique_ptr<mlir::Pass> createPMV11VerifierPass();

/// Build the bounded mixed-level static-matmul Schedule SSA contract.
std::unique_ptr<mlir::Pass> createGraphToSchedulePass();

/// Consume that contract into the six-operand launch-level Tile matmul ABI.
std::unique_ptr<mlir::Pass> createScheduleToTilePass();

/// Registers PM v1.1 dialects for `tessera-opt` parsing.
void registerPMPipelinesV11(mlir::DialectRegistry &registry);

/// Verify-only pipeline: verifier + CSE + canonicalize. No lowering.
void buildPMV11VerifyPipeline(mlir::OpPassManager &pm);

/// Verifier + bounded Graph/Schedule-to-Tile static-matmul lowering.
void buildPMV11LegalizePipeline(mlir::OpPassManager &pm);

/// Registers the passes and pipelines with the global MLIR pass registry.
void registerPMV11Passes();

}  // namespace tessera

#endif  // TESSERA_PROGRAMMING_MODEL_PMPASSES_H
