//===- TesseraMLIRPlugin.h — Global dialect/pass registration entry --------===//
//
// Central plugin interface for the Tessera compiler MLIR layer.
//
// This header declares the three registration functions that must be called
// once (typically from tessera-opt main() or a dynamic plugin loader) before
// any Tessera IR can be parsed, verified, or compiled:
//
//   registerTesseraDialects(registry)   — insert all Tessera dialects
//   registerTesseraAllPasses()          — register all Tessera passes
//   registerTesseraAllPipelines()       — register all pass pipelines
//
// Usage in tessera-opt::main():
//
//   DialectRegistry registry;
//   registerTesseraDialects(registry);
//   registerTesseraAllPasses();
//   registerTesseraAllPipelines();
//   mlir::MlirOptMain(argc, argv, "Tessera optimizer", registry);
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace tessera {

// ---------------------------------------------------------------------------
// Dialect registration
//
// Inserts every Tessera dialect into the given DialectRegistry so that
// tessera-opt can parse .mlir files that use those dialect prefixes.
//
// Dialects registered:
//   tessera.neighbors.*  — Stencil / halo / topology ops
//   tessera.queue.*      — Warp-specialised token queues
//   tessera.attn.*       — Flash-attention ops (online softmax, LSE)
//   tessera.schedule.*   — Programming Model v1.1 Schedule dialect
//   tessera.cache.*      — Programming Model v1.1 Cache dialect
//   tessera.tile.*       — Programming Model v1.1 TileMemory dialect
// ---------------------------------------------------------------------------
void registerTesseraDialects(mlir::DialectRegistry &registry);

// ---------------------------------------------------------------------------
// Pass registration
//
// Registers every Tessera pass individually so they appear in
// tessera-opt --help and can be selected via -pass-pipeline.
//
// Registered passes:
//   -tessera-halo-infer            (HaloInferPass)
//   -tessera-stencil-lower         (StencilLowerPass)
//   -tessera-pipeline-overlap      (PipelineOverlapPass)
//   -tessera-dynamic-topology      (DynamicTopologyPass)
//   -tessera-pm-verify             (PMV11VerifierPass)
//   -tessera-graph-to-schedule     (GraphToSchedulePass)
//   -tessera-schedule-to-tile      (ScheduleToTilePass)
//   -tessera-lower-to-stablehlo    (TPU backend)
//   -tessera-annotate-sharding     (TPU backend)
//   -tessera-export-shardy         (ShardyExportPass)
// ---------------------------------------------------------------------------
void registerTesseraAllPasses();

// ---------------------------------------------------------------------------
// Pipeline registration
//
// Registers high-level pass pipelines that combine multiple passes into
// end-to-end compilation flows.
//
// Registered pipelines:
//   tessera-neighbors-pipeline     — HaloInfer + StencilLower + PipelineOverlap
//   tessera-pm-verify-pipeline     — PMV11VerifierPass + CSE + Canonicalize
//   tessera-pm-legalize-pipeline   — verify + GraphToSchedule + ScheduleToTile
//   tessera-tpu-backend            — full TPU lowering (stablehlo + shardy)
//   tessera-full-pipeline          — all of the above end-to-end
// ---------------------------------------------------------------------------
void registerTesseraAllPipelines();

// ---------------------------------------------------------------------------
// Convenience: call all three in order.
// ---------------------------------------------------------------------------
inline void registerTesseraAll(mlir::DialectRegistry &registry) {
  registerTesseraDialects(registry);
  registerTesseraAllPasses();
  registerTesseraAllPipelines();
}

// ---------------------------------------------------------------------------
// Per-layer pipeline builders (used internally and in tests).
// ---------------------------------------------------------------------------

/// Build the Neighbors stencil/halo compilation pipeline.
void buildNeighborsPipeline(mlir::OpPassManager &pm);

/// Build the Programming Model v1.1 verification pipeline.
void buildPMVerifyPipeline(mlir::OpPassManager &pm);

/// Build the Programming Model v1.1 legalization pipeline.
void buildPMLegalizePipeline(mlir::OpPassManager &pm);

/// Build the full Tessera → StableHLO + Shardy pipeline.
void buildFullPipeline(mlir::OpPassManager &pm);

} // namespace tessera
