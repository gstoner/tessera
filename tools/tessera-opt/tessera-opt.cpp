
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef TESSERA_HAVE_CORE_TESSERA_IR
#include "Tessera/IR/Dialects.h"
#include "Tessera/Transforms/Passes.h"
#endif

// Sprint V7 (2026-05-22): FA-4 attention dialect registration.
// Wrapped behind a separate define so the dialect can be unwired
// independently of the core IR if a constrained build needs it.
#ifdef TESSERA_HAVE_FA4_ATTN
#include "tessera/Dialect/Attn/AttnDialect.h"
#endif

// Sprint V8 (2026-05-22): FA-4 tile-queue dialect registration.
// Same separation as V7's TESSERA_HAVE_FA4_ATTN.
#ifdef TESSERA_HAVE_FA4_QUEUE
#include "tessera/Dialect/Queue/QueueDialect.h"
#endif

#ifdef TESSERA_HAVE_SOLVERS
#include "SolversPasses.h"
#include "tessera/Dialect/Solver/SolverDialect.h"
#include "tessera/Solvers/LinalgPasses.h"
#endif

#ifdef TESSERA_HAVE_SCALING_RESILIENCE
#include "tessera/sr/Passes.h"
#endif

#ifdef TESSERA_HAVE_NEIGHBORS
#include "tessera/Dialect/Neighbors/IR/NeighborsDialect.h"
#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#endif

#ifdef TESSERA_HAVE_TPP
#include "tpp/InitTPP.h"
#endif

#ifdef TESSERA_HAVE_APPLE_BACKEND
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"
#endif

#ifdef TESSERA_HAVE_ROCM_BACKEND
#include "TesseraROCM/Passes.h"
#endif

#ifdef TESSERA_HAVE_NVIDIA_BACKEND
#include "tessera/gpu/BackendRegistration.h"
#endif

namespace tessera {
namespace diagnostics {
void registerShapeInferencePass();
void registerErrorReporterPass();
} // namespace diagnostics
} // namespace tessera

#if defined(TESSERA_HAVE_APPLE_BACKEND) && defined(TESSERA_HAVE_CORE_TESSERA_IR)
namespace {
// L-series linalg pilot (2026-06-02) — the full Graph→Schedule→Tile→Target
// Apple spine in a single alias.  Chains the SSA dataflow passes
// (effect-annotation → distribution-lowering → tiling) with the Apple Target-IR
// artifact projection (tile-to-apple_{cpu,gpu}).  Unlike the artifact-only
// `tessera-lower-to-apple_{cpu,gpu}` (which assume Tile-IR input) and the
// op-direct `-runtime` pipelines, this drives the whole stack from Graph IR.
// Registered here (not in the Apple backend library) because it spans Transforms
// passes that the backend library does not link.
mlir::PassPipelineRegistration<> gAppleCPUFullPipeline(
    "tessera-lower-to-apple_cpu-full",
    "Full Graph->Schedule->Tile->Target Apple CPU spine (effect-annotation -> "
    "distribution-lowering -> tiling -> tile-to-apple_cpu)",
    [](mlir::OpPassManager &pm) {
      pm.addPass(tessera::createEffectAnnotationPass());
      pm.addPass(tessera::createDistributionLoweringPass());
      pm.addPass(tessera::createTilingPass());
      pm.addPass(tessera::apple::createLowerTileToAppleCPUPass());
    });

mlir::PassPipelineRegistration<> gAppleGPUFullPipeline(
    "tessera-lower-to-apple_gpu-full",
    "Full Graph->Schedule->Tile->Target Apple GPU spine (effect-annotation -> "
    "distribution-lowering -> tiling -> tile-to-apple_gpu)",
    [](mlir::OpPassManager &pm) {
      pm.addPass(tessera::createEffectAnnotationPass());
      pm.addPass(tessera::createDistributionLoweringPass());
      pm.addPass(tessera::createTilingPass());
      pm.addPass(tessera::apple::createLowerTileToAppleGPUPass());
    });
} // namespace
#endif

int main(int argc, char **argv) {
#if (defined(TESSERA_HAVE_ROCM_BACKEND) || defined(TESSERA_HAVE_NVIDIA_BACKEND)) && !defined(TESSERA_HAVE_CORE_TESSERA_IR)
  // Hardware-free target artifact builds intentionally keep tessera-opt lean:
  // only the dialects and passes needed by the target contract spine are
  // registered, avoiding a dependency on every upstream MLIR component.
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendPasses();
#endif
#ifdef TESSERA_HAVE_ROCM_BACKEND
  mlir::tessera_rocm::registerTesseraROCMBackendPasses();
#endif

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                  mlir::ROCDL::ROCDLDialect>();
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendDialects(registry);
#endif
#ifdef TESSERA_HAVE_ROCM_BACKEND
  mlir::tessera_rocm::registerTesseraROCMBackendDialects(registry);
#endif

  return failed(mlir::MlirOptMain(argc, argv, "tessera-opt\n", registry));
#else
#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  tessera::registerTesseraPasses();
  tessera::diagnostics::registerShapeInferencePass();
  tessera::diagnostics::registerErrorReporterPass();
#endif

#ifdef TESSERA_HAVE_SOLVERS
  tessera::passes::registerTesseraSolversPipeline();
  tessera::solver::registerTesseraLinalgSolverPipeline();
#endif

#ifdef TESSERA_HAVE_SCALING_RESILIENCE
  mlir::tessera::sr::registerPasses();
#endif

#ifdef TESSERA_HAVE_NEIGHBORS
  // Phase 7: Neighbors dialect passes (halo infer, stencil lower,
  // pipeline overlap, dynamic topology).
  tessera::neighbors::registerHaloInferPass();
  tessera::neighbors::registerStencilLowerPass();
  tessera::neighbors::registerBoundaryConditionLowerPass();
  tessera::neighbors::registerStencilLoopMaterializePass();
  tessera::neighbors::registerHaloMeshIntegrationPass();
  tessera::neighbors::registerHaloTransportLowerPass();
  tessera::neighbors::registerPipelineOverlapPass();
  tessera::neighbors::registerDynamicTopologyPass();
#endif

#ifdef TESSERA_HAVE_TPP
  // TPP solver passes + `-tpp-space-time` pipeline alias.
  tessera::tpp::registerTPPPasses();
  tessera::tpp::registerTPPPipelines();
#endif

#ifdef TESSERA_HAVE_APPLE_BACKEND
  // Phase 8: Apple Silicon Target IR pipelines
  // (tessera-lower-to-apple_cpu, tessera-lower-to-apple_gpu).
  tessera::apple::registerTesseraAppleBackendPipelines();
#endif

#ifdef TESSERA_HAVE_ROCM_BACKEND
  mlir::tessera_rocm::registerTesseraROCMBackendPasses();
#endif
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendPasses();
#endif

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect,
                  mlir::ROCDL::ROCDLDialect>();

#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  tessera::registerTesseraDialects(registry);
#endif

#ifdef TESSERA_HAVE_FA4_ATTN
  // Sprint V7 (2026-05-22) — FA-4 attention dialect.  Unblocks the
  // three `tessera.attn.scaled_dot_product` lit fixtures
  // (flash_attn_full.mlir, tile_ir_lowering.mlir, V6c) that were
  // XFAIL'd because tessera-opt could not load this dialect.
  tessera::attn::registerAttnDialect(registry);
#endif

#ifdef TESSERA_HAVE_FA4_QUEUE
  // Sprint V8 (2026-05-22) — FA-4 tile-queue dialect.  Required for
  // the queue-op verifier lit fixtures and any future IR that uses
  // `!tessera.queue.tile_queue` / `!tessera.queue.token` types
  // directly (rather than only through the FA-4 lowering passes).
  tessera::queue::registerQueueDialect(registry);
#endif

#ifdef TESSERA_HAVE_SOLVERS
  tessera::solver::registerTesseraLinalgSolverDialect(registry);
#endif
#ifdef TESSERA_HAVE_NEIGHBORS
  tessera::neighbors::registerNeighborsDialect(registry);
#endif

#ifdef TESSERA_HAVE_TPP
  tessera::tpp::registerTPPDialect(registry);
#endif

#ifdef TESSERA_HAVE_APPLE_BACKEND
  tessera::apple::registerTesseraAppleBackendDialects(registry);
#endif

#ifdef TESSERA_HAVE_ROCM_BACKEND
  mlir::tessera_rocm::registerTesseraROCMBackendDialects(registry);
#endif
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendDialects(registry);
#endif

  return failed(mlir::MlirOptMain(argc, argv, "tessera-opt\n", registry));
#endif
}
