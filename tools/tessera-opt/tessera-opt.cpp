
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"  // Phase 4 GPU emission: per-pass register decls
// Phase 4 GPU emission: BufferizableOpInterface external models — without these,
// one-shot-bufferize reports "op was not bufferized" for linalg/tensor/etc.
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"  // canonicalize / cse (per-op folders)
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"  // report_fatal_error (tessera-emit-nvvm)

#ifdef TESSERA_HAVE_CORE_TESSERA_IR
#include "Tessera/IR/Dialects.h"
#include "Tessera/Dialect/Tile/TileDialect.h"
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
// Stage L3 — in-process MLIR -> hsaco serialization (no mlir-opt shell-out).
// Only meaningful in a full ROCm build (real HIP toolchain): the lean
// artifact driver stays lean.
#ifdef TESSERA_HAVE_CORE_TESSERA_IR
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
// convert-gpu-to-rocdl gathers cf/arith/func/memref/vector/index/ub -> LLVM
// patterns through each dialect's ConvertToLLVMPatternInterface external model;
// those must be registered for the gpu.func body (incl. cf block args) to fully
// lower (mlir-opt does this via registerAllExtensions).
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "llvm/Support/TargetSelect.h"
#endif
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
//
// Sprint 10 (2026-06-03) — Apple reasoning-model attention-family prologue.
// Run the Graph IR attention-family *recognizer* passes (SwiGLU / MLA / DeepSeek
// NSA / Ling-Kimi hybrid / Lightning / DeltaNet-Kimi) BEFORE distribution and
// tiling, exactly as `buildCUDA13Pipeline` does for NVIDIA. This makes reasoning
// models compiler-visible on the Apple spine: MLA / NSA fuse into their fused
// ops, and Lightning / Delta / Hybrid run their (currently IR-preserving) pass
// slots so a later backend rewrite has a stable position to attach to. The stage
// is IR-preserving for inputs it does not recognize — it never blocks the linalg
// value lane that the rest of the `-full` pipeline drives.
auto addAppleReasoningAttentionPrologue = [](mlir::OpPassManager &pm) {
  pm.addPass(tessera::createSwigluFusionPass());
  pm.addPass(tessera::createMLAFusionPass());
  pm.addPass(tessera::createNativeSparseAttnFusionPass());
  pm.addPass(tessera::createHybridAttnExpandPass());
  pm.addPass(tessera::createLightningAttnFusionPass());
  pm.addPass(tessera::createDeltaAttnChunkingPass());
  pm.addPass(tessera::createLookaheadSparseAttnExpandPass());
  pm.addPass(tessera::createMSAExpandPass());
  pm.addPass(tessera::createRLLossDecomposePass());
};

class VerifyAppleValueTileIRPass
    : public mlir::PassWrapper<VerifyAppleValueTileIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAppleValueTileIRPass)

  llvm::StringRef getArgument() const override {
    return "tessera-verify-apple-value-tile-ir";
  }

  llvm::StringRef getDescription() const override {
    return "Verify that Apple value pipelines contain only registered, "
           "allowlisted Tile IR ops before Tile->Apple lowering";
  }

  void runOnOperation() override {
    bool failed = false;
    getOperation().walk([&](mlir::Operation *op) {
      llvm::StringRef name = op->getName().getStringRef();
      if (!name.starts_with("tile."))
        return;
      if (!op->getName().getRegisteredInfo()) {
        op->emitError("apple value Tile IR contains unregistered op '")
            << name << "'; value pipelines must not rely on opaque tile.* ops";
        failed = true;
        return;
      }
      if (!isAllowedValueTileOp(name)) {
        op->emitError("apple value Tile IR op '")
            << name
            << "' is outside the value allowlist (linalg family, "
               "rank-2 matmul/gemm, rank-3 batched_gemm, PPO policy loss, "
               "EBM value kernels, GA/Clifford value seam)";
        failed = true;
      }
    });
    if (failed)
      signalPassFailure();
  }

private:
  static bool isAllowedValueTileOp(llvm::StringRef name) {
    return name == "tile.matmul" || name == "tile.gemm" ||
           name == "tile.batched_gemm" || name == "tile.ppo_policy_loss" ||
           name == "tile.ebm_energy_quadratic" ||
           name == "tile.ebm_langevin_step" ||
           name == "tile.ebm_refinement" ||
           name == "tile.ebm_partition_exact" ||
           name == "tile.clifford_geometric_product" ||
           name == "tile.clifford_outer_product" ||
           name == "tile.clifford_inner_product" ||
           name == "tile.clifford_reverse" ||
           name == "tile.clifford_grade_project" ||
           name == "tile.clifford_norm" ||
           name == "tile.clifford_rotor_sandwich" ||
           name == "tile.cholesky" ||
           name == "tile.tri_solve" || name == "tile.cholesky_solve" ||
           name == "tile.lu" || name == "tile.qr" || name == "tile.svd";
  }
};

std::unique_ptr<mlir::Pass> createVerifyAppleValueTileIRPass() {
  return std::make_unique<VerifyAppleValueTileIRPass>();
}

mlir::PassPipelineRegistration<> gAppleCPUFullPipeline(
    "tessera-lower-to-apple_cpu-full",
    "Full Graph->Schedule->Tile->Target Apple CPU spine (effect-annotation -> "
    "distribution-lowering -> tiling -> tile-to-apple_cpu). Sprint 10: runs the "
    "reasoning-model attention-family prologue before distribution/tiling.",
    [](mlir::OpPassManager &pm) {
      pm.addPass(tessera::createEffectAnnotationPass());
      // Sprint 10: reasoning-model attention-family stage (compiler-visible).
      addAppleReasoningAttentionPrologue(pm);
      pm.addPass(tessera::createDistributionLoweringPass());
      // Sprint 5: value-mode tiling preserves static rank-2 f32 matmul/gemm as a
      // single tile op for the Accelerate GEMM value call (CPU `-full` only).
      pm.addPass(tessera::createTilingPass(/*valueMode=*/true));
      // Sprint 11: `--allow-unregistered-dialect` was necessary but not
      // sufficient. The Tile dialect still permits unknown ops for legacy
      // artifact lanes, so the value spine gets its own hard gate here.
      pm.addPass(createVerifyAppleValueTileIRPass());
      // Apple Value Target IR sprint: the `-full` pipeline is value-preserving
      // — it emits value-producing tessera_apple.cpu.call ops (no artifact
      // metadata / ub.poison husk) and fails with a named diagnostic if an op
      // has no value lowering.
      pm.addPass(tessera::apple::createLowerTileToAppleCPUPass(/*valueMode=*/true));
    });

mlir::PassPipelineRegistration<> gAppleGPUFullPipeline(
    "tessera-lower-to-apple_gpu-full",
    "Full Graph->Schedule->Tile->Target Apple GPU spine (effect-annotation -> "
    "distribution-lowering -> tiling -> tile-to-apple_gpu, value-preserving). "
    "Sprint 10: runs the reasoning-model attention-family prologue before "
    "distribution/tiling.",
    [](mlir::OpPassManager &pm) {
      pm.addPass(tessera::createEffectAnnotationPass());
      // Sprint 10: reasoning-model attention-family stage (compiler-visible).
      addAppleReasoningAttentionPrologue(pm);
      pm.addPass(tessera::createDistributionLoweringPass());
      // Sprint 8: value-mode tiling preserves static rank-3 f32/f16/bf16
      // batched matmul as a single tile.batched_gemm for the GPU bmm value call
      // (rank-2 matmul → tile.matmul stays gated in the GPU value block).
      pm.addPass(tessera::createTilingPass(/*valueMode=*/true));
      // Sprint 11: reject opaque tile.* ops in the value spine before any
      // backend-specific handoff can treat them as valid compiler IR.
      pm.addPass(createVerifyAppleValueTileIRPass());
      pm.addPass(tessera::apple::createLowerTileToAppleGPUPass(/*valueMode=*/true));
    });
} // namespace
#endif

// Phase 4 GPU emission (2026-06-17): convenience alias for the linalg→gpu→NVVM
// emission spine — tessera kernel → linalg → scf.parallel → gpu.launch → NVVM IR
// text. EMISSION ONLY: the NVVM kernel is produced for inspection/codegen; GPU
// launch (cuLaunchKernel/hipLaunchKernel) stays hardware-gated. Composed via
// parsePassPipeline so it reuses the upstream passes registered in main().
static mlir::PassPipelineRegistration<> gEmitNVVM(
    "tessera-emit-nvvm",
    "Phase 4 GPU emission: lower a tessera kernel through linalg -> scf.parallel "
    "-> gpu -> NVVM (emission only; GPU launch is hardware-gated).",
    [](mlir::OpPassManager &pm) {
      if (failed(mlir::parsePassPipeline(
              "func.func(tessera-to-linalg),empty-tensor-to-alloc-tensor,"
              "one-shot-bufferize{bufferize-function-boundaries=true},"
              "func.func(convert-linalg-to-parallel-loops),"
              "func.func(gpu-map-parallel-loops),"
              "func.func(convert-parallel-loops-to-gpu),gpu-kernel-outlining,"
              "gpu.module(lower-affine,convert-gpu-to-nvvm)",
              pm)))
        llvm::report_fatal_error("tessera-emit-nvvm: failed to build pipeline");
    });

// ROCDL twin of tessera-emit-nvvm — identical spine, AMD GPU backend
// (convert-gpu-to-rocdl). Emission only; HIP launch is hardware-gated.
static mlir::PassPipelineRegistration<> gEmitROCDL(
    "tessera-emit-rocdl",
    "Phase 4 GPU emission: lower a tessera kernel through linalg -> scf.parallel "
    "-> gpu -> ROCDL (emission only; GPU launch is hardware-gated).",
    [](mlir::OpPassManager &pm) {
      if (failed(mlir::parsePassPipeline(
              "func.func(tessera-to-linalg),empty-tensor-to-alloc-tensor,"
              "one-shot-bufferize{bufferize-function-boundaries=true},"
              "func.func(convert-linalg-to-parallel-loops),"
              "func.func(gpu-map-parallel-loops),"
              "func.func(convert-parallel-loops-to-gpu),gpu-kernel-outlining,"
              "gpu.module(lower-affine,convert-gpu-to-rocdl)",
              pm)))
        llvm::report_fatal_error("tessera-emit-rocdl: failed to build pipeline");
    });

int main(int argc, char **argv) {
#if defined(TESSERA_HAVE_ROCM_BACKEND) && defined(TESSERA_HAVE_CORE_TESSERA_IR)
  // Stage L3: register the LLVM AMDGPU target (codegen + asm printer) so the
  // gpu-module-to-binary pass can emit + ld.lld-link the hsaco entirely
  // in-process. AMDGPU codegen ships in the shared libLLVM; ld.lld is found on
  // PATH by the ROCDL target serializer.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
#endif
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
  // Upstream canonicalize / cse so Tessera per-op folders + canonicalizers
  // (Phase 1: identity cast, transpose-of-transpose, …) are inspectable in lit.
  mlir::registerTransformsPasses();
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

#if defined(TESSERA_HAVE_APPLE_BACKEND) && defined(TESSERA_HAVE_CORE_TESSERA_IR)
  ::mlir::registerPass([]() { return createVerifyAppleValueTileIRPass(); });
#endif

#ifdef TESSERA_HAVE_ROCM_BACKEND
  mlir::tessera_rocm::registerTesseraROCMBackendPasses();
#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  // Stage L3 — the upstream GPU serialization spine, so the WHOLE chain runs in
  // ONE tessera-opt invocation (a runtime launch lane can't shell out to
  // mlir-opt). The full pipeline as a single --pass-pipeline string:
  //   builtin.module(generate-wmma-gemm-kernel, lower-tessera-target-to-rocdl,
  //     gpu.module(convert-scf-to-cf, convert-gpu-to-rocdl,
  //                reconcile-unrealized-casts),
  //     rocdl-attach-target{chip=gfx1151}, gpu-module-to-binary)
  // convert-gpu-to-rocdl is already registered below; add the rest.
  mlir::registerSCFToControlFlowPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerGpuROCDLAttachTargetPass();
  mlir::registerGpuModuleToBinaryPass();
#endif
#endif
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendPasses();
#endif

  // Phase 4 GPU emission (2026-06-17): register the upstream passes that compose
  // the linalg→gpu→NVVM emission spine, so the `tessera-emit-nvvm` pipeline (and
  // ad-hoc --pass-pipeline lit fixtures) can lower a tessera kernel to NVVM IR
  // text. Emission only — GPU launch (cuLaunchKernel/hipLaunchKernel) stays
  // hardware-gated. Only the specific passes we use are registered (minimal
  // link surface), not the full conversion umbrella.
  mlir::registerConvertLinalgToParallelLoopsPass();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerGpuMapParallelLoopsPass();
  mlir::registerGpuKernelOutliningPass();
  mlir::registerConvertParallelLoopToGpuPass();
  mlir::registerLowerAffinePass();  // affine.apply in the outlined kernel → std
  mlir::registerConvertGpuOpsToNVVMOpsPass();
  mlir::registerConvertGpuOpsToROCDLOpsPass();  // ROCDL twin of the NVVM lane

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  mlir::ub::UBDialect,
                  mlir::vector::VectorDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect,
                  mlir::ROCDL::ROCDLDialect>();

  // Phase 4 GPU emission: BufferizableOpInterface external models so
  // one-shot-bufferize can lower linalg/tensor/scf/arith + func boundaries.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  tessera::registerTesseraDialects(registry);
  // Sprint 9 — Tile IR dialect (value-lane lowering spine). Registering it lets
  // the Apple `-full` pipelines run without --allow-unregistered-dialect.
  tessera::tile::registerTileDialect(registry);
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
#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  // Stage L3: LLVM-IR translations + the #rocdl.target serialization interface
  // that gpu-module-to-binary needs to lower the gpu.module to an ELF hsaco.
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(registry);
  // ConvertToLLVM external models so convert-gpu-to-rocdl lowers the whole
  // gpu.func body (cf block args, arith, memref, vector, index, ub, func).
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
#endif
#endif
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendDialects(registry);
#endif

  return failed(mlir::MlirOptMain(argc, argv, "tessera-opt\n", registry));
#endif
}
