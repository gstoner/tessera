
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
#include "mlir/Dialect/Math/IR/Math.h"  // flash_attn softmax exp
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"  // Phase 4 GPU emission: per-pass register decls
#include "mlir/InitAllExtensions.h"
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
#include "llvm/Support/raw_ostream.h"  // --tessera-build-info
#include <string>

#ifdef TESSERA_HAVE_CORE_TESSERA_IR
#include "Tessera/IR/Dialects.h"
#include "Tessera/Transforms/Passes.h"
#endif
#if defined(TESSERA_HAVE_CORE_TESSERA_IR) ||                                  \
    defined(TESSERA_HAVE_NVIDIA_BACKEND) || defined(TESSERA_HAVE_ROCM_BACKEND)
#include "Tessera/Dialect/Tile/TileDialect.h"
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

// W0.10 (2026-08-02): hardware-free x86 Target IR dialect. x86 was the one
// backend with no Target IR dialect at all, though Decision #19 says backends
// MUST expose one; no carve-out was granted, so it is built and registered.
#ifdef TESSERA_HAVE_X86_TARGET_IR
#include "TesseraX86/IR/TesseraX86Dialect.h"
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
      if (!tessera::apple::isValueLaneTileOp(name)) {
        op->emitError("apple value Tile IR op '")
            << name << "' is outside the value allowlist ("
            << tessera::apple::valueLaneTileOpEnvelopeDescription() << ")";
        failed = true;
      }
    });
    if (failed)
      signalPassFailure();
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

// Phase 4 GPU emission (2026-06-17): convenience aliases for the linalg→gpu→
// {NVVM,ROCDL} emission spine — tessera kernel → linalg → scf.parallel →
// gpu.launch → target IR text. EMISSION ONLY: the kernel is produced for
// inspection/codegen; GPU launch (cuLaunchKernel/hipLaunchKernel) stays
// hardware-gated. Composed via parsePassPipeline so it reuses the upstream
// passes registered in main().
//
// The two lanes are the same eight stages and differ only in the final
// gpu-to-<target> conversion, so the spine is written once. Divergence between
// them used to be a copy-paste edit away.
namespace {
constexpr llvm::StringLiteral kEmitSpinePrefix =
    "func.func(tessera-to-linalg),empty-tensor-to-alloc-tensor,"
    "one-shot-bufferize{bufferize-function-boundaries=true},"
    "func.func(convert-linalg-to-parallel-loops),"
    "func.func(gpu-map-parallel-loops),"
    "func.func(convert-parallel-loops-to-gpu),gpu-kernel-outlining,"
    "gpu.module(lower-affine,";

// Register the two emission aliases.
//
// Uses `registerPassPipeline` rather than the `PassPipelineRegistration<>`
// convenience wrapper because that wrapper's builder returns `void`: a failed
// `parsePassPipeline` would leave the pass manager empty and the driver would
// run a silent no-op pipeline, reporting success for a translation it never
// performed (Decision #21). The registry's error handler turns the same failure
// into a real driver diagnostic naming the alias.
//
// Only registered when the core Tessera IR is linked in — the spine opens with
// `tessera-to-linalg`, so advertising these aliases in a lean artifact driver
// would offer a pipeline that cannot be built.
// Builder for one emission alias. Shared, so the eight-stage spine cannot
// drift between the NVVM and ROCDL lanes; the `registerPassPipeline` calls
// below stay separate and literal so a source-scanning drift gate
// (tests/unit/test_pipeline_registry.py) can still see both names.
mlir::PassRegistryFunction emitPipelineBuilder(llvm::StringRef gpuConversionPass) {
  return [pass = gpuConversionPass.str()](
             mlir::OpPassManager &pm, llvm::StringRef options,
             llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                 errorHandler) {
    if (!options.empty())
      return errorHandler("this pipeline takes no options");
    std::string pipeline = (kEmitSpinePrefix + pass + ")").str();
    if (failed(mlir::parsePassPipeline(pipeline, pm)))
      return errorHandler(
          "failed to build '" + pipeline +
          "'; the required upstream passes are not registered in this "
          "tessera-opt build (see --tessera-build-info)");
    return mlir::success();
  };
}

void noEmitPipelineOptions(
    llvm::function_ref<void(const mlir::detail::PassOptions &)>) {}

void registerEmitPipelines() {
  mlir::registerPassPipeline(
      "tessera-emit-nvvm",
      "Phase 4 GPU emission: lower a tessera kernel through linalg -> "
      "scf.parallel -> gpu -> NVVM (emission only; GPU launch is "
      "hardware-gated).",
      emitPipelineBuilder("convert-gpu-to-nvvm"), noEmitPipelineOptions);
  mlir::registerPassPipeline(
      "tessera-emit-rocdl",
      "Phase 4 GPU emission: lower a tessera kernel through linalg -> "
      "scf.parallel -> gpu -> ROCDL (emission only; HIP launch is "
      "hardware-gated).",
      emitPipelineBuilder("convert-gpu-to-rocdl"), noEmitPipelineOptions);
}
} // namespace

// Build identity. `TESSERA_OPT_BUILD_{PROFILE,FEATURES}` come from the feature
// ledger in CMakeLists.txt, so this cannot drift from what was actually linked.
// Reported through the tool banner (`--help`) and via `--tessera-build-info`,
// because "which tessera-opt is this?" is otherwise only answerable by diffing
// the registered pass list — the failure mode when a stale build directory is
// picked up ahead of a current one.
#ifndef TESSERA_OPT_BUILD_PROFILE
#define TESSERA_OPT_BUILD_PROFILE "unknown"
#endif
#ifndef TESSERA_OPT_BUILD_FEATURES
#define TESSERA_OPT_BUILD_FEATURES "unknown"
#endif

namespace {
const char *tesseraOptBuildInfo() {
  static const std::string info =
      std::string("tessera-opt\n  build profile: ") + TESSERA_OPT_BUILD_PROFILE +
      "\n  features: " + TESSERA_OPT_BUILD_FEATURES + "\n";
  return info.c_str();
}

// Detected (not consumed) before MlirOptMain so `--tessera-build-info` works
// with no input file and without registering a real CLI option.
bool hasBuildInfoFlag(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (llvm::StringRef(argv[i]) == "--tessera-build-info")
      return true;
  return false;
}
} // namespace

int main(int argc, char **argv) {
  if (hasBuildInfoFlag(argc, argv)) {
    llvm::outs() << tesseraOptBuildInfo();
    return 0;
  }
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
#ifdef TESSERA_OPT_LEAN_ARTIFACT_DRIVER
  // Hardware-free target artifact builds intentionally keep tessera-opt lean:
  // only the dialects and passes needed by the target contract spine are
  // registered, avoiding a dependency on every upstream MLIR component.
  //
  // This arm is selected by an explicit CMake intent, not re-derived from
  // backend macros. CMakeLists.txt fails configuration if any feature outside
  // {core-tessera-ir, nvidia-backend, rocm-backend} is linked into a lean
  // build, so reaching here with an unregistered Apple/Solvers/TPP/Neighbors
  // library is no longer possible.
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendPasses();
  tessera::registerTesseraNVIDIALegacyBackendPasses();
#endif
#ifdef TESSERA_HAVE_ROCM_BACKEND
  mlir::tessera_rocm::registerTesseraROCMBackendPasses();
#endif

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                  mlir::ROCDL::ROCDLDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect,
                  tessera::tile::TesseraTileDialect>();
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendDialects(registry);
#endif
#ifdef TESSERA_HAVE_ROCM_BACKEND
  mlir::tessera_rocm::registerTesseraROCMBackendDialects(registry);
#endif

  return failed(mlir::MlirOptMain(argc, argv, tesseraOptBuildInfo(), registry));
#else
#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  tessera::registerTesseraPasses();
  // Upstream canonicalize / cse so Tessera per-op folders + canonicalizers
  // (Phase 1: identity cast, transpose-of-transpose, …) are inspectable in lit.
  mlir::registerTransformsPasses();
  tessera::diagnostics::registerShapeInferencePass();
  tessera::diagnostics::registerErrorReporterPass();
  // Needs `tessera-to-linalg`, so it belongs with the core IR registration.
  registerEmitPipelines();
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
  tessera::registerTesseraNVIDIALegacyBackendPasses();
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
                  mlir::math::MathDialect,
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

  // LLVM 23's GPU lowering discovers conversion and target interfaces through
  // the dialect registry.  Keep this aligned with upstream mlir-opt rather
  // than maintaining a fragile hand-selected subset: NVVM/ROCDL target
  // attributes and every loaded dialect's ConvertToLLVM interface must be
  // registered before MlirOptMain creates its context.
  mlir::registerAllExtensions(registry);

#ifdef TESSERA_HAVE_CORE_TESSERA_IR
  tessera::registerTesseraDialects(registry);
  // Sprint 9 — Tile IR dialect (value-lane lowering spine). Registering it lets
  // the Apple `-full` pipelines run without --allow-unregistered-dialect.
  tessera::tile::registerTileDialect(registry);
#endif

#ifdef TESSERA_HAVE_FA4_ATTN
  // Sprint V7 (2026-05-22) — FA-4 attention dialect.  Unblocks the
  // three `tessera_attn.scaled_dot_product` lit fixtures
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

#ifdef TESSERA_HAVE_X86_TARGET_IR
  // W0.10 — `tessera_x86.*` ops and the `!tessera_x86.tile` type, so the x86
  // Target IR is parseable and verifiable by lit like every other backend.
  mlir::tessera_x86::registerTesseraX86Dialect(registry);
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
#endif
#endif
#ifdef TESSERA_HAVE_NVIDIA_BACKEND
  tessera::registerTesseraNVIDIABackendDialects(registry);
#endif

  return failed(mlir::MlirOptMain(argc, argv, tesseraOptBuildInfo(), registry));
#endif
}
