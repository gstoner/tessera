#include "TesseraROCM/Passes.h"
#include "Tessera/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "TesseraROCMDialect.h.inc"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;

namespace mlir::tessera_rocm {
std::unique_ptr<Pass> createLowerTesseraToROCDLImpl();
std::unique_ptr<Pass> createLowerTileToROCMImpl();

std::unique_ptr<Pass> createLowerTileToROCMPass() {
  return createLowerTileToROCMImpl();
}

std::unique_ptr<Pass> createLowerTesseraTargetToROCDLPass() {
  return createLowerTesseraToROCDLImpl();
}

namespace {

constexpr StringLiteral kExecutablePipeline = "tessera-rocm-executable";

struct ROCMExecutablePipelineOptions
    : public PassPipelineOptions<ROCMExecutablePipelineOptions> {
  Option<std::string> family{
      *this, "family",
      llvm::cl::desc("physical family plugin: matmul, softmax, reduction, "
                     "paged_kv, attention, attention_backward, moe_dispatch, "
                     "or an explicit registered family plugin"),
      llvm::cl::init("matmul")};
  Option<std::string> input{
      *this, "input",
      llvm::cl::desc("producer level: graph, tile, or directive"),
      llvm::cl::init("tile")};
  Option<std::string> output{
      *this, "output",
      llvm::cl::desc("terminal artifact level: target or binary"),
      llvm::cl::init("binary")};
  Option<std::string> arch{
      *this, "arch", llvm::cl::desc("exact ROCm architecture"),
      llvm::cl::init("gfx1151")};
  Option<std::string> staging{
      *this, "staging",
      llvm::cl::desc("matmul staging policy: register or lds"),
      llvm::cl::init("register")};
  Option<int> tileQ{*this, "tile-q", llvm::cl::desc("attention Q tile"),
                    llvm::cl::init(64)};
  Option<int> tileKv{*this, "tile-kv", llvm::cl::desc("attention KV tile"),
                     llvm::cl::init(64)};
};

struct DeclareROCMPipelineContractPass
    : PassWrapper<DeclareROCMPipelineContractPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeclareROCMPipelineContractPass)

  DeclareROCMPipelineContractPass() = default;
  DeclareROCMPipelineContractPass(StringRef family, StringRef input,
                                  StringRef output, StringRef arch)
      : family(family.str()), input(input.str()), output(output.str()),
        arch(arch.str()) {}
  DeclareROCMPipelineContractPass(const DeclareROCMPipelineContractPass &other)
      : PassWrapper(other), family(other.family), input(other.input),
        output(other.output), arch(other.arch) {}

  StringRef getArgument() const final { return "declare-rocm-pipeline-contract"; }
  StringRef getDescription() const final {
    return "Declare the Tile producer, Target IR consumer, and ROCm codegen plugin";
  }

  void runOnOperation() override {
    static constexpr StringLiteral families[] = {
        "algebra_clifford", "attention_mla_decode", "draft_dspark",
        "ebm_affine_langevin", "ebm_decode_init", "ebm_ebt_tiny",
        "ebm_energy_quadratic", "ebm_langevin", "ebm_partition",
        "fused_silu_mul", "indexing_gather", "indexing_scatter",
        "loss_binary", "loss_pointwise", "loss_policy", "matmul_batched_f32",
        "matmul_f32", "normalization", "optimizer", "ordering_sort",
        "position_alibi", "position_rope", "quant_dequant_gemm", "quant_fp",
        "quant_int4_pack", "reduction_arg", "rng_philox", "scan",
        "spectral_dft", "spectral_backward", "matmul", "softmax",
        "depth_attention", "reduction", "paged_kv", "attention",
        "attention_backward", "moe_dispatch", "scalar_activation",
        "scalar_binary", "scalar_bitwise", "scalar_compare", "scalar_logical",
        "scalar_predicate", "scalar_unary", "scalar_where", "sequence_deltanet",
        "sequence_linear_attention", "sequence_recurrent_cell",
        "sequence_selective_ssm", "sequence_selective_ssm_backward",
        "solver_cholesky", "solver_lu", "solver_qr", "solver_svd",
        "solver_triangular_solve", "solver_ift", "sparse_block_attention", "sparse_block_topk",
        "sparse_sddmm", "sparse_spmm"};
    if (llvm::find(families, family) == std::end(families)) {
      getOperation().emitError("unknown ROCm family plugin '") << family << "'";
      return signalPassFailure();
    }
    if (input != "graph" && input != "tile" && input != "directive") {
      getOperation().emitError("ROCm pipeline input must be graph, tile, or directive");
      return signalPassFailure();
    }
    if (output != "target" && output != "binary") {
      getOperation().emitError("ROCm pipeline output must be target or binary");
      return signalPassFailure();
    }
    // gfx1200/gfx1250 remain fail-closed until their family plugins have exact
    // device evidence. This prevents a gfx1151 physical schedule being
    // relabelled at the serialization boundary.
    if (arch != "gfx1151") {
      getOperation().emitError(kExecutablePipeline)
          << ": architecture '" << arch
          << "' has no promoted family-plugin profile";
      return signalPassFailure();
    }
    Builder b(&getContext());
    ModuleOp module = getOperation();
    module->setAttr("tessera.pipeline.schema",
                    b.getStringAttr("tessera.executable_pipeline.v1"));
    module->setAttr("tessera.pipeline.family", b.getStringAttr(family));
    module->setAttr("tessera.pipeline.tile_producer",
                    b.getStringAttr(input == "graph" ? "graph_to_tile"
                                                     : "content_addressed_tile"));
    module->setAttr("tessera.pipeline.target_ir_consumer",
                    b.getStringAttr("tessera_rocm"));
    module->setAttr("tessera.pipeline.backend_codegen",
                    b.getStringAttr("rocdl_hsaco"));
    module->setAttr("tessera.pipeline.arch", b.getStringAttr(arch));
    module->setAttr("tessera.pipeline.output", b.getStringAttr(output));
  }

  std::string family, input, output, arch;
};

struct VerifyROCMExecutablePass
    : PassWrapper<VerifyROCMExecutablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyROCMExecutablePass)

  StringRef getArgument() const final { return "verify-rocm-executable"; }
  StringRef getDescription() const final {
    return "Reject target-only ROCm operations, contract markers, and undef "
           "results at the binary boundary";
  }

  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name.starts_with("tessera_rocm.") || name.starts_with("tile.")) {
        op->emitError("operation survived the strict ROCm executable boundary");
        invalid = true;
      }
      if (name == "llvm.mlir.undef") {
        op->emitError("undef is forbidden at the strict ROCm executable boundary");
        invalid = true;
      }
      if (auto symbol = op->getAttrOfType<StringAttr>("sym_name");
          symbol && symbol.getValue().contains(".contract")) {
        op->emitError("contract-marker symbol is forbidden in an executable artifact");
        invalid = true;
      }
    });
    if (invalid)
      signalPassFailure();
  }
};

static std::unique_ptr<Pass> configuredPass(std::unique_ptr<Pass> pass,
                                            const Twine &options) {
  std::string text = options.str();
  std::string detail;
  if (failed(pass->initializeOptions(text, [&](const Twine &message) {
        detail = message.str();
        return failure();
      })))
    llvm::report_fatal_error(Twine("invalid options for canonical ROCm pass: ") +
                             text + ": " + detail);
  return pass;
}

static void addFamilyGenerator(OpPassManager &pm, StringRef family,
                               bool viaTile, StringRef staging) {
  if (family == "algebra_clifford") {
    pm.addPass(createGenerateROCMCliffordKernelPass());
  } else if (family == "attention_mla_decode") {
    pm.addPass(createGenerateROCMMLAAbsorbDecodeKernelPass());
  } else if (family == "draft_dspark") {
    pm.addPass(createGenerateROCMDSparkDraftBlockKernelPass());
  } else if (family == "ebm_affine_langevin") {
    pm.addPass(createGenerateROCMEbmAffineLangevinKernelPass());
  } else if (family == "ebm_decode_init") {
    pm.addPass(createGenerateROCMEbmDecodeInitKernelPass());
  } else if (family == "ebm_ebt_tiny") {
    pm.addPass(createGenerateROCMEbmEbtTinyKernelPass());
  } else if (family == "ebm_energy_quadratic") {
    pm.addPass(createGenerateROCMEbmEnergyQuadraticKernelPass());
  } else if (family == "ebm_langevin") {
    pm.addPass(createGenerateROCMEbmLangevinKernelPass());
  } else if (family == "ebm_partition") {
    pm.addPass(createGenerateROCMEbmPartitionKernelPass());
  } else if (family == "fused_silu_mul") {
    pm.addPass(createGenerateROCMSiluMulKernelPass());
  } else if (family == "indexing_gather") {
    pm.addPass(createGenerateROCMGatherKernelPass());
  } else if (family == "indexing_scatter") {
    pm.addPass(createGenerateROCMScatterKernelPass());
  } else if (family == "loss_binary") {
    pm.addPass(createGenerateROCMBinaryLossKernelPass());
  } else if (family == "loss_pointwise") {
    pm.addPass(createGenerateROCMPointwiseLossKernelPass());
  } else if (family == "loss_policy") {
    pm.addPass(createGenerateROCMPolicyLossKernelPass());
  } else if (family == "matmul_batched_f32") {
    pm.addPass(createGenerateROCMBatchedGemmF32KernelPass());
  } else if (family == "matmul_f32") {
    pm.addPass(createGenerateROCMGemmF32KernelPass());
  } else if (family == "normalization") {
    pm.addPass(createGenerateROCMNormKernelPass());
  } else if (family == "optimizer") {
    pm.addPass(createGenerateROCMOptimizerKernelPass());
  } else if (family == "ordering_sort") {
    pm.addPass(createGenerateROCMSortKernelPass());
  } else if (family == "position_alibi") {
    pm.addPass(createGenerateROCMAlibiKernelPass());
  } else if (family == "position_rope") {
    pm.addPass(createGenerateROCMRopeKernelPass());
  } else if (family == "quant_dequant_gemm") {
    pm.addPass(createGenerateROCMDequantGemmKernelPass());
  } else if (family == "quant_fp") {
    pm.addPass(createGenerateROCMFpQuantKernelPass());
  } else if (family == "quant_int4_pack") {
    pm.addPass(createGenerateROCMInt4PackKernelPass());
  } else if (family == "reduction_arg") {
    pm.addPass(createGenerateROCMArgReduceKernelPass());
  } else if (family == "rng_philox") {
    pm.addPass(createGenerateROCMPhiloxKernelPass());
  } else if (family == "scan") {
    pm.addPass(createGenerateROCMScanKernelPass());
  } else if (family == "spectral_dft") {
    pm.addPass(createGenerateROCMDftKernelPass());
  } else if (family == "spectral_backward") {
    pm.addPass(createGenerateROCMSpectralBackwardKernelPass());
  } else if (family == "es_low_rank_correction") {
    pm.addPass(createGenerateROCMESLowRankKernelPass());
  } else if (family == "matmul") {
    pm.addPass(configuredPass(createGenerateWMMAGemmKernelPass(),
                              Twine("via-tile=") + (viaTile ? "true" : "false") +
                                  " canonical-staging=" + staging));
  } else if (family == "softmax") {
    pm.addPass(createGenerateROCMSoftmaxKernelPass());
  } else if (family == "depth_attention") {
    pm.addPass(createGenerateROCMDepthAttentionKernelPass());
  } else if (family == "reduction") {
    pm.addPass(createGenerateROCMReduceKernelPass());
  } else if (family == "paged_kv") {
    pm.addPass(createGenerateROCMPagedKVReadKernelPass());
  } else if (family == "attention") {
    pm.addPass(createGenerateWMMAFlashAttnKernelPass());
  } else if (family == "attention_backward") {
    pm.addPass(createGenerateWMMAFlashAttnKernelPass());
    pm.addPass(createGenerateWMMAFlashAttnBwdKernelPass());
  } else if (family == "moe_dispatch") {
    pm.addPass(createGenerateROCMMoeKernelPass());
  } else if (family == "scalar_activation") {
    pm.addPass(createGenerateROCMActivationKernelPass());
  } else if (family == "scalar_binary") {
    pm.addPass(createGenerateROCMBinaryKernelPass());
  } else if (family == "scalar_bitwise") {
    pm.addPass(createGenerateROCMBitwiseKernelPass());
  } else if (family == "scalar_compare") {
    pm.addPass(createGenerateROCMCompareKernelPass());
  } else if (family == "scalar_logical") {
    pm.addPass(createGenerateROCMLogicalKernelPass());
  } else if (family == "scalar_predicate") {
    pm.addPass(createGenerateROCMPredicateKernelPass());
  } else if (family == "scalar_unary") {
    pm.addPass(createGenerateROCMUnaryKernelPass());
  } else if (family == "scalar_where") {
    pm.addPass(createGenerateROCMWhereKernelPass());
  } else if (family == "sequence_deltanet") {
    pm.addPass(createGenerateROCMDeltaNetKernelPass());
  } else if (family == "sequence_linear_attention") {
    pm.addPass(createGenerateWMMALinearAttnKernelPass());
  } else if (family == "sequence_recurrent_cell") {
    pm.addPass(createGenerateROCMRecurrentCellKernelPass());
  } else if (family == "sequence_selective_ssm") {
    pm.addPass(createGenerateROCMSelectiveSsmKernelPass());
  } else if (family == "sequence_selective_ssm_backward") {
    pm.addPass(createGenerateROCMSelectiveSsmBwdKernelPass());
  } else if (family == "solver_cholesky") {
    pm.addPass(createGenerateROCMCholeskyKernelPass());
  } else if (family == "solver_lu") {
    pm.addPass(createGenerateROCMLuKernelPass());
  } else if (family == "solver_qr") {
    pm.addPass(createGenerateROCMQrKernelPass());
  } else if (family == "solver_svd") {
    pm.addPass(createGenerateROCMSvdKernelPass());
  } else if (family == "solver_triangular_solve") {
    pm.addPass(createGenerateROCMTriSolveKernelPass());
  } else if (family == "solver_ift") {
    pm.addPass(createGenerateROCMSolverIFTKernelPass());
  } else if (family == "sparse_block_attention") {
    pm.addPass(createGenerateROCMBlockSparseAttnKernelPass());
  } else if (family == "sparse_block_topk") {
    pm.addPass(createGenerateROCMBlockSparseTopKKernelPass());
  } else if (family == "sparse_sddmm") {
    pm.addPass(createGenerateROCMSddmmKernelPass());
  } else if (family == "sparse_spmm") {
    pm.addPass(createGenerateROCMSpmmKernelPass());
  }
}

static void buildROCMExecutablePipeline(
    OpPassManager &pm, const ROCMExecutablePipelineOptions &opts) {
  StringRef family = opts.family;
  StringRef input = opts.input;
  StringRef output = opts.output;
  StringRef arch = opts.arch;
  pm.addPass(std::make_unique<DeclareROCMPipelineContractPass>(
      family, input, output, arch));

  if (input == "graph") {
    if (family == "matmul")
      pm.addPass(::tessera::createTilingPass());
    auto tileLowering = ::tessera::createTileIRLoweringPass();
    pm.addPass(configuredPass(
        std::move(tileLowering),
        Twine("sm=90 tile-q=") + Twine(opts.tileQ) +
            " tile-kv=" + Twine(opts.tileKv)));
  }

  // Directive and scheduled-matmul inputs must first materialize the Tile
  // producer. Every other family already arrives as canonical Tile IR and its
  // plugin runs after the Target-IR consumer.
  bool matmulPlugin = family == "matmul";
  if (matmulPlugin && input != "graph" && output == "binary")
    addFamilyGenerator(pm, family, input == "tile", opts.staging);

  pm.addPass(createROCMWaveLdsPipelinePass());
  pm.addPass(createROCMWaveLdsLegalityPass());
  if (matmulPlugin && input == "graph" && output == "binary")
    addFamilyGenerator(pm, family, false, opts.staging);
  pm.addPass(configuredPass(createLowerTileToROCMPass(),
                            Twine("arch=") + arch));

  // Target output is the typed architecture boundary.  It must retain the
  // tessera_rocm.* directive selected by TileToROCM; running the family
  // generator here would silently return backend GPU IR while labelling it
  // Target IR.  Binary output alone crosses into architecture codegen.
  if (output == "target")
    return;

  if (!matmulPlugin)
    addFamilyGenerator(pm, family, false, opts.staging);

  pm.addPass(createLowerKernelABIPass());
  // This ordering is architectural: executable global->LDS copies and their
  // targeted waits must materialize before the catch-all Target IR conversion.
  pm.addPass(createLowerROCMAsyncCopyToLoopPass());
  pm.addPass(createLowerTesseraTargetToROCDLPass());
  pm.addPass(std::make_unique<VerifyROCMExecutablePass>());
  pm.addNestedPass<gpu::GPUModuleOp>(createSCFToControlFlowPass());
  ConvertGpuOpsToROCDLOpsOptions conversion;
  conversion.chipset = arch.str();
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToROCDLOps(conversion));
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createROCMDynamicLDSPass());
  GpuROCDLAttachTargetOptions target;
  target.chip = arch.str();
  target.wave64Flag = false;
  pm.addPass(createGpuROCDLAttachTarget(target));
  pm.addPass(createGpuModuleToBinaryPass());
}

} // namespace

void buildTesseraROCMBackendPipeline(OpPassManager &pm) {
  // CORE-COMPILER-2: ROCm is the first target where dtype legalization is a
  // default rather than an opt-in annotation. Its WMMA generator consumes the
  // concrete storage-pack descriptor, so the full compute -> storage -> consume
  // chain is executable and cannot leave an inert packed marker behind.
  pm.addPass(::tessera::createComputeLegalizePass());
  pm.addPass(::tessera::createStorageLegalizePass("rocm_gfx11"));
  pm.addPass(::tessera::createStoragePackConsumePass());
  // Consume launch-level tile.matmul_kernel contracts before the lower-level
  // Tile pass materializes individual typed fragments. The generator is a no-op
  // when neither a portable kernel nor a wmma_gemm directive is present.
  pm.addPass(createGenerateWMMAGemmKernelPass());
  // Make Tile shared-memory planning executable before architecture lowering:
  // disjoint buffers alias one address-space-3 LDS arena through byte-offset
  // memref views, which the ordinary ROCDL conversion consumes directly.
  pm.addPass(::tessera::createTileBufferReusePass());
  pm.addPass(::tessera::createTileBufferArenaPass());
  pm.addPass(createROCMWaveLdsPipelinePass());
  pm.addPass(createROCMWaveLdsLegalityPass());
  pm.addPass(createLowerTileToROCMPass());
  // Static head/value buckets from the canonical attention carrier select the
  // physical gfx1151 WMMA schedule here. Runtime sequence dimensions remain
  // launch operands; unsupported semantic combinations retain the direct
  // recurrence emitted by lower-tile-to-rocm.
  pm.addPass(createGenerateWMMAFlashAttnKernelPass());
  // ROCM-E2E-1/-2 wire only families with typed producers and descriptor
  // consumers. Unrelated standalone generators remain outside this pipeline.
  pm.addPass(createGenerateROCMSoftmaxKernelPass());
  pm.addPass(createGenerateROCMDepthAttentionKernelPass());
  pm.addPass(createGenerateROCMReduceKernelPass());
  pm.addPass(createGenerateROCMPagedKVReadKernelPass());
  pm.addPass(createLowerKernelABIPass());
  // These compatibility aliases are Target-IR inspection pipelines. They stop
  // before executable ROCDL conversion; binary production is owned solely by
  // tessera-rocm-executable{output=binary}.
}

void registerTesseraROCMPasses() {
  registerPass([]() { return createROCMWaveLdsPipelinePass(); });
  registerPass([]() { return createROCMWaveLdsLegalityPass(); });
  registerPass([]() { return createROCMDynamicLDSPass(); });
  registerPass([]() { return createLowerTileToROCMPass(); });
  registerPass([]() { return createLowerKernelABIPass(); });
  registerPass([]() { return createLowerTesseraTargetToROCDLPass(); });
  registerPass([]() { return createGenerateWMMAGemmKernelPass(); });
  registerPass([]() { return createGenerateWMMAFlashAttnKernelPass(); });
  registerPass([]() { return createGenerateWMMAFlashAttnBwdKernelPass(); });
  registerPass([]() { return createGenerateWMMALinearAttnKernelPass(); });
  registerPass([]() { return createGenerateROCMActivationKernelPass(); });
  registerPass([]() { return createGenerateROCMSiluMulKernelPass(); });
  registerPass([]() { return createGenerateROCMPointwiseLossKernelPass(); });
  registerPass([]() { return createGenerateROCMBinaryLossKernelPass(); });
  registerPass([]() { return createGenerateROCMPolicyLossKernelPass(); });
  registerPass([]() { return createGenerateROCMFpQuantKernelPass(); });
  registerPass([]() { return createGenerateROCMInt4PackKernelPass(); });
  registerPass([]() { return createGenerateROCMDequantGemmKernelPass(); });
  registerPass([]() { return createGenerateROCMDftKernelPass(); });
  registerPass([]() { return createGenerateROCMSpmmKernelPass(); });
  registerPass([]() { return createGenerateROCMSddmmKernelPass(); });
  registerPass([]() { return createGenerateROCMSelectiveSsmKernelPass(); });
  registerPass([]() { return createGenerateROCMSelectiveSsmBwdKernelPass(); });
  registerPass([]() { return createGenerateROCMCholeskyKernelPass(); });
  registerPass([]() { return createGenerateROCMTriSolveKernelPass(); });
  registerPass([]() { return createGenerateROCMLuKernelPass(); });
  registerPass([]() { return createGenerateROCMQrKernelPass(); });
  registerPass([]() { return createGenerateROCMSvdKernelPass(); });
  registerPass([]() { return createGenerateROCMOptimizerKernelPass(); });
  registerPass([]() { return createGenerateROCMPredicateKernelPass(); });
  registerPass([]() { return createGenerateROCMMoeKernelPass(); });
  registerPass([]() { return createGenerateROCMGemmF32KernelPass(); });
  registerPass([]() { return createGenerateROCMRecurrentCellKernelPass(); });
  registerPass([]() { return createGenerateROCMBatchedGemmF32KernelPass(); });
  registerPass([]() { return createGenerateROCMAlibiKernelPass(); });
  registerPass([]() { return createGenerateROCMDeltaNetKernelPass(); });
  registerPass([]() { return createGenerateROCMRopeKernelPass(); });
  registerPass([]() { return createGenerateROCMDSparkDraftBlockKernelPass(); });
  registerPass([]() { return createGenerateROCMMLAAbsorbDecodeKernelPass(); });
  registerPass([]() { return createGenerateROCMBlockSparseAttnKernelPass(); });
  registerPass([]() { return createGenerateROCMBlockSparseTopKKernelPass(); });
  registerPass([]() { return createGenerateROCMSoftmaxKernelPass(); });
  registerPass([]() { return createGenerateROCMDepthAttentionKernelPass(); });
  registerPass([]() { return createGenerateROCMNormKernelPass(); });
  registerPass([]() { return createGenerateROCMReduceKernelPass(); });
  registerPass([]() { return createGenerateROCMPagedKVReadKernelPass(); });
  registerPass([]() { return createGenerateROCMArgReduceKernelPass(); });
  registerPass([]() { return createGenerateROCMScanKernelPass(); });
  registerPass([]() { return createGenerateROCMUnaryKernelPass(); });
  registerPass([]() { return createGenerateROCMSolverIFTKernelPass(); });
  registerPass([]() { return createGenerateROCMSpectralBackwardKernelPass(); });
  registerPass([]() { return createGenerateROCMESLowRankKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForGemvKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForNormKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForWmmaKernelPass(); });
  registerPass(
      []() { return createGenerateROCMControlForWmmaTileKernelPass(); });
  registerPass([]() { return createGenerateROCMControlScanKernelPass(); });
  registerPass([]() { return createGenerateROCMControlIfNormKernelPass(); });
  registerPass([]() { return createGenerateROCMControlScanGemvKernelPass(); });
  registerPass([]() { return createGenerateROCMControlScanRnnKernelPass(); });
  registerPass([]() { return createGenerateROCMControlWhileGemvKernelPass(); });
  registerPass([]() { return createGenerateROCMSpecAcceptKernelPass(); });
  registerPass([]() { return createGenerateROCMSpecAcceptSampleKernelPass(); });
  registerPass([]() { return createGenerateROCMSpecAcceptTreeSampleKernelPass(); });
  registerPass([]() { return createGenerateROCMBinaryKernelPass(); });
  registerPass([]() { return createGenerateROCMCompareKernelPass(); });
  registerPass([]() { return createGenerateROCMLogicalKernelPass(); });
  registerPass([]() { return createGenerateROCMBitwiseKernelPass(); });
  registerPass([]() { return createGenerateROCMWhereKernelPass(); });
  registerPass([]() { return createGenerateROCMPhiloxKernelPass(); });
  registerPass([]() { return createGenerateROCMGatherKernelPass(); });
  registerPass([]() { return createGenerateROCMSortKernelPass(); });
  registerPass([]() { return createGenerateROCMScatterKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmLangevinKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmAffineLangevinKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmPartitionKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmDecodeInitKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmEnergyQuadraticKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmEbtTinyKernelPass(); });
  registerPass([]() { return createGenerateROCMCliffordKernelPass(); });
  registerPass([]() { return createLowerROCMAsyncCopyToLoopPass(); });
  registerPass([]() { return std::make_unique<VerifyROCMExecutablePass>(); });
  PassPipelineRegistration<> pipeline(
      "tessera-rocm-backend",
      "Produce typed Tessera ROCm Target IR for inspection",
      [](OpPassManager &pm) { buildTesseraROCMBackendPipeline(pm); });
  PassPipelineRegistration<> canonicalPipeline(
      "tessera-lower-to-rocm",
      "Compatibility alias producing typed Tessera ROCm Target IR",
      [](OpPassManager &pm) { buildTesseraROCMBackendPipeline(pm); });
  PassPipelineRegistration<ROCMExecutablePipelineOptions> executablePipeline(
      "tessera-rocm-executable",
      "Typed family-plugin ROCm pipeline from Graph/Tile to Target IR or HSACO",
      buildROCMExecutablePipeline);
}

void registerTesseraROCMDialects(DialectRegistry &registry) {
  registry.insert<TesseraROCMDialect>();
}

void registerTesseraROCMBackendPasses() { registerTesseraROCMPasses(); }

void registerTesseraROCMBackendDialects(DialectRegistry &registry) {
  registerTesseraROCMDialects(registry);
}
}
