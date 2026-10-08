//===- ROCMKernelIdentity.cpp - native physical image identity ------------===//
//
// Shape-bound host scaffolding is not part of the serialized GPU kernel.
// Project only the audited attribute-only directive; preserve all physical
// attributes and module attributes. Schedule ancestry remains launch-owned.
#include "TesseraROCM/Passes.h"
#include "ROCMFoldedW4A8Contract.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace {
struct ProjectROCMKernelIdentityPass
    : PassWrapper<ProjectROCMKernelIdentityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProjectROCMKernelIdentityPass)
  ProjectROCMKernelIdentityPass() = default;
  ProjectROCMKernelIdentityPass(const ProjectROCMKernelIdentityPass &other)
      : PassWrapper(other) {}
  Option<std::string> family{*this, "family",
                            llvm::cl::desc("Audited native package family"),
                            llvm::cl::init("")};
  Option<bool> runtimeK{*this, "runtime-k",
                        llvm::cl::desc("Project LDS W8A8 or folded K to its checked runtime group loop"),
                        llvm::cl::init(false)};
  Option<bool> primalProfile{*this, "primal-profile",
      llvm::cl::desc("Use the measured gfx1201 scaled-primal image policy"),
      llvm::cl::init(false)};
  StringRef getArgument() const final {
    return "tessera-rocm-project-kernel-identity";
  }
  StringRef getDescription() const final {
    return "Project an audited ROCm Target directive onto native image identity";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    StringRef directive = llvm::StringSwitch<StringRef>(family.getValue())
      .Case("scalar_unary", "tessera_rocm.unary")
      .Case("scalar_binary", "tessera_rocm.binary")
      .Case("scan", "tessera_rocm.scan")
      .Case("softmax", "tessera_rocm.softmax")
      .Case("reduction", "tessera_rocm.reduce")
      .Case("normalization", "tessera_rocm.norm")
      .Case("paged_kv", "tessera_rocm.paged_kv_read")
      .Case("moe_dispatch", "tessera_rocm.moe_dispatch")
      .Case("attention", "tessera_rocm.flash_attn")
      .Case("matmul", "tessera_rocm.wmma_gemm")
      .Case("scaled_matmul", "tessera_rocm.scaled_wmma_gemm")
      .Case("scaled_matmul_lds", "tessera_rocm.scaled_wmma_gemm")
      .Case("folded_matmul", "tessera_rocm.scaled_wmma_gemm").Default("");
    if (runtimeK && family != "scaled_matmul_lds" && family != "folded_matmul") {
      module.emitError("ROCM_FP8_BLOCKSCALE_CONTRACT: runtime-K image projection requires the LDS W8A8 or folded family");
      return signalPassFailure();
    }
    if (directive.empty()) {
      module.emitError("kernel identity requires an audited ROCm family");
      return signalPassFailure();
    }
    bool wrapper = family == "paged_kv" || family == "moe_dispatch";
    Operation *kernel = nullptr;
    unsigned functions = 0, returns = 0;
    bool invalid = false;
    module.walk([&](Operation *op) {
      if (op == module.getOperation()) return;
      StringRef name = op->getName().getStringRef();
      if (name.starts_with("tessera_rocm.")) {
        if (kernel || name != directive || op->getNumOperands() ||
            op->getNumResults() || op->getNumRegions() ||
            !op->getAttrOfType<StringAttr>("name")) {
          op->emitError("kernel identity requires exactly one named attribute-only directive for the requested family");
          invalid = true;
        }
        kernel = op;
        return;
      }
      if (wrapper && name == "llvm.func") { ++functions; return; }
      if (wrapper && name == "llvm.return") { ++returns; return; }
      if (name == "tensor.dim") {
        // Runtime M/N/K extraction is host ABI scaffolding, never kernel code.
        // Admit only dimensions of an entry argument with a constant axis.
        auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
        Operation *axis = op->getOperand(1).getDefiningOp();
        if (arg && arg.getOwner()->getParentOp()->getName().getStringRef() == "func.func" &&
            axis && axis->getName().getStringRef() == "arith.constant" &&
            axis->getAttrOfType<IntegerAttr>("value"))
          return;
        op->emitError("kernel identity requires entry-owned runtime dimensions");
        invalid = true;
        return;
      }
      bool scaffold = llvm::StringSwitch<bool>(name)
        .Cases({"func.func", "func.return", "arith.constant"}, true)
        .Cases({"arith.index_cast", "bufferization.to_buffer"}, true)
        .Cases({"bufferization.to_tensor", "memref.alloc"}, true)
        .Cases({"memref.extract_aligned_pointer_as_index", "llvm.inttoptr"}, true)
        .Default(false);
      if (!scaffold) {
        op->emitError("kernel identity cannot drop an unaudited Target operation");
        invalid = true;
      }
    });
    if (invalid) return signalPassFailure();
    if (!kernel || (wrapper && (functions != 1 || returns != 1))) {
      module.emitError("kernel identity requires one directive and the audited wrapper structure");
      return signalPassFailure();
    }
    if (family == "matmul" || family == "scaled_matmul" || family == "scaled_matmul_lds" || family == "folded_matmul") {
      auto hash = kernel->getAttrOfType<StringAttr>("tessera.schedule_hash");
      if (!hash || hash.getValue().size() != 64 ||
          !llvm::all_of(hash.getValue(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
          })) {
        kernel->emitError("matmul kernel identity requires a verified Schedule hash");
        return signalPassFailure();
      }
    }
    if (family == "folded_matmul" &&
        !tessera_rocm::isValidFoldedW4A8Target(kernel, /*allowRuntimeMN=*/false)) {
      kernel->emitError("ROCM_FOLDED_NATIVE_CONTRACT: folded image projection requires the verified static gfx1201 shape, policy, ABI and physical schedule");
      return signalPassFailure();
    }
    if (family == "scaled_matmul" || family == "scaled_matmul_lds") {
      auto contract = kernel->getAttrOfType<StringAttr>("physical_contract");
      auto staging = kernel->getAttrOfType<StringAttr>("staging");
      auto waves = kernel->getAttrOfType<IntegerAttr>("warps");
      bool shapeValid = !kernel->hasAttr("runtime_shape") &&
          !kernel->hasAttr("runtime_mn") && !kernel->hasAttr("runtime_k") &&
          !kernel->hasAttr("whole_m") &&
          !kernel->hasAttr("whole_n");
      for (StringRef axis : {"m", "n", "k"}) {
        auto extent = kernel->getAttrOfType<IntegerAttr>(axis);
        shapeValid &= extent && extent.getInt() > 0;
      }
      auto text = [&](StringRef key) -> StringRef {
        auto value = kernel->getAttrOfType<StringAttr>(key);
        return value ? value.getValue() : StringRef();
      };
      auto integer = [&](StringRef key) -> int64_t {
        auto value = kernel->getAttrOfType<IntegerAttr>(key);
        return value ? value.getInt() : 0;
      };
      auto policy = kernel->getAttrOfType<DictionaryAttr>("numeric_policy");
      auto policyText = [&](StringRef key) -> StringRef {
        auto value = policy ? policy.getAs<StringAttr>(key) : StringAttr();
        return value ? value.getValue() : StringRef();
      };
      bool mxfp8 = text("physical_contract") == "rocm_mxfp8_e4m3_e8m0_k32_v1" ||
                   text("physical_contract") == "rocm_mxfp8_e4m3_e8m0_k32_nk_v1";
      bool nk = text("physical_contract") == "rocm_fp8_w8a8_blockscale_nk_v1" ||
                text("physical_contract") == "rocm_mxfp8_e4m3_e8m0_k32_nk_v1";
      bool bf16 = text("output") == "bf16";
      std::string packageABI =
          (llvm::Twine("tessera.rocm.fp8_w8a8_blockscale.") +
           (nk ? "a_bnk_sa_sb_o_m_n_k." : "a_b_sa_sb_o_m_n_k.") +
           (bf16 ? "e4m3_e4m3_f32_bf16.wmma_exact.v1"
                 : "e4m3_e4m3_f32_f32.wmma_exact.v1")).str();
      if (mxfp8)
        packageABI = (llvm::Twine("tessera.rocm.mxfp8_e4m3_e8m0_k32.") +
            (nk ? "a_bnk_sa_sb_o_m_n_k." : "a_b_sa_sb_o_m_n_k.") +
            (bf16 ? "bf16.wide_scale.v1" : "f32.wide_scale.v1")).str();
      int64_t scaleK = integer("scale_k"), macroK = integer("macro_k");
      int64_t blockM = integer("block_m"), blockN = integer("block_n");
      auto raster = kernel->getAttrOfType<StringAttr>("schedule_raster_order");
      auto rasterGroup = kernel->getAttrOfType<IntegerAttr>("schedule_raster_group");
      StringRef order = raster ? raster.getValue() : StringRef("row_major");
      bool rasterValid =
          (order == "row_major" || order == "column_major" ||
           order == "grouped_m" || order == "grouped_n") &&
          (!rasterGroup || rasterGroup.getInt() > 0);
      bool physicalValid = rasterValid &&
          text("abi") == (nk ? "a_bnk_lhs_scale_rhs_scale_d_m_n_k"
                            : "a_b_lhs_scale_rhs_scale_d_m_n_k") &&
          text("package_abi") == packageABI &&
          text("scale_format") == (mxfp8 ? "e8m0" : "fp32") &&
          text("partial_combine") == "scale_outer_product_then_add" &&
          text("k_step_schedule") == "isolated_scale_group" &&
          (text("output") == "f32" || bf16) &&
          policyText("storage") == "e4m3" && policyText("accum") == "f32" &&
          policyText("execution_mode") == "exact_per_block" &&
          integer("instruction_k") == 16 && scaleK > 0 && scaleK % 16 == 0 &&
          (integer("k") % scaleK == 0 ||
           (text("batching") == "broadcast" && text("staging") == "global" &&
            integer("k") > 0 && integer("k") <= INT64_MAX-scaleK+1)) &&
          macroK > 0 && macroK % scaleK == 0 &&
          integer("scale_n") > 0 && blockM > 0 && blockN > 0 &&
          blockM % 16 == 0 && blockN % 16 == 0 &&
          (integer("pipeline_depth") == 1 || integer("pipeline_depth") == 2);
      const bool lds = family == "scaled_matmul_lds";
      bool mxfp8Profile = scaleK == 32 && integer("scale_n") == 1 &&
          integer("pipeline_depth") == 1 && waves &&
          (lds ? ((macroK == 32 || macroK == 64) && nk && blockM == 128 &&
                  (blockN == 64 || blockN == 128) && waves.getInt() == 8 &&
                  (macroK != 64 || kernel->hasAttr("runtime_k") || integer("k") % 64 == 0))
               : (macroK == 32 && blockM == 16 && blockN == 16 && waves.getInt() == 1));
      bool stagingValid = (!mxfp8 || mxfp8Profile) && staging && waves &&
          (lds ? (staging.getValue() == "lds" && nk && blockM > 0 &&
                  blockM % 32 == 0 && waves.getInt() > 0 &&
                  waves.getInt() <= 16 && waves.getInt() % (blockM / 32) == 0 &&
                  blockN % (16 * (waves.getInt() / (blockM / 32))) == 0)
               : (staging.getValue() == "global" && waves.getInt() == 1));
      if (!physicalValid || !contract ||
          (contract.getValue() != "rocm_fp8_w8a8_blockscale_v1" &&
           contract.getValue() != "rocm_fp8_w8a8_blockscale_nk_v1" && !mxfp8) ||
          !stagingValid || !shapeValid) {
        kernel->emitError("ROCM_FP8_BLOCKSCALE_CONTRACT: runtime image identity "
                          "requires the verified static W8A8 shape, scale and "
                          "wave-grid contract for the requested projection");
        return signalPassFailure();
      }
    }

    bool mathRecipe = family == "scalar_unary" || family == "scalar_binary" || family == "scan";
    if (mathRecipe) {
      auto c = kernel->getAttrOfType<DictionaryAttr>("native_math_contract");
      auto seal = kernel->getAttrOfType<StringAttr>("schedule_hash");
      auto kind = kernel->getAttrOfType<StringAttr>("kind");
      auto dtype = kernel->getAttrOfType<StringAttr>("dtype");
      auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
      std::string text;
      if (c) { llvm::raw_string_ostream os(text); c.print(os); }
      auto hash = llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
      StringRef expectedFamily = family == "scan" ? "scan" : family == "scalar_binary" ? "binary" : "unary";
      auto contractKind = c ? c.getAs<StringAttr>("kind") : StringAttr();
      bool valid = c && seal && seal.getValue() == hash && kind && dtype && arch &&
          (arch.getValue() == "gfx1151" || arch.getValue() == "gfx1201") &&
          c.get("architecture") == arch && c.get("family") == StringAttr::get(&getContext(), expectedFamily) &&
          (dtype.getValue() == "f32" || dtype.getValue() == "f16" || dtype.getValue() == "bf16") &&
          c.get("storage") == dtype &&
          c.get("output_storage") == StringAttr::get(&getContext(), "f32") &&
          kernel->getAttr("output_dtype") == c.get("output_storage") && contractKind &&
          c.get("numeric_policy") == StringAttr::get(&getContext(), family == "scan" ? "f32_inclusive_scan" : "f32_compute");
      if (valid) {
        valid &= family == "scan" ?
            ((contractKind.getValue() == "sum" && kind.getValue() == "cumsum") ||
             (contractKind.getValue() == "max" && kind.getValue() == "cummax")) :
            kind == contractKind && (family == "scalar_binary" ?
                (kind.getValue() == "add" || kind.getValue() == "div") :
                (kind.getValue() == "sqrt" || kind.getValue() == "exp"));
        for (NamedAttribute attr : kernel->getAttrs())
          if (attr.getName() != "name" && attr.getName() != "kind" &&
              attr.getName() != "dtype" && attr.getName() != "native_math_contract" &&
              attr.getName() != "schedule_hash" && attr.getName() != "output_dtype") valid = false;
      }
      if (!valid) {
        kernel->emitError("native math image identity requires the sealed Target contract");
        return signalPassFailure();
      }
    }
    if (primalProfile) {
      auto member = module->getAttrOfType<DictionaryAttr>("tessera.autodiff.scaled_member");
      auto kind = member ? member.getAs<StringAttr>("kind") : StringAttr{};
      if ((family != "scaled_matmul" && family != "scaled_matmul_lds") ||
          !kind || kind.getValue() != "primal") {
        module.emitError("scaled primal image policy requires a compiler-owned primal member");
        return signalPassFailure();
      }
      auto contract = kernel->getAttrOfType<StringAttr>("physical_contract");
      auto format = kernel->getAttrOfType<StringAttr>("scale_format");
      auto m = kernel->getAttrOfType<IntegerAttr>("m");
      auto n = kernel->getAttrOfType<IntegerAttr>("n");
      auto bm = kernel->getAttrOfType<IntegerAttr>("block_m");
      auto bn = kernel->getAttrOfType<IntegerAttr>("block_n");
      if (!m || !n || !bm || !bn || bm.getInt() <= 0 || bn.getInt() <= 0) {
        module.emitError("scaled primal image policy lost resolved tile alignment");
        return signalPassFailure();
      }
      auto batching = kernel->getAttrOfType<StringAttr>("batching");
      if (batching && batching.getValue() == "broadcast") {
        // Independent logical planes retain static suffix/prefix addressing.
        // A runtime image may not erase those types from its cache identity.
        module->setAttr("tessera.rocm.primal_image_policy",
                       StringAttr::get(&getContext(), "static_independent_prefix_v1"));
        return;
      }
      bool aligned = m.getInt() % bm.getInt() == 0 && n.getInt() % bn.getInt() == 0;
      bool fp8KN = contract && contract.getValue() == "rocm_fp8_w8a8_blockscale_v1" &&
                   format && format.getValue() == "fp32";
      bool fp8NK = contract && contract.getValue() == "rocm_fp8_w8a8_blockscale_nk_v1";
      bool mxNK = contract && contract.getValue() == "rocm_mxfp8_e4m3_e8m0_k32_nk_v1";
      // Static masks favor ragged FP8 KN; constant strides favor aligned NK.
      if ((fp8KN && !aligned) || ((fp8NK || mxNK) && aligned)) {
        module->setAttr("tessera.rocm.primal_image_policy",
                       StringAttr::get(&getContext(), fp8KN ?
                           "static_fp8_kn_v1" : "static_aligned_nk_v1"));
        return;
      }
      module->setAttr("tessera.rocm.primal_image_policy",
                     StringAttr::get(&getContext(), "runtime_projected_v1"));
    }
    OwningOpRef<ModuleOp> projected = ModuleOp::create(module.getLoc());
    projected->getOperation()->setAttrs(module->getAttrs());
    // Program SSA/lifetimes bind the launch, not the physical kernel symbol.
    // Keep them on the actual module for member ABI export and replay.
    for (StringRef key : {"tessera.autodiff.scaled_member",
                          "tessera.autodiff.scaled_program_json",
                          "tessera.autodiff.scaled_program_witness",
                          "tessera.rocm.primal_image_policy"})
      projected->getOperation()->removeAttr(key);

    if (family == "scaled_matmul" || family == "scaled_matmul_lds")
      // The verified directive already owns the resolved physical profile.
      // Frontend selection intent must not duplicate that image cache key.
      projected->getOperation()->removeAttr("tessera.rocm.mxfp8_schedule");
    if (mathRecipe) projected->getOperation()->removeAttr("tessera.launch_bindings");
    Operation *copy = kernel->clone();
    if (mathRecipe) {
      copy->removeAttr("native_math_contract");
      copy->removeAttr("schedule_hash");
    }

    copy->setAttr("name", StringAttr::get(&getContext(), ""));
    if (family == "matmul" || family == "scaled_matmul" || family == "scaled_matmul_lds" || family == "folded_matmul")
      copy->removeAttr("tessera.schedule_hash");
    auto batching = copy->getAttrOfType<StringAttr>("batching");
    bool independentPlanes = batching && batching.getValue() == "broadcast";
    if (family == "scaled_matmul" && !independentPlanes) {
      OpBuilder builder(&getContext());
      copy->setAttr("runtime_shape", builder.getUnitAttr());
      for (StringRef axis : {"m", "n", "k"})
        copy->setAttr(axis, builder.getI64IntegerAttr(0));
    }
    if ((family == "scaled_matmul_lds" || family == "folded_matmul") && !independentPlanes) {
      OpBuilder builder(&getContext());
      copy->setAttr("runtime_mn", builder.getUnitAttr());
      for (StringRef axis : {"m", "n"}) {
        auto extent = copy->getAttrOfType<IntegerAttr>(axis);
        auto block = copy->getAttrOfType<IntegerAttr>(
            axis == "m" ? "block_m" : "block_n");
        copy->setAttr(axis == "m" ? "whole_m" : "whole_n",
                      builder.getBoolAttr(extent.getInt() % block.getInt() == 0));
        copy->setAttr(axis, builder.getI64IntegerAttr(0));
      }
      if (runtimeK) {
        copy->setAttr("runtime_k", builder.getUnitAttr());
        copy->setAttr("k", builder.getI64IntegerAttr(0));
        if (family == "folded_matmul") {
          // Full-K is a semantic scope, not a constant scale group. Zero
          // records the runtime extent; the physical stage remains K64.
          copy->setAttr("scale_k", builder.getI64IntegerAttr(0));
          copy->setAttr("macro_k", builder.getI64IntegerAttr(0));
        }
      }
    }
    projected->getBody()->push_back(copy);
    std::string canonical;
    llvm::raw_string_ostream os(canonical);
    projected->print(os, OpPrintingFlags().useLocalScope());
    os.flush();
    llvm::SHA256 hash;
    hash.update("tessera.rocm_shape_free_kernel.v2");
    hash.update("\x1f");
    hash.update(family.getValue());
    hash.update("\x1f");
    hash.update(canonical);
    auto digest = llvm::toHex(hash.final(), true);
    std::string symbol = "tessera_rocm_" + family.getValue() + "_" +
                         digest.substr(0, 16);
    copy->setAttr("name", StringAttr::get(&getContext(), symbol));
    // The clone owns no operands, so erasing the host scaffold cannot leave
    // dangling references. Mutate only after the complete contract is checked.
    if (family == "scaled_matmul" || family == "scaled_matmul_lds")
      module->removeAttr("tessera.rocm.mxfp8_schedule");
    if (mathRecipe) module->removeAttr("tessera.launch_bindings");
    module.getBody()->clear();
    module.getBody()->push_back(copy->clone());
  }
};
} // namespace
std::unique_ptr<Pass>
mlir::tessera_rocm::createProjectROCMKernelIdentityPass() {
  return std::make_unique<ProjectROCMKernelIdentityPass>();
}
