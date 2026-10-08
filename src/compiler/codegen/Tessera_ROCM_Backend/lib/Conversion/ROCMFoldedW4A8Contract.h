// Shared native admission for the folded full-K Target contract.
#ifndef TESSERA_ROCM_FOLDED_W4A8_CONTRACT_H
#define TESSERA_ROCM_FOLDED_W4A8_CONTRACT_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
namespace mlir::tessera_rocm {
inline bool isValidFoldedW4A8Target(Operation *op, bool allowRuntimeMN) {
  auto text = [&](llvm::StringRef key) -> llvm::StringRef {
    auto a = op->getAttrOfType<StringAttr>(key);
    return a ? a.getValue() : llvm::StringRef();
  };
  auto integer = [&](llvm::StringRef key) -> int64_t {
    auto a = op->getAttrOfType<IntegerAttr>(key);
    return a ? a.getInt() : -1;
  };
  auto module = op->getParentOfType<ModuleOp>();
  auto arch = module ? module->getAttrOfType<StringAttr>("tessera.arch") : StringAttr();
  auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy");
  auto policyText = [&](llvm::StringRef key) -> llvm::StringRef {
    auto a = policy ? policy.getAs<StringAttr>(key) : StringAttr();
    return a ? a.getValue() : llvm::StringRef();
  };
  bool packed = text("physical_contract") == "rocm_mxfp4_w4a8_packed_folded_prefill_v1";
  int64_t m = integer("m"), n = integer("n"), k = integer("k");
  // The existing packed Schedule profile has one fixed physical schedule.
  // Defaults are native-owned; undeclared tuning attributes cannot override it.
  bool scheduleValid = packed
      ? !op->hasAttr("raster_group_m") && !op->hasAttr("workgroup_mode") &&
        !op->hasAttr("row_guard") && !op->hasAttr("staging_prefetch") &&
        !op->hasAttr("epilogue_schedule") &&
        ((n > 0 && n % 16 == 0) ||
         (allowRuntimeMN && op->hasAttrOfType<UnitAttr>("runtime_mn") && n == 0))
      : integer("raster_group_m") >= 0 && integer("raster_group_m") <= 64 &&
        (text("workgroup_mode") == "cu" || text("workgroup_mode") == "wgp") &&
        (text("row_guard") == "wave" || text("row_guard") == "cta") &&
        (text("staging_prefetch") == "none" || text("staging_prefetch") == "register_next_slab") &&
        (text("epilogue_schedule") == "predicated_scalar_scales" ||
         text("epilogue_schedule") == "complete_tile_vector_scales");
  bool runtimeMN = op->hasAttr("runtime_mn");
  bool runtimeK = op->hasAttr("runtime_k");
  bool kValid = runtimeK
      ? allowRuntimeMN && runtimeMN && isa<UnitAttr>(op->getAttr("runtime_k")) &&
            k == 0 && integer("scale_k") == 0 && integer("macro_k") == 0
      : k > 0 && k % 64 == 0 && integer("scale_k") == k && integer("macro_k") == k;
  bool shapeValid = runtimeMN
      ? allowRuntimeMN && isa<UnitAttr>(op->getAttr("runtime_mn")) &&
            m == 0 && n == 0 &&
            op->getAttrOfType<BoolAttr>("whole_m") &&
            op->getAttrOfType<BoolAttr>("whole_n")
      : m > (packed ? 0 : 64) && n > 0 && !op->hasAttr("whole_m") && !op->hasAttr("whole_n");
  return arch && arch.getValue() == "gfx1201" && shapeValid &&
      !op->hasAttr("runtime_shape") && kValid &&
      !text("name").empty() &&
      (packed || text("physical_contract") == "rocm_mxfp4_w4a8_folded_prefill_v1") &&
      text("abi") == (packed ? "a_bpacked_sa_scaleplane_d_m_n_k" : "a_bfold_sa_rowref_d_m_n_k") &&
      text("package_abi") == (packed
          ? "tessera.rocm.mxfp4_w4a8.a_bpacked_sa_scaleplane_o_m_n_k.e4m3_e2m1_e8m0_bf16.approx_bm256_tm4.v1"
          : "tessera.rocm.mxfp4_w4a8.a_bfold_sa_rowref_o_m_n_k.e4m3_e4m3_e8m0_bf16.approx_bm256_tm4.v1") &&
      text("scale_format") == (packed ? "e8m0_k32_plus_row_reference" : "e8m0_row_reference") &&
      text("partial_combine") == "row_reference_after_full_k" &&
      text("k_step_schedule") == "isolated_k_stage" &&
      text("output") == "bf16" && policyText("accum") == "f32" &&
      policyText("storage") == "e4m3_raw_u8" &&
      policyText("execution_mode") == "folded_row_reference_explicit_approximate" &&
      integer("instruction_k") == 16 && integer("stage_k") == 64 &&
      integer("block_m") == 256 && integer("block_n") == 64 &&
      integer("tile_m_per_wave") == 4 && integer("tile_n_per_wave") == 2 &&
      scheduleValid;

}
} // namespace mlir::tessera_rocm
#endif
