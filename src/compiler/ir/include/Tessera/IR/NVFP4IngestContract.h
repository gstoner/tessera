#ifndef TESSERA_IR_NVFP4INGESTCONTRACT_H
#define TESSERA_IR_NVFP4INGESTCONTRACT_H
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include <limits>
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tessera_contract {
inline std::string nvfp4ContractHash(DictionaryAttr contract) {
  std::string text; llvm::raw_string_ostream os(text); contract.print(os);os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
inline LogicalResult verifyNVFP4Policy(Operation *op, DictionaryAttr policy) {
  if (!policy) return op->emitError("NVFP4 ingest requires an explicit numeric policy");
  const std::pair<StringRef,StringRef> fields[] = {
    {"execution_mode","explicit_scale_requantization"},
    {"source_format","nvfp4_e2m1_e4m3_k16"},
    {"destination_format","mxfp4_e2m1_e8m0_k32"},
    {"source_scale_application","projection_global_times_e4m3"},
    {"destination_code_selection","nearest_signed_e2m1_by_weight_sse"},
    {"destination_scale_order","k_group_n"}};
  if (policy.size() != 7)
    return op->emitError("NVFP4 ingest numeric policy has unexpected fields");
  for (auto [key,value] : fields) {
    auto attr = policy.getAs<StringAttr>(key);
    if (!attr || attr.getValue() != value)
      return op->emitError("NVFP4 ingest conflicting numeric policy field ") << key;
  }
  auto loss = policy.getAs<ArrayAttr>("lossy_steps");
  if (!loss || loss.size() != 1 || !isa<StringAttr>(loss[0]) ||
      cast<StringAttr>(loss[0]).getValue() !=
      "nvfp4_e4m3_k16_to_mxfp4_e8m0_k32_scale_and_code_requantization")
    return op->emitError("NVFP4 ingest must declare its single scale/code requantization loss");
  return success();
}
inline LogicalResult verifyNVFP4Extents(Operation *op, int64_t n, int64_t k,
                                       ArrayAttr offsets) {
  if (n <= 0 || k <= 0 || k % 32 ||
      n > std::numeric_limits<int64_t>::max() / (k/2))
    return op->emitError("NVFP4 ingest requires representable positive N/K32 extents");
  if (!offsets || offsets.size() < 2)
    return op->emitError("NVFP4 ingest requires complete projection row boundaries");
  int64_t previous = -1;
  for (Attribute attr : offsets) {
    auto value = dyn_cast<IntegerAttr>(attr);
    if (!value || value.getInt() <= previous)
      return op->emitError("NVFP4 ingest requires increasing integer row boundaries");
    previous = value.getInt();
  }
  if (cast<IntegerAttr>(offsets[0]).getInt() != 0 || previous != n)
    return op->emitError("NVFP4 ingest row boundaries must start at zero and end at N");
  return success();
}

// Codes are exact; the scale product is rounded once from f64 to f32.
inline LogicalResult verifyNVFP4ExpandPolicy(Operation *op, DictionaryAttr policy) {
  const std::pair<StringRef,StringRef> fields[] = {
    {"execution_mode","source_preserving_expansion"},
    {"source_format","nvfp4_e2m1_e4m3_k16"},
    {"destination_format","fp8_e4m3_k16_f32_scales"},
    {"source_scale_application","projection_global_times_e4m3"},
    {"destination_code_selection","exact_e2m1_signed_zero"},
    {"destination_scale_order","k_group_n"},
    {"scale_rounding","f64_product_to_f32_rne"}};
  if (!policy || policy.size() != 7)
    return op->emitError("NVFP4 expansion requires its complete explicit numeric policy");
  for (auto [key,value] : fields) {
    auto attr = policy.getAs<StringAttr>(key);
    if (!attr || attr.getValue() != value)
      return op->emitError("NVFP4 expansion conflicting numeric policy field ") << key;
  }
  return success();
}
inline LogicalResult verifyNVFP4ExpandExtents(Operation *op, int64_t n,
                                             int64_t k, ArrayAttr offsets) {
  if (n <= 0 || k <= 0 || k % 16 ||
      n > std::numeric_limits<int64_t>::max() / k)
    return op->emitError("NVFP4 expansion requires representable positive N/K16 extents");
  if (!offsets || offsets.size() < 2)
    return op->emitError("NVFP4 expansion requires complete projection row boundaries");
  int64_t previous = -1;
  for (Attribute attr : offsets) {
    auto value = dyn_cast<IntegerAttr>(attr);
    if (!value || value.getInt() <= previous)
      return op->emitError("NVFP4 expansion requires increasing integer row boundaries");
    previous = value.getInt();
  }
  if (cast<IntegerAttr>(offsets[0]).getInt() != 0 || previous != n)
    return op->emitError("NVFP4 expansion row boundaries must start at zero and end at N");
  return success();
}
inline LogicalResult verifyMXFP4StorageExtents(Operation *op, int64_t n, int64_t k,
                                               StringAttr contract) {
  if (!contract || contract.getValue() !=
      "mxfp4.gfx12.n16_k16_lane_u32.plus_row_reference.v1" ||
      n <= 0 || n % 16 || k <= 0 || k % 64 ||
      n > std::numeric_limits<int64_t>::max() / (k / 2))
    return op->emitError("MXFP4 storage bridge requires its named lossless layout contract and representable N16/K64 extents");
  return success();
}
}
#endif
