#ifndef TESSERA_ROCM_FRAGMENT_LAYOUT_H
#define TESSERA_ROCM_FRAGMENT_LAYOUT_H

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <initializer_list>
#include <optional>

namespace mlir::tessera_rocm {

enum class FragmentFamily {
  RDNA3WMMA,
  RDNA4WMMA,
  CDNA5WMMA,
  CDNA2MFMA,
  CDNA3MFMA,
  CDNA4MFMA,
};

enum class FragmentRegisterFormat {
  WMMAInputGFX11,
  WMMAAccGFX11,
  SOA,
  SOAInt,
};

struct FragmentLayoutDescriptor {
  FragmentFamily family;
  llvm::StringRef familyName;
  llvm::StringRef matrixOp;
  /// Exact ISA operation selected for this architecture/dtype/shape tuple.
  /// This is provenance, not an execution claim: materializationReady remains
  /// the independent gate for fragment packing and intrinsic lowering.
  llvm::StringRef matrixInstruction;
  int64_t waveSize;
  int64_t inputElementsPerLane;
  int64_t inputRegistersPerLane;
  int64_t accumulatorElementsPerLane;
  int64_t accumulatorRegistersPerLane;
  FragmentRegisterFormat inputFormat;
  FragmentRegisterFormat accumulatorFormat;
  int64_t inputLaneReplication;
  llvm::StringRef intrinsicABI;
  bool materializationReady;

  bool usesGfx11AccumulatorMap() const {
    return accumulatorFormat == FragmentRegisterFormat::WMMAAccGFX11;
  }
};

inline llvm::StringRef registerFormatName(FragmentRegisterFormat format) {
  switch (format) {
  case FragmentRegisterFormat::WMMAInputGFX11:
    return "wmma_input_gfx11";
  case FragmentRegisterFormat::WMMAAccGFX11:
    return "wmma_acc_gfx11";
  case FragmentRegisterFormat::SOA:
    return "soa";
  case FragmentRegisterFormat::SOAInt:
    return "soa_int";
  }
  llvm_unreachable("unknown ROCm fragment register format");
}

inline bool isAnyOf(llvm::StringRef value,
                    std::initializer_list<llvm::StringRef> choices) {
  for (llvm::StringRef choice : choices)
    if (value == choice)
      return true;
  return false;
}

inline int64_t dtypeBits(llvm::StringRef dtype) {
  if (isAnyOf(dtype, {"f16", "bf16"}))
    return 16;
  if (isAnyOf(dtype, {"int8", "e4m3", "e5m2", "fp8", "bf8"}))
    return 8;
  if (isAnyOf(dtype, {"int4", "fp4"}))
    return 4;
  if (dtype == "f32")
    return 32;
  return 0;
}

inline llvm::StringRef denseMatrixInstruction(llvm::StringRef arch,
                                              llvm::StringRef dtype,
                                              int64_t k) {
  if (arch == "gfx1100" || arch == "gfx1151") {
    if (dtype == "f16")
      return "V_WMMA_F32_16X16X16_F16";
    if (dtype == "bf16")
      return "V_WMMA_F32_16X16X16_BF16";
    if (dtype == "int8")
      return "V_WMMA_I32_16X16X16_IU8";
    if (dtype == "int4")
      return "V_WMMA_I32_16X16X16_IU4";
  }
  if (arch == "gfx1200" || arch == "gfx1201") {
    if (dtype == "f16")
      return "V_WMMA_F32_16X16X16_F16";
    if (dtype == "bf16")
      return "V_WMMA_F32_16X16X16_BF16";
    if (dtype == "e4m3" || dtype == "fp8")
      return "V_WMMA_F32_16X16X16_FP8_FP8";
    if (dtype == "e5m2" || dtype == "bf8")
      return "V_WMMA_F32_16X16X16_BF8_BF8";
    if (dtype == "int8")
      return "V_WMMA_I32_16X16X16_IU8";
    if (dtype == "int4" && k == 16)
      return "V_WMMA_I32_16X16X16_IU4";
    if (dtype == "int4" && k == 32)
      return "V_WMMA_I32_16X16X32_IU4";
  }
  if (arch == "gfx1250" || arch == "gfx1251") {
    if (dtype == "f32" && k == 4)
      return "V_WMMA_F32_16X16X4_F32";
    if (dtype == "f16" && k == 32)
      return "V_WMMA_F32_16X16X32_F16";
    if (dtype == "bf16" && k == 32)
      return "V_WMMA_F32_16X16X32_BF16";
    if ((dtype == "e4m3" || dtype == "fp8") && k == 64)
      return "V_WMMA_F32_16X16X64_FP8_FP8";
    if ((dtype == "e5m2" || dtype == "bf8") && k == 64)
      return "V_WMMA_F32_16X16X64_BF8_BF8";
    if ((dtype == "e4m3" || dtype == "fp8") && k == 128)
      return "V_WMMA_F32_16X16X128_FP8_FP8";
    if ((dtype == "e5m2" || dtype == "bf8") && k == 128)
      return "V_WMMA_F32_16X16X128_BF8_BF8";
    if (dtype == "int8" && k == 64)
      return "V_WMMA_I32_16X16X64_IU8";
    if (dtype == "fp4" && k == 128)
      return "V_WMMA_SCALE_F32_16X16X128_F8F6F4";
  }
  return {};
}

inline std::optional<FragmentLayoutDescriptor>
resolveFragmentLayout(tessera::tile::TileMmaDescAttr desc,
                      llvm::StringRef arch) {
  if (!desc || desc.getM() != 16 || desc.getN() != 16 ||
      desc.getAType() != desc.getBType() ||
      desc.getALayout() != "row_major" ||
      desc.getBLayout() != "col_major" || desc.getKBlocks() != 1)
    return std::nullopt;

  llvm::StringRef dtype = desc.getAType();
  int64_t bits = dtypeBits(dtype);
  bool integer = dtype == "int8" || dtype == "int4";
  if (bits == 0 || (integer ? !isAnyOf(desc.getAccType(), {"i32", "int32"})
                            : desc.getAccType() != "f32"))
    return std::nullopt;

  auto make = [&](FragmentFamily family, llvm::StringRef familyName,
                  llvm::StringRef matrixOp, llvm::StringRef matrixInstruction,
                  int64_t waveSize,
                  int64_t inputElements, FragmentRegisterFormat inputFormat,
                  FragmentRegisterFormat accumulatorFormat,
                  int64_t replication, llvm::StringRef abi,
                  bool ready = true) -> FragmentLayoutDescriptor {
    return {family,
            familyName,
            matrixOp,
            matrixInstruction,
            waveSize,
            inputElements,
            (inputElements * bits + 31) / 32,
            256 / waveSize,
            256 / waveSize,
            inputFormat,
            accumulatorFormat,
            replication,
            abi,
            ready};
  };

  if (arch == "gfx1100" || arch == "gfx1151") {
    if (desc.getK() != 16 ||
        !isAnyOf(dtype, {"f16", "bf16", "int8", "int4"}) ||
        (desc.getFamily() != "auto" && desc.getFamily() != "wmma"))
      return std::nullopt;
    return make(FragmentFamily::RDNA3WMMA, "rdna3_wmma", "wmma",
                denseMatrixInstruction(arch, dtype, desc.getK()), 32,
                16, FragmentRegisterFormat::WMMAInputGFX11,
                FragmentRegisterFormat::WMMAAccGFX11, 2,
                "abc_3arg_gfx11");
  }

  if (arch == "gfx1200" || arch == "gfx1201") {
    int64_t expectedK = dtype == "int4" ? 32 : 16;
    if (desc.getK() != expectedK ||
        !isAnyOf(dtype,
                 {"f16", "bf16", "e4m3", "e5m2", "fp8", "bf8",
                  "int8", "int4"}) ||
        (desc.getFamily() != "auto" && desc.getFamily() != "wmma"))
      return std::nullopt;
    int64_t inputElements = 16 * expectedK / 32;
    FragmentRegisterFormat inputFormat =
        bits >= 16 ? FragmentRegisterFormat::SOA
                   : FragmentRegisterFormat::SOAInt;
    return make(FragmentFamily::RDNA4WMMA, "rdna4_wmma", "wmma",
                denseMatrixInstruction(arch, dtype, desc.getK()), 32,
                inputElements, inputFormat,
                integer ? FragmentRegisterFormat::SOAInt
                        : FragmentRegisterFormat::SOA,
                1, "abc_3arg_gfx12");
  }

  if (arch == "gfx1250" || arch == "gfx1251") {
    bool validShape =
        (dtype == "f32" && desc.getK() == 4) ||
        (isAnyOf(dtype, {"f16", "bf16"}) && desc.getK() == 32) ||
        (isAnyOf(dtype, {"e4m3", "e5m2", "fp8", "bf8"}) &&
         (desc.getK() == 64 || desc.getK() == 128)) ||
        (dtype == "int8" && desc.getK() == 64) ||
        (dtype == "fp4" && desc.getK() == 128);
    if (!validShape ||
        (desc.getFamily() != "auto" && desc.getFamily() != "wmma"))
      return std::nullopt;
    int64_t inputElements = 16 * desc.getK() / 32;
    FragmentRegisterFormat inputFormat =
        bits >= 16 ? FragmentRegisterFormat::SOA
                   : FragmentRegisterFormat::SOAInt;
    return make(FragmentFamily::CDNA5WMMA, "cdna5_wmma", "wmma",
                denseMatrixInstruction(arch, dtype, desc.getK()), 32,
                inputElements, inputFormat,
                integer ? FragmentRegisterFormat::SOAInt
                        : FragmentRegisterFormat::SOA,
                1, "mods_reuse_scale_gfx125x",
                isAnyOf(dtype, {"f16", "bf16"}));
  }

  FragmentFamily family;
  llvm::StringRef familyName;
  if (arch == "gfx90a") {
    family = FragmentFamily::CDNA2MFMA;
    familyName = "cdna2_mfma";
  } else if (arch == "gfx940" || arch == "gfx942") {
    family = FragmentFamily::CDNA3MFMA;
    familyName = "cdna3_mfma";
  } else if (arch == "gfx950") {
    family = FragmentFamily::CDNA4MFMA;
    familyName = "cdna4_mfma";
  } else {
    return std::nullopt;
  }
  if (desc.getFamily() != "auto" && desc.getFamily() != "mfma")
    return std::nullopt;
  int64_t expectedK = isAnyOf(dtype, {"f16", "bf16", "int8"}) ? 16
                    : dtype == "f32"                              ? 8
                    : isAnyOf(dtype, {"e4m3", "e5m2", "fp8", "bf8"})
                        ? 32
                        : dtype == "fp4" ? 64 : 0;
  if (expectedK == 0 || desc.getK() != expectedK)
    return std::nullopt;
  if (family == FragmentFamily::CDNA2MFMA && bits < 16)
    return std::nullopt;
  if (family == FragmentFamily::CDNA3MFMA && dtype == "fp4")
    return std::nullopt;
  int64_t inputElements = 16 * expectedK / 64;
  FragmentRegisterFormat inputFormat =
      bits >= 16 ? FragmentRegisterFormat::SOA
                 : FragmentRegisterFormat::SOAInt;
  return make(family, familyName, "mfma", "mfma", 64, inputElements, inputFormat,
              integer ? FragmentRegisterFormat::SOAInt
                      : FragmentRegisterFormat::SOA,
              1, "mfma_abc_ctrl", isAnyOf(dtype, {"f16", "bf16"}));
}

} // namespace mlir::tessera_rocm

#endif // TESSERA_ROCM_FRAGMENT_LAYOUT_H
