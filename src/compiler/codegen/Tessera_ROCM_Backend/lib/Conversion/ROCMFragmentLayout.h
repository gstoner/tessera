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
  GFX125XWMMAV2,
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
                  llvm::StringRef matrixOp, int64_t waveSize,
                  int64_t inputElements, FragmentRegisterFormat inputFormat,
                  FragmentRegisterFormat accumulatorFormat,
                  int64_t replication, llvm::StringRef abi,
                  bool ready = true) -> FragmentLayoutDescriptor {
    return {family,
            familyName,
            matrixOp,
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
    return make(FragmentFamily::RDNA3WMMA, "rdna3_wmma", "wmma", 32,
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
    return make(FragmentFamily::RDNA4WMMA, "rdna4_wmma", "wmma", 32,
                inputElements, inputFormat,
                integer ? FragmentRegisterFormat::SOAInt
                        : FragmentRegisterFormat::SOA,
                1, "abc_3arg_gfx12");
  }

  if (arch == "gfx1250" || arch == "gfx1251") {
    int64_t expectedK = isAnyOf(dtype, {"f16", "bf16"}) ? 32 : 64;
    if (desc.getK() != expectedK ||
        !isAnyOf(dtype, {"f16", "bf16", "e4m3", "e5m2", "fp8", "bf8"}) ||
        (desc.getFamily() != "auto" && desc.getFamily() != "wmma"))
      return std::nullopt;
    return make(FragmentFamily::GFX125XWMMAV2, "gfx125x_wmma_v2", "wmma",
                32, 16 * expectedK / 32, FragmentRegisterFormat::SOA,
                FragmentRegisterFormat::SOA, 1,
                "mods_reuse_8arg_gfx125x", isAnyOf(dtype, {"f16", "bf16"}));
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
  return make(family, familyName, "mfma", 64, inputElements, inputFormat,
              integer ? FragmentRegisterFormat::SOAInt
                      : FragmentRegisterFormat::SOA,
              1, "mfma_abc_ctrl", isAnyOf(dtype, {"f16", "bf16"}));
}

} // namespace mlir::tessera_rocm

#endif // TESSERA_ROCM_FRAGMENT_LAYOUT_H
