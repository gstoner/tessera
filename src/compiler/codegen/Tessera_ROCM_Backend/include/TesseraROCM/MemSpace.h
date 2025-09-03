#pragma once
#include "mlir/IR/Attributes.h"
namespace mlir::tessera_rocm {
struct MemSpace {
  enum Kind { Global = 1, LDS = 3, Private = 5, Unknown = 0 };
  Kind kind = Unknown;
  static std::optional<MemSpace> parse(mlir::Attribute attr);
  static const char* kindToStr(Kind k);
  static unsigned toAddressSpace(Kind k);
};
} // namespace mlir::tessera_rocm
