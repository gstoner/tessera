#ifndef TESSERA_TARGET_METALIUM_UTIL_MEMSPACE_H
#define TESSERA_TARGET_METALIUM_UTIL_MEMSPACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace tessera {
namespace metalium {

enum class MemSpaceKind { Unknown, DRAM, SRAM };

/// Return a canonical name ("dram"/"sram"/"") for a memorySpace attribute.
llvm::StringRef getMemSpaceName(Attribute msAttr);

/// Read memory space from a MemRef type's memorySpace attribute.
llvm::StringRef getMemSpaceNameFromType(Type type);

/// Convenience: map canonical names to enum.
MemSpaceKind toMemSpaceKind(llvm::StringRef name);

} // namespace metalium
} // namespace tessera
} // namespace mlir

#endif // TESSERA_TARGET_METALIUM_UTIL_MEMSPACE_H
