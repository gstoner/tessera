#include "Tessera/Target/Metalium/Util/MemSpace.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tessera::metalium;

static llvm::StringRef parseMemspaceFromPrinted(Attribute a, std::string &storage) {
  storage.clear();
  {
    llvm::raw_string_ostream os(storage);
    a.print(os);
  }
  auto l = storage.find('<');
  auto r = storage.rfind('>');
  if (l == std::string::npos || r == std::string::npos || r <= l + 1) return "";
  auto inner = storage.substr(l + 1, r - l - 1);
  if (inner.size() >= 2 && inner.front() == '"' && inner.back() == '"')
    inner = inner.substr(1, inner.size() - 2);
  static std::string keep;
  keep = inner;
  return keep;
}

llvm::StringRef mlir::tessera::metalium::getMemSpaceName(Attribute msAttr) {
  if (!msAttr) return "";
  if (auto s = dyn_cast<StringAttr>(msAttr)) return s.getValue();
  if (auto oa = dyn_cast<OpaqueAttr>(msAttr)) {
    if (oa.getDialectNamespace() == "tessera_metalium") {
      std::string printed;
      return parseMemspaceFromPrinted(msAttr, printed);
    }
  }
  std::string printed;
  return parseMemspaceFromPrinted(msAttr, printed);
}

llvm::StringRef mlir::tessera::metalium::getMemSpaceNameFromType(Type type) {
  if (auto mr = dyn_cast<MemRefType>(type)) {
    Attribute ms = mr.getMemorySpace();
    return getMemSpaceName(ms);
  }
  return "";
}

MemSpaceKind mlir::tessera::metalium::toMemSpaceKind(llvm::StringRef name) {
  if (name == "dram") return MemSpaceKind::DRAM;
  if (name == "sram") return MemSpaceKind::SRAM;
  return MemSpaceKind::Unknown;
}
