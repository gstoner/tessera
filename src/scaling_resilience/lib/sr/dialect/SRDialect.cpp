#include "tessera/sr/Dialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace mlir { namespace tessera { namespace sr {

namespace {
struct SRDialect : public Dialect {
  explicit SRDialect(MLIRContext *ctx) : Dialect("tessera_sr", ctx, TypeID::get<SRDialect>()) {
    // Normally generated: addOperations<...>(); addTypes<...>();
  }
};
} // namespace

void registerDialect() {
  // In a real build this is auto-registered via GEN code, but keep an explicit hook.
  // (No-op placeholder; parent tree will register via ODS-gen in CMake).
}

}}} // namespaces