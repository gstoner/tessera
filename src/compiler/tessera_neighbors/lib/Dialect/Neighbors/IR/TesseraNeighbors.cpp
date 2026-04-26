#include "mlir/IR/Dialect.h"
using namespace mlir;
namespace tessera { namespace neighbors {
struct NeighborsDialect : public Dialect {
  explicit NeighborsDialect(MLIRContext *ctx) : Dialect("tessera.neighbors", ctx, TypeID::get<NeighborsDialect>()) {
    // TODO: add operations/types once generated from ODS.
  }
};
}} // namespace
