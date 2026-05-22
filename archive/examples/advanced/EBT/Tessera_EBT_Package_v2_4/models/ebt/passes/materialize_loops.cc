#include "materialize_loops.h"
// Scaffold implementation notes:
// - In-tree, derive from mlir::PassWrapper<...> and walk functions that carry
//   attributes like tessera.ebt.graph to find the driver region.
// - Insert scf.for %k in [0,K) and nested scf.for %t in [0,T) using builders.
// - Hoist loop-invariant context (h) and clone inner ops into the loop body.
// - Attach attributes: scf.for { mapping = "candidates" } / { mapping = "steps" }.
// - Read K/T from pipeline options or module attrs.
namespace tessera { namespace ebt {
void registerEBTMaterializeLoopsPass() {}
}} // ns
