//===- InitTPP.cpp - Init stub for the TPP solver -----------*- C++ -*-===//
//
// Placeholder translation unit referenced by `lib/CMakeLists.txt`. The TPP
// solver is currently in scaffold state per CLAUDE.md ("dialect defined,
// needs wiring"); the dialect and passes self-register via static registration
// objects (see lib/Dialect/TPP/TPPDialect.cpp and lib/Passes/PassPipeline.cpp),
// so this file deliberately exposes no additional symbols.
//
// Once the TPP solver moves out of scaffold state, this file is the right
// place to add explicit `tessera::tpp::register*` entry points called by
// tessera-opt / tessera-tpp-opt at startup.
//
//===----------------------------------------------------------------------===//

namespace tessera {
namespace tpp {

// Intentionally empty.

} // namespace tpp
} // namespace tessera
