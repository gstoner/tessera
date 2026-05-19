//===- InitTPP.h - Public registration entry points for the TPP dialect --===//
//
// Thin public header that tools (tessera-opt, tessera-tpp-opt, ...) include
// to register the TPP dialect + passes + named pipeline aliases.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TPP_INIT_H
#define TESSERA_TPP_INIT_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace tessera {
namespace tpp {

/// Insert the TPP dialect into the registry.
void registerTPPDialect(::mlir::DialectRegistry &registry);

/// Register all TPP individual passes (`-tpp-halo-infer`,
/// `-tpp-legalize-space-time`, `-tpp-fuse-stencil-time`,
/// `-tpp-async-prefetch`, `-tpp-vectorize`, `-tpp-distribute-halo`,
/// `-lower-tpp-to-target-ir`).
void registerTPPPasses();

/// Register the canonical `-tpp-space-time` pipeline alias.
void registerTPPPipelines();

/// Convenience — register dialect into the registry AND all passes/pipelines.
inline void registerAllTPP(::mlir::DialectRegistry &registry) {
  registerTPPDialect(registry);
  registerTPPPasses();
  registerTPPPipelines();
}

} // namespace tpp
} // namespace tessera

#endif // TESSERA_TPP_INIT_H
