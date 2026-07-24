// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_cpu)' --allow-unregistered-dialect | FileCheck --check-prefix=CPU %s
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu)' --allow-unregistered-dialect | FileCheck --check-prefix=GPU %s

// kv_cache_coverage_matrix.md (2026-05-10) — Apple CPU + GPU kv_cache
// lowering must emit real `tessera_apple.{cpu,gpu}.kv_cache_op`
// artifacts (carrying `kind=tessera.kv_cache.*` and the runtime ABI
// tag), NOT the historical `tessera_apple.diagnostic("unsupported")`.

module {
  // CPU-LABEL: module
  // CPU:       tessera_apple.cpu.kv_cache_cpu
  // CPU-SAME:    abi = "kv_cache_handle"
  // CPU-SAME:    kind = "tessera.kv_cache.append"
  // CPU-SAME:    status = "executable"
  // CPU-NOT:   tessera_apple.diagnostic

  // GPU-LABEL: module
  // GPU:       tessera_apple.gpu.kv_cache_gpu
  // GPU-SAME:    abi = "kv_cache_handle"
  // GPU-SAME:    framework = "Metal"
  // GPU-SAME:    kind = "tessera.kv_cache.append"
  // GPU-SAME:    status = "artifact_only"
  // GPU-NOT:   tessera_apple.diagnostic

  func.func @kv_pipeline(%c: !tessera.kv_cache,
                          %k: tensor<8x4x16xf32>,
                          %v: tensor<8x4x16xf32>) {
    // Artifact-only contract — TileToApple erases the source op after
    // emitting the artifact, so we don't keep the result alive.
    %updated = "tessera.kv_cache.append"(%c, %k, %v)
        : (!tessera.kv_cache, tensor<8x4x16xf32>, tensor<8x4x16xf32>) -> !tessera.kv_cache
    return
  }
}
