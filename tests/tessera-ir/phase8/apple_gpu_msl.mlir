// RUN: tessera-opt %s --allow-unregistered-dialect | FileCheck %s

// Phase 8.4 — round-trip the new tessera_apple.gpu.msl_kernel op through
// tessera-opt to verify the dialect declaration parses + prints cleanly. The
// op is the IR-level carrier for a custom Metal Shading Language kernel:
// the runtime compiles via [device newLibraryWithSource:options:error:] and
// caches the resulting MTLComputePipelineState by `cache_key`.
//
// End-to-end pass-level lowering of tessera.rope -> tessera_apple_gpu_rope_f32
// is verified by the Python unit tests (test_apple_backend_roadmap.py) which
// exercise the full Graph -> Schedule -> Tile -> Target IR pipeline. The lit
// fixture here is the dialect contract for the new op only; lit pass tests
// for tessera.rope require the op to be registered in the Tessera dialect,
// which is out of scope for Phase 8.4.

module attributes {tessera.ir.level = "target"} {
  // CHECK:      tessera_apple.gpu.msl_kernel
  // CHECK-SAME: cache_key = "{{[0-9a-f]+}}"
  // CHECK-SAME: dtype = "f32"
  // CHECK-SAME: entry_point = "rope_f32"
  // CHECK-SAME: framework = "Metal"
  // CHECK-SAME: msl_source = "{{.*kernel void rope_f32.*}}"
  "tessera_apple.gpu.msl_kernel"() {
    source = "tessera.rope",
    result = "v0",
    ordinal = 0 : i64,
    entry_point = "rope_f32",
    msl_source = "kernel void rope_f32(device const float* x [[buffer(0)]]) { }",
    framework = "Metal",
    dtype = "f32",
    cache_key = "deadbeef0badc0de",
    grid = "tokens_pairs",
    threadgroup = "32x?"
  } : () -> ()

  // CHECK:      tessera_apple.gpu.mps_dispatch
  // CHECK-SAME: execution_mode = "metal_runtime"
  // CHECK-SAME: framework = "Metal"
  "tessera_apple.gpu.mps_dispatch"() {
    ordinal = 0 : i64,
    queue = "MTLCommandQueue",
    framework = "Metal",
    execution_mode = "metal_runtime"
  } : () -> ()
}
