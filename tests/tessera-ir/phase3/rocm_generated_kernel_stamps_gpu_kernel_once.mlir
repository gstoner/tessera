// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --pass-pipeline='builtin.module(generate-rocm-unary-kernel,gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts))' %s | FileCheck %s
//
// A generated kernel carries `gpu.kernel` exactly once. Every ROCm (and the
// NVIDIA Philox) generator used to stamp it by raw attribute name on top of the
// `kernel` property LLVM 23's gpu.func already owns, so the in-memory op held
// it twice and GPU-to-LLVM copied both onto the llvm.func. A text round-trip
// collapses the pair, which is why upstream mlir-opt on the printed IR never
// saw it and the NDEBUG fleet ran green; the runtime's single-invocation
// pipelines do not round-trip, and on the assertions-ON driver (Tajasarus) they
// aborted with "DictionaryAttr element names must be unique" -- the unary,
// binary and loss families, 50 red tests, invisible on every other host.
// This fixture runs the same single invocation, so a duplicate shows up as a
// printed pair even where no assertion would fire.

module {
  "tessera_rocm.unary"() {name = "u", kind = "exp", dtype = "f32"} : () -> ()
}

// CHECK:       llvm.func @u(
// CHECK-SAME:    attributes {gpu.kernel, rocdl.kernel}
// CHECK-NOT:     gpu.kernel, gpu.kernel
