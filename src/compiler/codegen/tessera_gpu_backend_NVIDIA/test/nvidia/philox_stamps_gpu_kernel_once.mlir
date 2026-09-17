// RUN: %tnv %s --pass-pipeline='builtin.module(generate-nvidia-philox-kernel,gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm,reconcile-unrealized-casts))' | FileCheck %s
//
// The generated Philox kernel carries `gpu.kernel` exactly once. This generator
// stamped it by raw attribute name on top of the `kernel` property LLVM 23's
// gpu.func already owns -- the same defect as the 72 ROCm generators (see
// tests/tessera-ir/phase3/rocm_generated_kernel_stamps_gpu_kernel_once.mlir):
// the in-memory op held the attribute twice and GPU-to-LLVM copied both onto
// the llvm.func. A text round-trip collapses the pair, so a fixture that pipes
// the generator's output through a second invocation never sees it, and the
// NDEBUG fleet ran green while an assertions-ON driver aborts with
// "DictionaryAttr element names must be unique". This fixture keeps the
// generation and the NVVM lowering in one invocation, so a duplicate prints as
// a pair on any driver and aborts on an assertions build.

module {
  "tessera_nvidia.philox"() {name = "philox_uniform", mode = "uniform_core"} : () -> ()
}

// CHECK:       llvm.func @philox_uniform(
// CHECK-SAME:    gpu.kernel
// CHECK-SAME:    nvvm.kernel
// CHECK-NOT:     gpu.kernel, gpu.kernel
