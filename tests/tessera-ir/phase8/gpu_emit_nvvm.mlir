// RUN: tessera-opt --tessera-emit-nvvm %s | FileCheck %s
//
// Phase 4 GPU emission (2026-06-17): a tessera elementwise kernel lowers through
// the linalg spine to NVVM IR text — tessera.add → linalg → one-shot-bufferize →
// scf.parallel → gpu.launch → outlined gpu.func → NVVM. This retargets the
// tessera→linalg codegen spine (the executed CPU JIT lane's front) to the GPU.
//
// EMISSION ONLY: the NVVM kernel is produced for inspection/codegen. GPU launch
// (cuLaunchKernel / hipLaunchKernel) is hardware-gated and not exercised here.

// CHECK: gpu.module
// CHECK: llvm.func
// CHECK-SAME: nvvm.kernel
// CHECK: nvvm.read.ptx.sreg.ctaid.x
func.func @ew(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "tessera.add"(%a, %b) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
