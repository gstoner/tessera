// RUN: tessera-opt --tessera-emit-nvvm %s | FileCheck %s
// RUN: tessera-opt --tessera-emit-rocdl %s | FileCheck %s --check-prefix=ROCDL
//
// Phase 4 GPU emission (2026-06-17): a tessera elementwise kernel lowers through
// the linalg spine to NVVM (NVIDIA) and ROCDL (AMD) IR text — tessera.add →
// linalg → one-shot-bufferize → scf.parallel → gpu.launch → outlined gpu.func →
// NVVM/ROCDL. This retargets the tessera→linalg codegen spine (the executed CPU
// JIT lane's front) to the GPU; the two backends share the identical recipe and
// differ only in the final gpu.module(convert-gpu-to-{nvvm,rocdl}) step.
//
// EMISSION ONLY: the GPU kernel is produced for inspection/codegen. GPU launch
// (cuLaunchKernel / hipLaunchKernel) is hardware-gated and not exercised here.

// CHECK: gpu.module
// CHECK: llvm.func
// CHECK-SAME: nvvm.kernel
// CHECK: nvvm.read.ptx.sreg.ctaid.x

// ROCDL: gpu.module
// ROCDL: llvm.func
// ROCDL-SAME: rocdl.kernel
// ROCDL: rocdl.workgroup.id.x
func.func @ew(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "tessera.add"(%a, %b) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
