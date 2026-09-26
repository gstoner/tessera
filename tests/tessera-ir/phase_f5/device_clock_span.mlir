// RUN: tessera-opt "--tessera-device-clock-span=backend=rocm" %s | FileCheck %s
// RUN: tessera-opt "--tessera-device-clock-span=backend=nvidia" %s | FileCheck %s --check-prefix=NV
//
// Kernel-side timing witness (sync WSL-TIMING-ADMISSION-2026-09-26): the pass
// appends a span-buffer argument and stamps the kernel's span with the
// target's constant-rate device clock -- start by atomic umin from the block
// leader, then a barrier; a barrier, then the end by atomic umax. The
// barriers keep a multi-wave block from stamping its end while a wave still
// computes (an under-estimate that would make a kernel look fast).

module attributes {gpu.container_module} {
  gpu.module @m {
    // CHECK-LABEL: gpu.func @scale
    // CHECK-SAME: %[[A:.*]]: !llvm.ptr<1>, %[[N:.*]]: index, %[[SPAN:.*]]: !llvm.ptr<1>) kernel
    // CHECK-SAME: tessera.device_clock_span = {argument = 2 : i64, backend = "rocm", clock = "llvm.readsteadycounter"}
    // CHECK: scf.if
    // CHECK: %[[T0:.*]] = llvm.call_intrinsic "llvm.readsteadycounter"() : () -> i64
    // CHECK: llvm.atomicrmw umin %[[SPAN]], %[[T0]] monotonic
    // CHECK: gpu.barrier
    // CHECK: llvm.load %[[A]]
    // CHECK: gpu.barrier
    // CHECK: %[[T1:.*]] = llvm.call_intrinsic "llvm.readsteadycounter"() : () -> i64
    // CHECK: %[[END:.*]] = llvm.getelementptr %[[SPAN]][1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
    // CHECK: llvm.atomicrmw umax %[[END]], %[[T1]] monotonic
    // CHECK: gpu.return
    // NV: tessera.device_clock_span = {argument = 2 : i64, backend = "nvidia", clock = "llvm.nvvm.read.ptx.sreg.globaltimer"}
    // NV-COUNT-2: llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
    gpu.func @scale(%a: !llvm.ptr<1>, %n: index) kernel {
      %tid = gpu.thread_id x
      %lt = arith.cmpi ult, %tid, %n : index
      scf.if %lt {
        %v = llvm.load %a : !llvm.ptr<1> -> f32
        %w = arith.addf %v, %v : f32
        llvm.store %w, %a : f32, !llvm.ptr<1>
      }
      gpu.return
    }
  }
}
