// RUN: tessera-opt "--tessera-device-clock-span=backend=rocm" -split-input-file -verify-diagnostics %s
// RUN: not tessera-opt "--tessera-device-clock-span" %s 2>&1 | FileCheck %s --check-prefix=NOBACKEND
//
// The pass fails closed: a barrier is only safe where every thread reaches it,
// so a kernel whose return is not the single final terminator is refused
// rather than instrumented into a deadlock; the clock is a semantic key and is
// never defaulted; and an image is instrumented once.
// NOBACKEND: TESSERA_DEVICE_CLOCK_BACKEND

module attributes {gpu.container_module} {
  gpu.module @m {
    // expected-error @below {{TESSERA_DEVICE_CLOCK_UNSTRUCTURED}}
    gpu.func @early_return(%a: !llvm.ptr<1>, %n: index) kernel {
      %tid = gpu.thread_id x
      %lt = arith.cmpi ult, %tid, %n : index
      llvm.cond_br %lt, ^work, ^done
    ^work:
      gpu.return
    ^done:
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @m {
    // expected-error @below {{TESSERA_DEVICE_CLOCK_ALREADY_INSTRUMENTED}}
    gpu.func @twice(%a: !llvm.ptr<1>) kernel attributes {tessera.device_clock_span = {argument = 1 : i64, backend = "rocm", clock = "llvm.readsteadycounter"}} {
      gpu.return
    }
  }
}

// -----

// expected-error @below {{TESSERA_DEVICE_CLOCK_NO_KERNEL}}
module attributes {gpu.container_module} {
  gpu.module @m {
    gpu.func @helper(%a: !llvm.ptr<1>) {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @m {
    gpu.func @alloca_after_work(%a: !llvm.ptr<1>) kernel {
      // expected-error @below {{TESSERA_DEVICE_CLOCK_ALLOCA_AFTER_WORK}}
      %v = llvm.load %a : !llvm.ptr<1> -> f32
      %one = llvm.mlir.constant(1 : i64) : i64
      %s = llvm.alloca %one x f32 : (i64) -> !llvm.ptr
      llvm.store %v, %s : f32, !llvm.ptr
      gpu.return
    }
  }
}
