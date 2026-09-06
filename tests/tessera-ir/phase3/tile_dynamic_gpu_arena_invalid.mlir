// RUN: tessera-opt %s --split-input-file --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --verify-diagnostics
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic size is not a supported launch-argument expression}}
    gpu.func @thread_sized() kernel {
      %n = gpu.thread_id x
      %a = memref.alloca(%n) : memref<?xf32>
      "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena cannot overlap an existing dynamic shared allocation}}
    gpu.func @existing(%n: index) kernel {
      %shared = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %a = memref.alloca(%n) : memref<?xf32>
      "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
    gpu.func @helper(%n: index) {
      %a = memref.alloca(%n) : memref<?xf32>
      "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
    gpu.func @nested(%n: index) kernel {
      %tid = gpu.thread_id x
      %cond = arith.cmpi eq, %tid, %n : index
      scf.if %cond {
        %a = memref.alloca(%n) : memref<?xf32>
        "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      }
      gpu.return
    }
  }
}
// -----
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  gpu.module @m {
    // expected-error @+1 {{dynamic launch sizing requires a 64-bit host index}}
    gpu.func @index32(%n: index) kernel {
      %a = memref.alloca(%n) : memref<?xf32>
      "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      gpu.return
    }
  }
}
// -----
module {
  func.func @__tessera_shared_bytes_m_conflict() { return }
  gpu.module @m {
    // expected-error @+1 {{launch sizing symbol already exists}}
    gpu.func @conflict(%n: index) kernel {
      %a = memref.alloca(%n) : memref<?xf32>
      "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic size is not a supported launch-argument expression}}
    gpu.func @iteration_size(%n: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %i = %one to %n step %one {
        %a = memref.alloca(%i) : memref<?xf32>
        "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      }
      gpu.return
    }
  }
}
