// RUN: tessera-opt %s --split-input-file --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --verify-diagnostics
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
    gpu.func @missing(%src: memref<?xf32, 1>, %n: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %i = %zero to %n step %one {
        %a = memref.alloca(%n) : memref<?xf32, 3>
        "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()
        %copy = nvgpu.device_async_copy %src[%zero], %a[%zero], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %group = nvgpu.device_async_create_group %copy

        gpu.barrier
      }
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
    gpu.func @partial(%src: memref<?xf32, 1>, %n: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %i = %zero to %n step %one {
        %a = memref.alloca(%n) : memref<?xf32, 3>
        "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()
        %copy = nvgpu.device_async_copy %src[%zero], %a[%zero], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %group = nvgpu.device_async_create_group %copy
        nvgpu.device_async_wait %group {numGroups = 1 : i32}
        gpu.barrier
      }
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
    gpu.func @wrong(%src: memref<?xf32, 1>, %n: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %i = %zero to %n step %one {
        %a = memref.alloca(%n) : memref<?xf32, 3>
        "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()
        %copy = nvgpu.device_async_copy %src[%zero], %a[%zero], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %group = nvgpu.device_async_create_group %copy
        %wrong = nvgpu.device_async_create_group
        nvgpu.device_async_wait %wrong
        gpu.barrier
      }
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
    gpu.func @changing_backedge(%src: memref<?xf32, 1>, %n: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %i = %zero to %n step %one {
        %a = memref.alloca(%n) : memref<?xf32, 3>
        "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()
        %copy = nvgpu.device_async_copy %src[%zero], %a[%zero], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %group = nvgpu.device_async_create_group %copy
        %forwarded = scf.for %j = %zero to %n step %one iter_args(%token = %group) -> !nvgpu.device.async.token {
          %fresh = nvgpu.device_async_create_group
          scf.yield %fresh : !nvgpu.device.async.token
        }
        nvgpu.device_async_wait %forwarded
        gpu.barrier
      }
      gpu.return
    }
  }
}
// -----
module {
  gpu.module @m {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
    gpu.func @mixed_branch(%src: memref<?xf32, 1>, %n: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %i = %zero to %n step %one {
        %a = memref.alloca(%n) : memref<?xf32, 3>
        "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()
        %copy = nvgpu.device_async_copy %src[%zero], %a[%zero], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %group = nvgpu.device_async_create_group %copy
        %condition = arith.cmpi ult, %n, %one : index
        %unrelated = "nvgpu.device_async_create_group"() : () -> !nvgpu.device.async.token
        %selected = scf.if %condition -> !nvgpu.device.async.token {
          scf.yield %group : !nvgpu.device.async.token
        } else {
          scf.yield %unrelated : !nvgpu.device.async.token
        }
        nvgpu.device_async_wait %selected
        gpu.barrier
      }
      gpu.return
    }
  }
}
