// RUN: %trop --lower-tessera-kernel-abi %s | FileCheck %s

module {
  func.func @my_kernel(%arg0: memref<*xf32>) attributes {tessera_rocm.kernel = "true"} {
    return
  }
}

// CHECK: amdgpu-flat-work-group-size
