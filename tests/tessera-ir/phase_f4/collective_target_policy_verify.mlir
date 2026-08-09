// RUN: not %tessera_strict_opt -split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @bad_all_to_all(%x: tensor<7xf32>) -> tensor<7xf32> {
    %future = "tessera_collective.all_to_all"(%x) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "none",
      world_size = 2 : i64
    } : (tensor<7xf32>) -> !tessera_collective.future<tensor<7xf32>>
    %result = "tessera_collective.await"(%future)
      : (!tessera_collective.future<tensor<7xf32>>) -> tensor<7xf32>
    return %result : tensor<7xf32>
  }
}

// CHECK: exchange axis extent must divide evenly by world_size

// -----

module {
  func.func @bad_qos() {
    "tessera_collective.qos.limit"() {max_inflight = 0 : i64} : () -> ()
    return
  }
}

// CHECK: max_inflight must be positive
