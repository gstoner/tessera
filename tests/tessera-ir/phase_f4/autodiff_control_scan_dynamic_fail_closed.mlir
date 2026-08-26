// RUN: not tessera-opt %s --tessera-autodiff-paired 2>&1 | FileCheck %s

// The first relaxed envelope requires a statically shaped carry and stream
// slice. Dynamic storage cannot acquire an implicit tape bound.
module {
  func.func @dynamic_step(%c: tensor<?xf32>, %x: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %n = "tessera.add"(%c, %x) :
        (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %n, %n : tensor<?xf32>, tensor<?xf32>
  }
  func.func @dynamic_scan(%init: tensor<?xf32>, %xs: tensor<3x?xf32>)
      -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {
    %c, %ys = "tessera.control_scan"(%init, %xs) {
      body = @dynamic_step, trip = 3 : i64, carry_arg_index = 0 : i64
    } : (tensor<?xf32>, tensor<3x?xf32>) ->
        (tensor<?xf32>, tensor<3x?xf32>)
    return %c : tensor<?xf32>
  }
}

// CHECK: AUTODIFF_CONTROL_SCAN_UNSUPPORTED
// CHECK-SAME: outside the bounded reverse envelope
