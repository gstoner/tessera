// RUN: tessera-opt --tessera-autodiff-paired=export-product=forward %s | tessera-opt --tessera-autodiff-paired | FileCheck %s
//
// An exported product must survive a second run of the paired pass untouched.
// The JIT runs this pass unconditionally on everything it compiles, so a
// forward product that still read as a reverse-mode *request* was
// differentiated again, its residual tapes materialized twice, and its ABI no
// longer matched the contract the caller had read -- a segfault at the first
// invoke on every host with the native x86 JIT. The product now carries
// `tessera.autodiff.role = "forward"`, the same marker the backward already
// carried, and the pass leaves it alone.

module {
  func.func @shrink(%x: tensor<8xf32>) -> tensor<8xf32> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.mul"(%x, %x) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// CHECK-LABEL: {{^ *}}func.func @shrink(
// CHECK-SAME:    tessera.autodiff.role = "forward"
// The second run must not pair it again: exactly one function, no backward.
// CHECK-NOT:   func.func @shrink__bwd
// CHECK-NOT:   tessera.autodiff.paired
