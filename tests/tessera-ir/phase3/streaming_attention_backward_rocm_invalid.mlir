// The ROCm adapter must fail closed rather than infer a physical package from
// an incomplete subset of the shared backward phase bodies.
//
// REQUIRES: tessera-rocm-backend
// RUN: not tessera-opt --lower-tile-to-rocm='arch=gfx1151' \
// RUN:   --allow-unregistered-dialect %s 2>&1 | FileCheck %s

func.func @incomplete_backward(
    %do: tensor<1x2x17x64xf16>,
    %q: tensor<1x2x17x64xf16>,
    %key: tensor<1x1x19x64xf16>,
    %v: tensor<1x1x19x64xf16>,
    %bias: tensor<1x2x17x19xf32>)
    -> (tensor<1x2x17x64xf32>, tensor<1x1x19x64xf32>,
        tensor<1x1x19x64xf32>) {
  %c0 = arith.constant 0 : index
  %dq0 = arith.constant dense<0.000000e+00> : tensor<1x2x17x64xf32>
  %dq = "tessera_attn.backward_dq_block"(
      %do, %q, %key, %v, %bias, %c0, %c0, %c0, %c0, %dq0)
      : (tensor<1x2x17x64xf16>, tensor<1x2x17x64xf16>,
         tensor<1x1x19x64xf16>, tensor<1x1x19x64xf16>,
         tensor<1x2x17x19xf32>, index, index, index, index,
         tensor<1x2x17x64xf32>) -> tensor<1x2x17x64xf32>
  %dk = arith.constant dense<0.000000e+00> : tensor<1x1x19x64xf32>
  %dv = arith.constant dense<0.000000e+00> : tensor<1x1x19x64xf32>
  return %dq, %dk, %dv : tensor<1x2x17x64xf32>,
                          tensor<1x1x19x64xf32>,
                          tensor<1x1x19x64xf32>
}

// CHECK: error: ROCm canonical attention backward requires exactly one dQ block,
// CHECK-SAME: one split dK/dV block, and one fixed-order reduction body
