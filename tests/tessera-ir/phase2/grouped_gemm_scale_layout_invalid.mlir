// RUN: tessera-opt -split-input-file -verify-diagnostics %s

// verifyScaleLayoutDict rejects malformed scale-layout contracts at the IR level
// (mirrors python grouped_layout.ScaleLayout.__post_init__).

func.func @bad_granularity(%x: tensor<8x16xf32>, %w: tensor<2x16x32xf32>,
                           %gs: tensor<2xi64>) -> tensor<8x32xf32> {
  // expected-error @+1 {{scale_layout granularity must be one of {per_tensor, per_row, per_channel, block}}}
  %o = tessera.grouped_gemm %x, %w, %gs {scale_layout = {granularity = "bogus"}}
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2xi64>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// -----

func.func @block_missing_shape(%x: tensor<8x16xf32>, %w: tensor<2x16x32xf32>,
                               %gs: tensor<2xi64>) -> tensor<8x32xf32> {
  // expected-error @+1 {{scale_layout block granularity requires a 2-element 'block' shape}}
  %o = tessera.grouped_gemm %x, %w, %gs {scale_layout = {granularity = "block"}}
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2xi64>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// -----

func.func @block_on_per_tensor(%x: tensor<8x16xf32>, %w: tensor<2x16x32xf32>,
                               %gs: tensor<2xi64>) -> tensor<8x32xf32> {
  // expected-error @+1 {{scale_layout 'block' shape is only valid for granularity='block'}}
  %o = tessera.grouped_gemm %x, %w, %gs
         {scale_layout = {granularity = "per_tensor", block = [1, 128]}}
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2xi64>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// -----

func.func @bad_packing(%x: tensor<8x16xf32>, %w: tensor<2x16x32xf32>,
                       %gs: tensor<2xi64>) -> tensor<8x32xf32> {
  // expected-error @+1 {{scale_layout packing must be one of {none, e4m3, e5m2, e8m0, ue8m0}}}
  %o = tessera.grouped_gemm %x, %w, %gs
         {scale_layout = {granularity = "per_tensor", packing = "fp17"}}
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2xi64>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// -----

// The same verifier guards moe_swiglu_block.
func.func @moe_bad_granularity(%x: tensor<8x16xf32>, %wg: tensor<2x16x32xf32>,
    %wu: tensor<2x16x32xf32>, %wd: tensor<2x32x16xf32>, %gs: tensor<2xi64>)
    -> tensor<8x16xf32> {
  // expected-error @+1 {{scale_layout granularity must be one of}}
  %o = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
         {scale_layout = {granularity = "bogus"}}
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2x16x32xf32>,
          tensor<2x32x16xf32>, tensor<2xi64>) -> tensor<8x16xf32>
  return %o : tensor<8x16xf32>
}
