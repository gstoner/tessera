// RUN: tessera-opt -split-input-file -verify-diagnostics %s

// MoeSwigluBlockOp::verify rejects malformed SwiGLU-MoE blocks at the IR level.

func.func @bad_kind(%x: tensor<256x64xbf16>, %wg: tensor<3x64x32xbf16>,
                    %wu: tensor<3x64x32xbf16>, %wd: tensor<3x32x16xbf16>,
                    %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  // expected-error @+1 {{grouped_kind must be one of {dense, contiguous, masked, k_grouped}}}
  %o = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs {grouped_kind = "bogus"}
       : (tensor<256x64xbf16>, tensor<3x64x32xbf16>, tensor<3x64x32xbf16>,
          tensor<3x32x16xbf16>, tensor<3xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}

// -----

func.func @bad_proj_k(%x: tensor<256x64xbf16>, %wg: tensor<3x32x32xbf16>,
                      %wu: tensor<3x64x32xbf16>, %wd: tensor<3x32x16xbf16>,
                      %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  // expected-error @+1 {{contracting dim mismatch: x K vs w_gate/w_up K}}
  %o = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
       : (tensor<256x64xbf16>, tensor<3x32x32xbf16>, tensor<3x64x32xbf16>,
          tensor<3x32x16xbf16>, tensor<3xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}

// -----

func.func @bad_hidden(%x: tensor<256x64xbf16>, %wg: tensor<3x64x32xbf16>,
                      %wu: tensor<3x64x32xbf16>, %wd: tensor<3x99x16xbf16>,
                      %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  // expected-error @+1 {{hidden-dim mismatch: w_gate F / w_up F / w_down K}}
  %o = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
       : (tensor<256x64xbf16>, tensor<3x64x32xbf16>, tensor<3x64x32xbf16>,
          tensor<3x99x16xbf16>, tensor<3xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}
