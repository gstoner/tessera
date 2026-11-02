// REQUIRES: tessera-rocm-backend
// RUN: %if rocm-device-libs %{ tessera-opt %s --pass-pipeline='builtin.module(tessera-rocm-executable{family=attention_backward input=graph output=binary arch=gfx1151 tile-q=16 tile-kv=16})' --mlir-print-ir-after=convert-gpu-to-rocdl --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: %}
//
// Math lowers to OCML in the executable pipeline however it is entered. The
// same soft-capped attention backward Target IR lowered `math.tanh` to
// `__ocml_tanh_f32` from directive input but to `llvm.intr.tanh` from this
// Graph input (2026-11-01). On gfx1151 with LLVM 23.1.1 that intrinsic
// returns 0 for every input, and every soft-cap gradient came back NaN.
// The backend reported it ("error: no libcall available for ftanh") but the
// compile still exited 0, so the stderr check below also fails on any error.
// The forward recompute and three backward kernels each carry 8 soft-caps.

// CHECK-LABEL: IR Dump After ConvertGpuOpsToROCDLOps
// CHECK-NOT: llvm.intr.tanh
// CHECK-COUNT-32: llvm.call @__ocml_tanh_f32
// CHECK-NOT: llvm.intr.tanh

module {
  func.func @F(
      %q: tensor<1x1x16x64xf16>,
      %key: tensor<1x1x16x64xf16>,
      %v: tensor<1x1x16x64xf16>,
      %bias: tensor<1x1x16x16xf32>
  ) -> tensor<1x1x16x64xf32>
      attributes {tessera.vjp = @B, tessera.lse_checkpoint = "recompute"} {
    %o = "tessera.flash_attn"(%q, %key, %v, %bias)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> {
      causal = true,
      dropout_p = 0.0 : f64,
      dropout_seed = 37 : i64,
      head_dim = 64 : i64,
      scale = 0.125 : f32,
      softcap = 8.0 : f32,
      tessera.tile_q = 16 : i32,
      tessera.tile_kv = 16 : i32,
      window_left = 8 : i64,
      window_right = 0 : i64
    } : (tensor<1x1x16x64xf16>,
          tensor<1x1x16x64xf16>,
          tensor<1x1x16x64xf16>, tensor<1x1x16x16xf32>)
          -> tensor<1x1x16x64xf32>
    return %o : tensor<1x1x16x64xf32>
  }

  func.func @B(
      %do: tensor<1x1x16x64xf16>,
      %q: tensor<1x1x16x64xf16>,
      %key: tensor<1x1x16x64xf16>,
      %v: tensor<1x1x16x64xf16>,
      %bias: tensor<1x1x16x16xf32>
  ) -> (tensor<1x1x16x64xf32>,
        tensor<1x1x16x64xf32>,
        tensor<1x1x16x64xf32>)
      attributes {tessera.primal = @F, tessera.lse_checkpoint = "recompute"} {
    %dq, %dk, %dv_out = "tessera_attn.backward"(
        %do, %q, %key, %v, %bias) {
      causal = true,
      dropout_p = 0.0 : f32,
      dropout_seed = 37 : i64,
      key_block = 16 : i64,
      query_block = 16 : i64,
      scale = 0.125 : f32,
      softcap = 8.0 : f32,
      split_count = 2 : i64,
      window_left = 8 : i64,
      window_right = 0 : i64
    } : (tensor<1x1x16x64xf16>,
          tensor<1x1x16x64xf16>,
          tensor<1x1x16x64xf16>,
          tensor<1x1x16x64xf16>,
          tensor<1x1x16x16xf32>)
          -> (tensor<1x1x16x64xf32>,
              tensor<1x1x16x64xf32>,
              tensor<1x1x16x64xf32>)
    return %dq, %dk, %dv_out : tensor<1x1x16x64xf32>,
                                tensor<1x1x16x64xf32>,
                                tensor<1x1x16x64xf32>
  }
}
