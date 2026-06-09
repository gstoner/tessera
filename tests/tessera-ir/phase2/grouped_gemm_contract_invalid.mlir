// RUN: tessera-opt -split-input-file -verify-diagnostics %s

// GroupedGemmOp::verify rejects malformed grouped-layout contracts at the IR
// level (Step 3).

func.func @bad_kind(%x: tensor<256x64xbf16>, %w: tensor<3x64x16xbf16>,
                    %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  // expected-error @+1 {{grouped_kind must be one of {dense, contiguous, masked, k_grouped}}}
  %o = tessera.grouped_gemm %x, %w, %gs {grouped_kind = "bogus"}
       : (tensor<256x64xbf16>, tensor<3x64x16xbf16>, tensor<3xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}

// -----

func.func @bad_alignment(%x: tensor<256x64xbf16>, %w: tensor<3x64x16xbf16>,
                         %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  // expected-error @+1 {{grouped_alignment must be a positive power of two}}
  %o = tessera.grouped_gemm %x, %w, %gs {grouped_alignment = 100 : i64}
       : (tensor<256x64xbf16>, tensor<3x64x16xbf16>, tensor<3xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}

// -----

func.func @bad_k(%x: tensor<256x64xbf16>, %w: tensor<3x32x16xbf16>,
                 %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  // expected-error @+1 {{contracting dim mismatch: x K vs weights K}}
  %o = tessera.grouped_gemm %x, %w, %gs
       : (tensor<256x64xbf16>, tensor<3x32x16xbf16>, tensor<3xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}

// -----

func.func @bad_groups(%x: tensor<256x64xbf16>, %w: tensor<3x64x16xbf16>,
                      %gs: tensor<5xi64>) -> tensor<256x16xf32> {
  // expected-error @+1 {{group_sizes length must equal the expert count E}}
  %o = tessera.grouped_gemm %x, %w, %gs
       : (tensor<256x64xbf16>, tensor<3x64x16xbf16>, tensor<5xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}
