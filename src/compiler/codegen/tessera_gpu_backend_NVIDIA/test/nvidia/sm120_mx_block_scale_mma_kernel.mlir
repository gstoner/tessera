// RUN: %tnv --lower-tessera-nvidia-to-nvvm %s | FileCheck %s

// Register-level Target IR proofs for the SM120 family block-scaled formats.
// The packed-memory and scale-view materializers intentionally remain outside
// this contract until their Tile/runtime ABI is execution-proven.
module {
  llvm.func @tessera_sm120_mxfp6_e2m3(
      %a0: i32, %a1: i32, %a2: i32, %a3: i32,
      %b0: i32, %b1: i32, %c0: f32, %c1: f32, %c2: f32, %c3: f32,
      %scale_a: i32, %scale_b: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
    %result = tessera_nvidia.mx_block_scale_mma
        %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3, %scale_a, %scale_b
        {arch = "sm_120a", shape = "m16n8k32", dtype_ab = "e2m3",
         dtype_c = "f32", scale_a = "sfa", scale_b = "sfb",
         scale_dtype = "ue8m0", scale_vector = "1X", block_scaled = true}
        : (i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, i32, i32)
          -> !llvm.struct<(f32, f32, f32, f32)>
    llvm.return %result : !llvm.struct<(f32, f32, f32, f32)>
  }

  llvm.func @tessera_sm120_mxfp6_e3m2(
      %a0: i32, %a1: i32, %a2: i32, %a3: i32,
      %b0: i32, %b1: i32, %c0: f32, %c1: f32, %c2: f32, %c3: f32,
      %scale_a: i32, %scale_b: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
    %result = tessera_nvidia.mx_block_scale_mma
        %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3, %scale_a, %scale_b
        {arch = "sm_120a", shape = "m16n8k32", dtype_ab = "e3m2",
         dtype_c = "f32", scale_a = "sfa", scale_b = "sfb",
         scale_dtype = "ue8m0", scale_vector = "1X", block_scaled = true}
        : (i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, i32, i32)
          -> !llvm.struct<(f32, f32, f32, f32)>
    llvm.return %result : !llvm.struct<(f32, f32, f32, f32)>
  }

  llvm.func @tessera_sm120_mxfp4_e2m1(
      %a0: i32, %a1: i32, %a2: i32, %a3: i32,
      %b0: i32, %b1: i32, %c0: f32, %c1: f32, %c2: f32, %c3: f32,
      %scale_a: i32, %scale_b: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
    %result = tessera_nvidia.mx_block_scale_mma
        %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3, %scale_a, %scale_b
        {arch = "sm_120a", shape = "m16n8k64", dtype_ab = "e2m1",
         dtype_c = "f32", scale_a = "sfa", scale_b = "sfb",
         scale_dtype = "ue8m0", scale_vector = "2X", block_scaled = true}
        : (i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, i32, i32)
          -> !llvm.struct<(f32, f32, f32, f32)>
    llvm.return %result : !llvm.struct<(f32, f32, f32, f32)>
  }
}

// CHECK-LABEL: llvm.func @tessera_sm120_mxfp6_e2m3
// CHECK: kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m3.e2m3.f32.ue8m0
// CHECK-LABEL: llvm.func @tessera_sm120_mxfp6_e3m2
// CHECK: kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e3m2.f32.ue8m0
// CHECK-LABEL: llvm.func @tessera_sm120_mxfp4_e2m1
// CHECK: kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0
// CHECK-NOT: tessera_nvidia.mx_block_scale_mma
