// End-to-end Phase 2 pipeline test.
// Runs the full tessera-lower-to-x86 pipeline on a simple BF16 GEMM.
//
// RUN: tessera-opt \
// RUN:   --tessera-distribution-lowering='mesh-axes=dp mesh-sizes=4' \
// RUN:   --tessera-effect-annotation \
// RUN:   --tessera-tiling='tile-m=16 tile-n=16' \
// RUN:   --tessera-tile-to-x86='prefer-amx=true' \
// RUN:   %s | FileCheck %s
//
// Alternatively, use the named pipeline (requires tessera-opt pipeline reg):
// RUN: tessera-opt -tessera-lower-to-x86 %s | FileCheck %s --check-prefix=PIPE

// CHECK-LABEL:  func.func @step
// CHECK-SAME:   tessera.effect = "memory"
// CHECK:        schedule.mesh.define
// CHECK:        schedule.mesh.region
// CHECK:        scf.for
// CHECK:        scf.for
// CHECK:        tessera.matmul
// CHECK:        func.call @tessera_x86_amx_gemm_bf16
// CHECK-NOT:    tessera.matmul{{.*}}tensor<128x256xbf16>

// PIPE-LABEL:   func.func @step

module attributes {tessera.ir.version = "1.0"} {
  func.func @step(
      %W: tensor<256x512xbf16>
            {tessera.effect = "read"},
      %X: tensor<128x256xbf16>
            {tessera.effect  = "read",
             tessera.shard   = {axes = ["dp"], dims = [0], sizes = [4]}},
      %Y: tensor<128x512xf32>
            {tessera.effect = "write"}
  ) -> tensor<128x512xf32> {
    %C = "tessera.matmul"(%X, %W) : (tensor<128x256xbf16>, tensor<256x512xbf16>)
                                    -> tensor<128x512xf32>
    return %C : tensor<128x512xf32>
  }
}
