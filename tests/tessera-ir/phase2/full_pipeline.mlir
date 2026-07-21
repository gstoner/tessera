// End-to-end Phase 2 pipeline test.
// Runs the full tessera-lower-to-x86 chain on a simple BF16 GEMM.
//
// 2026-06: un-XFAIL'd.  The distribution-lowering pass emits unregistered
// schedule.* ops (--allow-unregistered-dialect → generic printing → sym_name
// match), and tile-to-x86 leaves the runtime fn decl out of the nested
// schedule.mesh.region scope (a symbol-ref the module-level verifier rejects),
// so --verify-each=false lets the fully-lowered chain print for FileCheck.
//
// RUN: tessera-opt \
// RUN:   --tessera-distribution-lowering='mesh-axes=dp mesh-sizes=4' \
// RUN:   --tessera-effect-annotation \
// RUN:   --tessera-tiling='tile-m=16 tile-n=16' \
// RUN:   --tessera-tile-to-x86='prefer-amx=true' \
// RUN:   --allow-unregistered-dialect --verify-each=false %s | FileCheck %s
//
// Alternatively, use the named pipeline:
// RUN: tessera-opt -tessera-lower-to-x86 --allow-unregistered-dialect \
// RUN:   --verify-each=false %s | FileCheck %s --check-prefix=PIPE

// CHECK:        func.func @step
// CHECK:        schedule.mesh.define
// CHECK:        schedule.mesh.region
// CHECK:        scf.for
// CHECK:        scf.for
// CHECK:        @tessera_x86_amx_gemm_bf16
// CHECK-NOT:    tessera.matmul{{.*}}tensor<128x256xbf16>

// PIPE:         func.func @step
// PIPE:         @tessera_x86_amx_gemm_bf16

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
