// RUN: tessera-opt --tessera-distribution-lowering='mesh-axes=dp mesh-sizes=4' %s | FileCheck %s --check-prefix=OPT
// RUN: tessera-opt --tessera-distribution-lowering='mesh-axes=dp,tp mesh-sizes=4,4' %s | FileCheck %s --check-prefix=MULTI
// RUN: tessera-opt %s | FileCheck %s --check-prefix=NOOP

// ── Test 1 (OPT): single-axis distribution, shard attr stripped ────────────
// OPT-LABEL: func.func @gemm_step
// OPT-NOT:   tessera.shard
// OPT:       schedule.mesh.define
// OPT-SAME:  axis_names
// OPT-SAME:  "dp"
// OPT:       schedule.mesh.region
// OPT-SAME:  axis = "dp"
// OPT:       tessera.matmul
// OPT:       schedule.yield
// OPT:       return

// ── Test 2 (MULTI): two-axis distribution ──────────────────────────────────
// MULTI-LABEL: func.func @gemm_step
// MULTI:        schedule.mesh.define
// MULTI-SAME:   "dp"
// MULTI-SAME:   "tp"
// MULTI:        schedule.mesh.region

// ── Test 3 (NOOP): no pass options → func is unchanged ────────────────────
// NOOP-LABEL:  func.func @gemm_step
// NOOP-NOT:    schedule.mesh.define

module attributes {tessera.ir.version = "1.0"} {
  func.func @gemm_step(
      %A: tensor<128x256xbf16>
            {tessera.shard = {axes = ["dp"], dims = [0], sizes = [4]}},
      %B: tensor<256x512xbf16>
  ) -> tensor<128x512xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<128x256xbf16>, tensor<256x512xbf16>)
                                    -> tensor<128x512xf32>
    return %C : tensor<128x512xf32>
  }
}
