// SD1 — GenerateROCMSpecAcceptKernel lowers a tessera.spec_accept (greedy
// longest-prefix path + bonus) to one cooperative-workgroup gpu.func: thread p
// computes its path's run of leading draft==target matches into LDS, a barrier,
// then thread 0 argmaxes the longest prefix (first wins ties) and reads the bonus
// target[path, length]. On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_spec_accept_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-spec-accept-kernel \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @f
// kernel ABI (DRAFT, TARGET, OUT : memref<?xi32>); per-path match-length loop
// (cmpi eq + select) into LDS, barrier, then the argmax (cmpi sgt + select) and
// the three OUT stores [path, length, bonus].
// CHECK:       gpu.func @tessera_spec_accept_0(%{{.*}}: memref<?xi32>, %{{.*}}: memref<?xi32>, %{{.*}}: memref<?xi32>) workgroup({{.*}}memref<256xi32, #gpu.address_space<workgroup>>) kernel
// CHECK:         scf.for
// CHECK:           arith.cmpi eq
// CHECK:           arith.select
// CHECK:         gpu.barrier
// CHECK:         arith.cmpi sgt
// CHECK:         memref.store
// CHECK:         gpu.return
func.func @f(%d: tensor<3x4xi32>, %t: tensor<3x5xi32>) -> tensor<3xi32> {
  %r = "tessera.spec_accept"(%d, %t) : (tensor<3x4xi32>, tensor<3x5xi32>) -> tensor<3xi32>
  return %r : tensor<3xi32>
}

// -----
// ─── more paths than a single workgroup holds (P > 256) verifies but exceeds
// ─── the one-workgroup envelope — left untouched (a future multi-workgroup
// ─── lowering / the guard), never an unlaunchable kernel ────────────────────
// CHECK-LABEL: func.func @g
// CHECK:       tessera.spec_accept
// CHECK-NOT:   gpu.func
func.func @g(%d: tensor<257x4xi32>, %t: tensor<257x5xi32>) -> tensor<3xi32> {
  %r = "tessera.spec_accept"(%d, %t) : (tensor<257x4xi32>, tensor<257x5xi32>) -> tensor<3xi32>
  return %r : tensor<3xi32>
}
