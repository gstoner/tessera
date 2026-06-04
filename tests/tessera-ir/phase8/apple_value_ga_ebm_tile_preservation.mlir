// Stage 16C — value-mode tiling preserves strict GA/EBM envelopes as registered
// Tile IR, and leaves unsupported variants as Graph IR for named downstream
// diagnostics.
//
// RUN: %tessera_strict_opt %s -tessera-tiling="value-mode=true" | FileCheck %s

// CHECK-LABEL: func.func @ebm_energy_preserved
// CHECK: tile.ebm_energy_quadratic
// CHECK-SAME: source = "tessera.ebm.energy_quadratic"
func.func @ebm_energy_preserved(%x: tensor<2x3xf32>,
                                %y: tensor<2x3xf32>) -> tensor<2xf32> {
  %0 = tessera.ebm.energy_quadratic %x, %y
    : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @ebm_langevin_preserved
// CHECK: tile.ebm_langevin_step
// CHECK-SAME: has_noise = true
// CHECK-SAME: source = "tessera.ebm.langevin_step"
func.func @ebm_langevin_preserved(%y: tensor<2x3xf32>,
                                  %g: tensor<2x3xf32>,
                                  %n: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tessera.ebm.langevin_step %y, %g, %n
    {eta = 1.250000e-01 : f64, noise_scale = 2.500000e-01 : f64}
    : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @ebm_langevin_no_noise_not_preserved
// CHECK: tessera.ebm.langevin_step
// CHECK-NOT: tile.ebm_langevin_step
func.func @ebm_langevin_no_noise_not_preserved(%y: tensor<2x3xf32>,
                                               %g: tensor<2x3xf32>)
                                               -> tensor<2x3xf32> {
  %0 = tessera.ebm.langevin_step %y, %g
    {eta = 1.250000e-01 : f64}
    : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @clifford_preserved
// CHECK: tile.clifford_geometric_product
// CHECK-SAME: has_signature = true
// CHECK-SAME: source = "tessera.clifford.geometric_product"
// CHECK: tile.clifford_grade_project
// CHECK-SAME: has_grade_mask = true
// CHECK-SAME: source = "tessera.clifford.grade_project"
func.func @clifford_preserved(%a: tensor<2x8xf32>,
                              %b: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = tessera.clifford.geometric_product %a, %b
    {p = 3 : i64, q = 0 : i64, signature = [1 : i64, 1 : i64, 1 : i64]}
    : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %1 = tessera.clifford.grade_project %0
    {p = 3 : i64, q = 0 : i64, grade_mask = [1 : i64, 2 : i64]}
    : (tensor<2x8xf32>) -> tensor<2x8xf32>
  return %1 : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @clifford_non_cl30_not_preserved
// CHECK: tessera.clifford.geometric_product
// CHECK-NOT: tile.clifford_geometric_product
func.func @clifford_non_cl30_not_preserved(%a: tensor<2x4xf32>,
                                           %b: tensor<2x4xf32>)
                                           -> tensor<2x4xf32> {
  %0 = tessera.clifford.geometric_product %a, %b
    {p = 2 : i64, q = 0 : i64}
    : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
