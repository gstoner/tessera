// RUN: tessera-opt %s | FileCheck %s

// Clifford-core IR-visible fixture.
//
// Pins the compiler-facing substrate a GA library needs for batched
// multivector workloads, in the dialect's own spellings:
//   * tessera_clifford.rotor_sandwich — R x R† (the equivariance primitive)
//   * tessera_clifford.geo_product    — full Clifford product
//   * tessera_clifford.grade          — pick a grade subset
//   * tessera_clifford.inner          — the scalar invariant ⟨a, a⟩
//
// Rewritten 2026-09-16: this fixture predated the Clifford dialect and used
// four op names it never declared (`rotor_from_axis`, `geometric_product`,
// `grade_projection`, `norm_squared`), parsed only because
// `--allow-unregistered-dialect` covered the whole dialect. Registering
// `tessera_clifford` in tessera-opt (the EBM/Clifford driver registration)
// made those names hard errors, since MLIR refuses an unknown op inside a
// REGISTERED dialect whatever that flag says. Rotor construction from an
// axis and angle is genuinely absent from the dialect — it is not faked
// here; the rotor arrives as an argument and the gap stays visible in the
// GA queue (`docs/audit/backend/rocm/todo.md`, GA family: `exp`/`log` and
// rotor construction).
//
// Shapes use Cl(3, 0)'s 8-blade layout: a batch of 8 multivectors is
// `tensor<8x8xf32>`, and a per-multivector scalar is `tensor<8xf32>`.

// CHECK-LABEL: func @clifford_core_block
// CHECK-DAG:   tessera_clifford.rotor_sandwich
// CHECK-DAG:   tessera_clifford.geo_product
// CHECK-DAG:   tessera_clifford.grade
// CHECK-DAG:   tessera_clifford.inner
// CHECK-DAG:   algebra [3, 0, 0]
// CHECK-DAG:   grades [2]

func.func @clifford_core_block(
    %x: tensor<8x8xf32>,
    %rotor: tensor<8x8xf32>
) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>) {

  // Sandwich-apply the rotor to the input multivectors.
  %sandwich = "tessera_clifford.rotor_sandwich"(%rotor, %x) {
      algebra = [3, 0, 0], dtype = "f32"
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>

  // Geometric-product composition — rotor · x.
  %composed = "tessera_clifford.geo_product"(%rotor, %x) {
      algebra = [3, 0, 0], dtype = "f32"
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>

  // Grade-2 projection of the sandwiched result.
  %bivec_part = "tessera_clifford.grade"(%sandwich) {
      grades = [2], algebra = [3, 0, 0], dtype = "f32"
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  // The scalar invariant ⟨a, a⟩ (the squared norm) of the composition.
  %nsq = "tessera_clifford.inner"(%composed, %composed) {
      algebra = [3, 0, 0], dtype = "f32"
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8xf32>

  return %composed, %bivec_part, %nsq : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>
}
