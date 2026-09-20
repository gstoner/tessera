// RUN: tessera-opt %s | FileCheck %s
// REQUIRES: tessera-clifford

// Clifford-core IR-visible fixture.
//
// Pins the compiler-facing substrate a GA library needs for batched
// multivector workloads, in the dialect's own spellings:
//   * tessera_clifford.rotor_sandwich — R x R† (the equivariance primitive)
//   * tessera_clifford.geo_product    — full Clifford product
//   * tessera_clifford.grade          — pick a grade subset
//   * tessera_clifford.inner          — the scalar invariant ⟨a, a⟩
//   * tessera_clifford.rotor_from_axis — the rotor constructor
//   * tessera_clifford.exp            — exp of a bivector (sampling on the group)
//
// Rewritten 2026-09-16: this fixture predated the Clifford dialect and used
// four op names it never declared (`rotor_from_axis`, `geometric_product`,
// `grade_projection`, `norm_squared`), parsed only because
// `--allow-unregistered-dialect` covered the whole dialect. Registering
// `tessera_clifford` in tessera-opt (the EBM/Clifford driver registration)
// made those names hard errors, since MLIR refuses an unknown op inside a
// REGISTERED dialect whatever that flag says.
//
// Amended later the same day: rotor construction was recorded here as
// "genuinely absent from the dialect", with the rotor arriving as an argument
// so the gap stayed visible. It is no longer absent — `rotor_from_axis` and
// `exp`/`log` now lower to their closed forms on Cl(3, 0) — so the rotor is
// built in IR from an axis and an angle, which is what this fixture wanted to
// pin in the first place.
//
// Shapes use Cl(3, 0)'s 8-blade layout: a batch of 8 multivectors is
// `tensor<8x8xf32>`, and a per-multivector scalar is `tensor<8xf32>`.

// CHECK-LABEL: func @clifford_core_block
// CHECK-DAG:   tessera_clifford.rotor_sandwich
// CHECK-DAG:   tessera_clifford.geo_product
// CHECK-DAG:   tessera_clifford.grade
// CHECK-DAG:   tessera_clifford.inner
// CHECK-DAG:   tessera_clifford.rotor_from_axis
// CHECK-DAG:   tessera_clifford.exp
// CHECK-DAG:   algebra [3, 0, 0]
// CHECK-DAG:   grades [2]

func.func @clifford_core_block(
    %x: tensor<8x8xf32>,
    %axis: tensor<8x8xf32>,
    %tangent: tensor<8x8xf32>
) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>) {

  // Build the rotor from a bivector axis and a fixed angle: the deterministic
  // constructor. The angle is an attribute, so both transcendentals fold.
  %rotor = "tessera_clifford.rotor_from_axis"(%axis) {
      algebra = [3, 0, 0], dtype = "f32", angle = 0.78539816339744828 : f64
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  // The other way onto the group: exponentiate a tangent bivector. This is
  // rotor *sampling* — the tangent is projected to grade 2 first, which is what
  // lets the lowering prove the closed form applies.
  %tangent_bivector = "tessera_clifford.grade"(%tangent) {
      grades = [2], algebra = [3, 0, 0], dtype = "f32"
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %sampled = "tessera_clifford.exp"(%tangent_bivector) {
      algebra = [3, 0, 0], dtype = "f32", terms = 24 : i64
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

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

  return %composed, %bivec_part, %nsq, %sampled
      : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>
}
