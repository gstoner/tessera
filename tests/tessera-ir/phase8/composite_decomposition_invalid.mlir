// ODS triage WIRE slice 1: a scaled ntk_rope whose theta cannot carry the
// scale splat is refused with a named diagnostic (Decisions #21 / #21a) -- it
// is never left for no consumer to see, on either route that runs the rewrite.
//
// RUN: %tessera_strict_opt %s --tessera-decompose-composite-ops --verify-diagnostics -o /dev/null
// RUN: %tessera_strict_opt %s --tessera-canonicalize --verify-diagnostics -o /dev/null

func.func @ntk_rope_dynamic_theta(%x: tensor<?x8xf32>, %th: tensor<?x8xf32>) -> tensor<?x8xf32> {
  // expected-error @+1 {{TESSERA_NTK_ROPE_THETA_UNREWRITABLE: tessera.ntk_rope with scale = 2.000000e+00 cannot be rewritten to tessera.rope(x, theta / scale): theta must be statically shaped}}
  %0 = tessera.ntk_rope %x, %th {scale = 2.0 : f64} : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>
}
