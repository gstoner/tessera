// RUN: tessera-opt --tessera-activation-rematerialization --verify-diagnostics %s
//
// Phase F2 (IR form) — a `tessera.recompute`-tagged op that carries nested
// regions is NOT safely clonable. Rematerializing it would either drop or
// duplicate control-flow side effects, so the pass fails loudly (Decision #21:
// never silently no-op) instead of leaving a stale marker and a wrong memory
// model.

module {
  func.func @remat_bad_region(%x: tensor<4xf32>) -> tensor<4xf32> {
    // expected-error @+1 {{REMAT_NON_CLONABLE:}}
    %a = "test.control_region"() ({
      "test.yield"() : () -> ()
    }) {tessera.recompute} : () -> tensor<4xf32>
    func.return %a : tensor<4xf32>
  }
}
