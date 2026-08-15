# Game Theory — plan-stage verification harness

Numerical verification of every algebraic claim in
[`docs/audit/compiler/GAME_THEORY_PLAN.md`](../../docs/audit/compiler/GAME_THEORY_PLAN.md),
run 2026-08-15. Plan-stage research material — **not** built, **not** imported
by `python/tessera/`.

| File | What it is |
|---|---|
| `verify_game_theory_plan.py` | The full 27-check audit: butterfly algebra (zeta/Möbius/adjoint), semivalue weights vs. direct Shapley/Banzhaf definitions, the Faigle-draft Banzhaf factor-of-2 erratum, Boltzmann temperature limits, the sign-structure-dependent fp32 wall, the potential-game prefix-scan rewrite + cost bound, extragradient+sparsemax saddle solving with envelope-derivative check, corrected Blum–Mansour swap-regret → CE, core separation oracle on a supermodular game, Hermitian-embed isometry, nim/Grundy. Exits nonzero on any failure. |
| `verify_fixups.py` | The failure-diagnosis run that produced plan §6 (fp32 wall by sign structure) and hazard H2 (two plausible-but-wrong regret dynamics). Kept as provenance. |

Run: `python3 verify_game_theory_plan.py` (needs only numpy).

These reference implementations are the seed of G1's
`tests/unit/test_game_lattice.py`; when G1 lands, the checks here migrate into
the unit suite and this directory becomes provenance only.
