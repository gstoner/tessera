# Core-substrate math verification

Executable backing for
[`docs/audit/compiler/CORE_SUBSTRATE_VIEW.md`](../../docs/audit/compiler/CORE_SUBSTRATE_VIEW.md)
(2026-08-15). `verify_substrate_math.py` machine-checks the 13 load-bearing
mathematical claims the view inherits that were previously **prose-only**: the
CAKE small-sample statistics (exact permutation/Fisher/Holm, and the Phase 4
filter-gate formula) and the TileRT models M1–M5 (bubble decomposition, the
|R|-overlap ceiling and its tightness, the roofline arithmetic, the
selection/composition non-commutation counterexample, Graham's bound with its
tight family, MTP acceptance), plus two closed-form spot checks repeated from
the game-theory plan (memory wall, fp64 digits-gone wall).

Deliberately NOT re-verified here: game theory (27 checks,
`research/game_theory/`), PDE/stencil (78 assertions,
`tests/unit/test_pde_stencil_model.py`), SparDA
(`tests/unit/test_sparda_contracts.py`) — those already have owning harnesses.

```bash
python3 research/core_substrate/verify_substrate_math.py   # 13/13 PASS
```
