# Tessera formal pilot (Lean 4) — `LEAN-PILOT-1`

A deliberately small Lean 4 + Mathlib project that machine-checks three
mathematical facts Tessera code relies on. Scope, non-claims and the go/no-go
criteria live in
[`docs/audit/compiler/LEAN_VALIDATION_PILOT_PLAN.md`](../../docs/audit/compiler/LEAN_VALIDATION_PILOT_PLAN.md).

| File | Theorem | Models |
|---|---|---|
| `TesseraFormal/VJPExp.lean` | `vjp_exp` = pullback of `exp'` | `python/tessera/autodiff/vjp.py` |
| `TesseraFormal/PagedKVSlotMap.lean` | paged-KV slot map in-bounds + injective | `GenerateROCMPagedKVReadKernel.cpp`, `NativePagedKV.h`, `models/ar_decode.py` |
| `TesseraFormal/SoftmaxShift.lean` | `softmax(x - c) = softmax(x)` | `GenerateWMMAFlashAttnKernel.cpp` online softmax |

**These are proofs over `ℝ` and `Nat`, not about IEEE floats, fixed-width
integers, or any generated kernel.** Each file says exactly what it does not
cover.

This directory is not part of any CMake target.

## Build

```bash
brew install elan-init          # or the elan installer from github.com/leanprover/elan
cd formal/lean
lake exe cache get              # prebuilt Mathlib oleans (~5 GB under .lake/, gitignored)
lake build                      # checks all three theorem files
lake env lean scripts/AxiomAudit.lean   # must list only propext / Classical.choice / Quot.sound
```

`lean-toolchain` and the Mathlib `rev` in `lakefile.toml` are pinned together
(both `v4.36.0`); `lake-manifest.json` pins the exact dependency commits. Bump
them together and re-run the gate.

## Gate

`tests/unit/test_formal_lean_pilot.py` always checks that every
`tessera-anchor:` line still matches its Tessera source verbatim, so a change to
the modeled code fails CI even on hosts without Lean. The `lake build` and axiom
audit tests **skip** when `lake` or the Mathlib cache is absent — a skip means
the host did not check the proofs, never that they passed.
