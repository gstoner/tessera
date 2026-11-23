---
audit_role: plan
plan_state: open
owner: Compiler foundation (correctness tooling)
last_updated: 2026-11-22
---

# Lean 4 validation pilot: three theorems before any broader commitment

Discovery: [compiler map](README.md). Global sequencing defers to
[INTEGRATED_COMPILER_PLAN.md](INTEGRATED_COMPILER_PLAN.md); this plan owns only
the pilot's scope, its acceptance contract and the go/no-go criteria for any
wider use of Lean. It proposes no competing compiler priority.

Synchronization key: `LEAN-PILOT-1` (proposed; not a live-queue record).

## 1. Question and answer

*Should Tessera add a Lean 4 foundation to drive the mathematical correctness of
optimizations, operators and math functions?* — **Yes, as a narrow pilot first.**
The pilot exists to measure, on real Tessera definitions, what a proof costs to
write and keep current, and whether a proof can be tied to the code it is about
tightly enough that it fails when the code changes. A broader commitment waits
on that measurement (§5).

## 2. What a proof here is, and is not

The pilot proves statements over Lean's `ℝ` (Mathlib reals) and `Nat`. That
fixes what it can claim, under CLAUDE.md's claim-integrity rule:

* **Not a proof about float codegen.** IEEE-754 rounding, overflow, `inf`/`nan`,
  fast-math, and fixed-width integer overflow are outside every theorem. A real
  identity can hold while its float evaluation is wrong (the reason the kernels
  subtract the max at all).
* **Not a proof about generated code.** No theorem says anything about a
  `tessera-opt` pass, a ROCm/NVIDIA/Apple/x86 kernel, or a runtime symbol. The
  link to code is a *model*: a Lean definition written to match one cited line.
* **Not a status change.** No primitive-coverage axis, backend manifest row or
  generated dashboard reads these proofs, and none should until §5 says so;
  Decision #24's registry and the device ladders are unchanged.

Every theorem file opens with a comment naming the modeled source line and a
"What this does NOT cover" list.

## 3. Scope — exactly three theorems

| # | Lean file | Modeled Tessera definition | Statement (over ℝ / Nat) |
|---|---|---|---|
| 1 | `formal/lean/TesseraFormal/VJPExp.lean` | `vjp_exp` in `python/tessera/autodiff/vjp.py` (`dout * exp(x)`) | `vjpExp dout x = dout * deriv exp x`; `HasDerivAt exp (vjpExp 1 x) x` |
| 2 | `formal/lean/TesseraFormal/PagedKVSlotMap.lean` | `logicalPage = pos / pageSize`, `pageOffset = pos % pageSize`, slot `table[page] * pageSize + offset` in `GenerateROCMPagedKVReadKernel.cpp`; the `NativePagedKV.h` contract; `ar_decode.paged_kv_pages` layout | `pos < LP*PS →` page `< LP`, offset `< PS`; `(page, offset)` determines `pos`; with table entries `< P` the physical slot is `< P*PS`; with an injective table, distinct positions get distinct slots |
| 3 | `formal/lean/TesseraFormal/SoftmaxShift.lean` | `p = exp(s - m_new)` / `O = acc / l` in `GenerateWMMAFlashAttnKernel.cpp` | `softmax (x - c) = softmax x` for every real `c` |

Finding recorded by theorem 2: the paged-KV contract declares only table
*bounds* (`runtime_checked_physical_page_indices`); table injectivity is neither
declared nor checked. A read-only gather does not need it; a future paged write
would, and `physical_injective` states the lemma it would require.

## 4. Mechanism and acceptance

* **Project.** `formal/lean/` — Lean `v4.36.0`, Mathlib `v4.36.0`, exact commits
  in `lake-manifest.json`. Excluded from every CMake target; `.lake/` is
  gitignored. Theorem 2 imports nothing from Mathlib.
* **Consumer (Decision #29).** `tests/unit/test_formal_lean_pilot.py`:
  * always runs (any host, no Lean): each `tessera-anchor: <path> :: <text>`
    line in a theorem file must occur verbatim **exactly once** in `<path>`
    (a literal `\n` spans lines, so a formula is tied to its owning function,
    not to an identical line elsewhere), so changing or moving a modeled line fails the unit lane instead of leaving a proof about
    code that no longer exists; files must carry their non-claims and no
    `sorry`/`admit`/`axiom`/`native_decide`; no CMake file may reference
    `formal`;
  * with `lake` and the Mathlib cache present: `lake build` must succeed and
    `scripts/AxiomAudit.lean` must show only `propext`, `Classical.choice`,
    `Quot.sound`. Otherwise these two tests **skip with the reason** — a skip is
    "this host did not check the proofs", never a pass (Decision #26).
* **Evidence at landing.** Built on the Mac (macOS 27, Apple M1 Max) on
  2026-11-22: `lake build` completed (all three files), axiom audit clean,
  10/10 gate tests passed; with `lake` off PATH the two toolchain tests skipped
  and 8 passed. No CI lane installs Lean, so CI evaluates only the anchor half.

Done for this plan = the three theorems above, the gate, and §5's measurement
written up after the anchors have survived some real code churn.

## 5. Go / no-go for anything wider

Record these before proposing more Lean work; move this plan to `landing` while
measuring and `closed` (archived) with the verdict.

1. **Anchor churn.** How often did an anchor fail on unrelated refactors versus
   real semantic changes? Frequent false alarms mean the model link is too
   textual and needs a different mechanism (e.g. an IR-level extracted spec).
2. **Cost.** Author time per theorem, Mathlib bump cost, CI cost if a Lean lane
   is added (cache size ~5 GB; Mathlib's cache is pinned to its toolchain).
3. **Did a proof find anything?** Theorem 2 already surfaced the undeclared
   table-injectivity assumption. A pilot that finds nothing is weak evidence for
   expansion.
4. **Float gap.** Whether the next proofs should target float semantics
   (e.g. via an IEEE model or error bounds) rather than more real identities —
   the class of bugs Tessera actually hit (`llvm.tanh` = 0 on AMDGPU, fast-math
   `tanh` NaN) lives there, not in ℝ.

Candidate follow-ons, **not committed**: the online-softmax recurrence
(`corr = exp(m_old - m_new)` rescaling reproduces one-pass softmax); a VJP rule
with broadcasting (`mul` + `_sum_to_shape`); CuTe-style layout composition
(`TileToROCM.cpp` composed-layout rem/div chain); a registry-driven sweep of
VJP/JVP rules if the per-rule cost is low.

## 6. Cross-backend assessment

No compiler, runtime, IR, ABI, dtype or diagnostic surface changes.
Apple / NVIDIA / ROCm / x86: **not applicable** — the pilot adds no backend code
and promotes nothing; theorem 2 and 3 *cite* ROCm generator lines as models only
and make no claim about the SM120 or Apple siblings.
