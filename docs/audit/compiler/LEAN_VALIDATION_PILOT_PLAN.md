---
audit_role: plan
plan_state: open
owner: Compiler foundation (correctness tooling)
last_updated: 2026-11-23
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

### 5a. Two further directions, examined 2026-11-23 (not committed)

Both were read against the code before being written down; neither widens the
pilot, which stays at three theorems.

**LDS data-race freedom and barrier safety.**

* *What already exists.* `TileBarrierReuseLegalityPass` checks Tile IR
  write/write reuse of `m`/`lds`/`tlane`/`tcol` footprints and async-copy
  completion (typed tokens, `wait_async`, `mbarrier.wait`; a CTA rendezvous does
  not complete DMA). It models program order of ops, not per-thread address
  sets, and a plain buffer *read* is refused as an unsupported access rather than
  checked. The ROCm kernel generators (e.g. `GenerateWMMAFlashAttnKernel.cpp`)
  build `memref` workgroup buffers and `gpu.barrier` directly in C++ and never
  pass through that pass, so their LDS synchronization is checked by nothing but
  device results. Upstream `--gpu-eliminate-barriers` (present in the fleet's
  LLVM 23.1.1) removes barriers using memory effects; it is an optimization, not
  a checker.
* *Proof shape.* Barrier-interval semantics: split a kernel at uniformly reached
  workgroup barriers; a phase is race-free iff no two distinct threads touch the
  same LDS address with at least one write. Async DMA is an access that stays
  live until its completing wait, the rule the pass already encodes. Proven once
  in Lean, this is the lemma that "race-free per phase ⇒ the result does not
  depend on the wave schedule".
* *First instance.* The split-K partial merge in
  `GenerateWMMAFlashAttnKernel.cpp`: each wave stores
  `sPartial[wave*256 + qi*16 + l15]` (with `qi = 2e+half` on RDNA3.5 and
  `e + 8*half` on RDNA4), then `gpu.barrier`, then wave 0 reads `pidx` and
  `256 + pidx`. Statements: the write map is injective (no write/write race),
  the reads are covered by the previous phase's writes (RAW coverage, which is
  what makes the merge *correct*, not only race-free), and every index is below
  the allocation size. Pure `Nat`, no Mathlib.
* *Where it should end up.* The per-kernel facts are affine index maps over
  finite thread grids: a Presburger question that MLIR's own
  `mlir/Analysis/Presburger` can decide inside the compiler on every kernel
  (Decision #30, derive don't ask). The useful split is Lean for the semantic
  rule, proved once, and a compiler analysis for its instances. A Lean proof per
  kernel would not scale and would duplicate that analysis.
* *Non-claims to carry.* It would not cover the lowering of `gpu.barrier` to
  `s_waitcnt` + `s_barrier`, the hardware LDS memory model, divergent barriers
  (assumed absent, not proven), or bank conflicts (a performance property).

**Bounded error with [FloatLib](https://github.com/lean-dojo/FloatLib).**

* *Facts checked (2026-11-23).* MIT-licensed, active (last push 2026-11-20), and
  its `lean-toolchain` is `v4.36.0`, the pilot's own pin. Whether its Mathlib
  commit matches ours is **unverified**, and a mismatch would force a joint bump.
  It models IEEE binary formats with subnormals, signed zeros, NaN, rounding
  modes and flags; FP8/FP6/FP4; and OCP MX and NVFP4 quantizers proved to compute
  the real formats. `float_decide` checks closed float facts in the kernel
  (`decide +kernel`, not `native_decide`, so the gate's escape-hatch ban still
  holds). Its native Half/BFloat16 *arithmetic* is tested, not proved. It is a
  young project: pin a commit, never `@ "main"`.
* *Highest-value first target: block-scaled quantizer conformance.* Model
  Tessera's NVFP4 / MXFP4 / MXFP8 scale selection and element rounding
  (`nvfp4_tensor.py`, `rocm_mxfp4_*`) and prove it equal to FloatLib's OCP MX /
  NVFP4 definitions, or exhibit the inputs where it differs. This is the
  candidate most likely to *find* something (§5 criterion 3): scale-selection
  and saturation rules are where implementations diverge, and Tessera has
  device programs riding on them.
* *Second: tolerances derived, not guessed.* An `accum=fp32` dot-product error
  bound (the standard `γ_n` form), and a binary32 error bound for the
  max-shifted softmax extending theorem 3, with the `exp` ulp bound as an
  explicit *hypothesis*. libdevice / ROCDL / MSL `exp` are not correctly
  rounded, so the bound is only as good as that premise. The consumer
  (Decision #29) would be device tests asserting the derived tolerance instead
  of a hand-picked `rtol`.
* *What it cannot catch.* The float bugs Tessera actually hit (`llvm.tanh`
  computed as 0 on AMDGPU, the fast-math `tanh` gelu NaN) were *lowering* defects.
  A proof about a float model says nothing about whether the device computes
  that model, so these proofs need a differential device test beside them.

## 6. Cross-backend assessment

No compiler, runtime, IR, ABI, dtype or diagnostic surface changes.
Apple / NVIDIA / ROCm / x86: **not applicable** — the pilot adds no backend code
and promotes nothing; theorem 2 and 3 *cite* ROCm generator lines as models only
and make no claim about the SM120 or Apple siblings.
