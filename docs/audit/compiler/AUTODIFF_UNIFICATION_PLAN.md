---
last_updated: 2026-07-11
audit_role: plan
plan_state: open
---

# Front-End / IR / Autodiff Unification Plan

> **Status truth lives in the generated dashboards, not this doc** (Decision
> #26). This is *direction + sequencing*. Live counts/states come from
> [`generated/compiler_progress.md`](../generated/compiler_progress.md),
> [`generated/runtime_execution_matrix.md`](../generated/runtime_execution_matrix.md),
> [`standalone_primitive_coverage.md`](../standalone_primitive_coverage.md),
> [`op_target_conformance.md`](../op_target_conformance.md), and the new
> **compiler-autodiff connection ledger**
> [`generated/autodiff_connection_ledger.md`](../generated/autodiff_connection_ledger.md)
> (**landed** Phase 0 — a *projection* over the registries above, **not** a new
> source of truth — see §3).
>
> **Paired with** [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md)
> (north star — three-tier kernel model + measured arbiter) and
> [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) (the backend spine this
> rides on). **Specifies against**
> [`../../spec/AUTODIFF_SPEC.md`](../../spec/AUTODIFF_SPEC.md) (the normative v1
> autodiff surface). **Current-state / Still-Open** in
> [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md).

---

## 1. Goal

Unify the three parallel systems — Python front end, compiler IR, and autodiff —
onto one owned semantic program, so that differentiation is a **compiler
request** with a **native execution path**, not an implicit Python-tape behavior
that is silently reported as if it were compiled.

Target architecture:

```
Python API / AST / typing / effects
        │  (owns validation, source spans, types, shapes, user errors)
        ▼
Graph IR   (single compiler-owned semantic program)
        │
        ▼  differentiate: forward + backward + residual contract
        │
        ▼
Schedule IR → Tile IR → Target IR
        │
        ▼
runtime ABI → native execution
        │
        ▼
oracle comparison against Python VJP/JVP reference
```

Python stays the **interface and reference/oracle** layer. The MLIR/LLVM path
becomes the **production execution** layer. The Python VJP/JVP tape
(`python/tessera/autodiff/`) is the semantic reference — **never** evidence of
native compiler support.

The one-sentence test the whole plan serves: *for a given op family and target,
can a generated report say whether the native compiler executes its **forward**
and its **backward**, and whether each was numerically proven — without reading
source?* Today it cannot. Every phase moves one concrete step toward "yes."

---

## 2. Why now — the current contradiction

The repo reports autodiff status two incompatible ways:

- The non-goals table (§ near line 30) of [`AUTODIFF_SPEC.md`](../../spec/AUTODIFF_SPEC.md)
  says **"✅ Phase F4 landed — verified end-to-end on MLIR 22."**
- The §F4 prose (near line 278) of [`AUTODIFF_SPEC.md`](../../spec/AUTODIFF_SPEC.md) still
  says **"design landed, build follow-up,"** the pass **"registers as a no-op,"**
  and the smoke test is **XFAIL**.

Ground truth sits between the two: the `AdjointInterface` tablegen **is** wired
([`src/compiler/ir/CMakeLists.txt:11-14`](../../../src/compiler/ir/CMakeLists.txt)),
real `buildAdjoint` bodies exist including native (compiler-visible)
tanh/sigmoid adjoints (W5,
[`src/compiler/ir/AdjointInterface.cpp:141-176`](../../../src/compiler/ir/AdjointInterface.cpp)),
**but** `@jit` emits no differentiation intent
([`python/tessera/compiler/jit.py`](../../../python/tessera/compiler/jit.py) has
no `autodiff=` path) and no runtime binds a compiled backward entry point. So
"forward native" and "gradients native" are conflated, and neither has a
per-op-family × per-target proof surface. Phase 0 removes that ambiguity before
any new native adjoint is built.

---

## 2a. This is a migration, not a rebuild

The systems this plan unifies **already exist**. The failure mode to avoid is
building a fourth parallel system (a new autodiff engine, a new audit registry, a
new provenance enum) alongside the three we already have — which is exactly the
"parallel-system problem" the phases warn against. So each surface below has a
defined role in the target architecture; the plan **reclassifies and extends**
these, it does not fork them.

| Existing surface | Role **today** | Role in target architecture | Touched in |
|---|---|---|---|
| `python/tessera/autodiff/{tape,vjp,jvp}.py` (`_VJPS`/`_JVPS`) | Differentiation **engine** *and* oracle | Oracle / reference **only** | P0 reclassify · P3 compare-target |
| `compiler/primitive_coverage.py` (Decision #24: 12 axes incl. `vjp`/`jvp`/`lowering_rule`/`backend_kernel`, auto-flips from `_VJPS`/`_JVPS`) | Audit truth for per-primitive axes | **Stays truth**; the ledger *joins* its autodiff axes — never re-registers them | P0 (read-only join) |
| `AdjointInterface.{td,cpp}` + `AutodiffPass.cpp` (native pointwise + `placeholderAdjoint` fallback) | Reverse-walk pass, unproven end-to-end | The `ir_adjoint` **producer** | P2 hardens contract |
| `@jit` / `CompileResult` (`execution_mode`, `is_native_supported`, `accelerator_proof`) | Forward compile + forward provenance | Extended to carry **backward** provenance, same enum vocabulary | P1 · P4 |
| `generated/runtime_execution_matrix.md` | Per-target **forward** executability truth | Source for `runtime_bound`/`hardware_proven`; gains a **backward** column | P0 (read) · P4 (write) |
| `op_target_conformance` + `execute_compare_fixture` | **Forward** numerical proof | Source for `oracle_proven`; gains **backward** fixtures | P0 (read) · P3 (write) |
| `custom_adjoint_call` bridge (round-trips to Python VJP at runtime) | Runtime escape hatch | Must become a **defined runtime ABI** or be **rejected** under `native_required` | P2 · P4 |

Reading this table top-to-bottom is the plan in one breath: **stop treating the
Python tape as proof, make the pass's output a contract, extend the provenance
and dashboards you already have to cover backward, and gate on the oracle you
already run for forward.**

---

## 3. The status ledger — a *derived projection*, not a new registry

One row per **op family × target**, six rungs, lowest-proven-rung wins. **The
ledger introduces zero new source of truth** — it is generated by *joining*
existing registries, so Decision #24 (`primitive_coverage.py` is the audit truth)
is preserved. Its contribution is the one dimension none of them make explicit
today: **forward vs. backward, per target.**

"Op family" = the existing `primitive_coverage` primitive key. No new taxonomy.

| Rung | Question it answers | Existing data source it joins |
|---|---|---|
| `python_reference` | Python VJP/JVP numerically checked? | `primitive_coverage` `vjp`/`jvp` axis (auto-flipped from `_VJPS`/`_JVPS`) |
| `ir_adjoint` | `AutodiffPass` emits valid **non-placeholder** backward IR? | ops declaring `Tessera_AdjointInterface` in `TesseraOps.td` with a native `buildAdjoint` (not `placeholderAdjoint`) |
| `target_lowered` | forward **and** backward lower for the target? | `primitive_coverage` `lowering_rule` (fwd) **+** a new backward-lowering probe |
| `runtime_bound` | backward launch ABI + entry point exist? | `runtime_execution_matrix` — **new backward column** (P4) |
| `oracle_proven` | native primal **and** grads == Python reference? | `op_target_conformance` `execute_compare_fixture` — **backward fixture** (P3) |
| `hardware_proven` | native proof completed on the device? | `runtime_execution_matrix` `hardware_verified` rows |

So the only genuinely *new* machinery Phase 0 needs is the **join + render** step
plus the two forward-only registries gaining a backward lane (a matrix column and
a fixture kind) later in P3/P4. Everything else is a read.

**Explicitly-tracked unresolved paths** (each is a row, none silently
"complete"): dynamic-placeholder adjoints
([`AdjointInterface.cpp:171-179`](../../../src/compiler/ir/AdjointInterface.cpp),
`placeholderAdjoint` — these are `ir_adjoint = no`, deliberately), the custom-VJP
bridge (`custom_adjoint_call`), structured control flow, residual saving,
Python-JIT differentiation selection, and the backend backward-launch ABI.

**Build-claim standardization.** Core MLIR fixtures are validated by
`build-llvm22-ninja` (LLVM/MLIR 22.1.6); lean/artifact-only builds must not claim
core-pass coverage. The ledger records *which build proved each rung*.

> **The rungs supersede the coarse "F4 landed / not landed" language.** A family
> can be `ir_adjoint` and still be nowhere near `oracle_proven` — that is the
> normal state, not a regression.

---

## 4. Phases

Serial by default; each phase has a hard exit criterion. Dependency shape:

```
P0 ─┬─► P1 ─► P2 ─► P3 ─► P4 ─► P5 ─► P6
    │   (request) (contract) (slice) (ABI) (families) (HW)
    └─ spec fix + ledger can land independently of P1
```

P0's spec correction is independent of the ledger generator (do them in parallel).
P1→P2→P3 are strictly serial (you cannot define the paired ABI before there is a
request to trigger it, nor prove a slice before the ABI is stable). P5 uses P3+P4
as its per-family template; P6 additionally needs real silicon.

### Phase 0 — Make status and contracts truthful  *(Task #1 — ✅ landed 2026-07-11)*
**Goal:** remove ambiguity about "implemented," "lowered," and "executable."

- ✅ Corrected the stale §F4 prose at
  [`AUTODIFF_SPEC.md`](../../spec/AUTODIFF_SPEC.md) — the pass is no longer a
  no-op, the smoke test passes (not `XFAIL`), `MatmulOp`/`tanh`/`sigmoid` have
  native adjoints while 7 pointwise ops are placeholder round-trips, and "F4
  landed" is scoped to the **IR level** (not execution).
- ✅ Added the generated ledger
  [`generated/autodiff_connection_ledger.md`](../generated/autodiff_connection_ledger.md)
  (+ `.csv`) as the §3 projection — module
  `python/tessera/compiler/autodiff_ledger.py`, registered in
  `tessera.compiler.generated_docs`, drift-gated by
  `scripts/check_generated_docs.sh`. It **reads** `primitive_coverage` +
  parses `AdjointInterface.cpp`; it writes none of them. First cut reports 287
  differentiable families, 2 native / 7 placeholder IR adjoints, 0 backward-
  executable on any target.
- ✅ Reconciliation locked by
  [`tests/unit/test_autodiff_ledger.py`](../../../tests/unit/test_autodiff_ledger.py):
  the ledger's `python_reference` set may not diverge from `primitive_coverage`;
  native/placeholder classes stay grounded in the C++ source; a missing source
  raises (no silent zero).
- ⬜ *Residual:* wire the build-validates-which-claim mapping into the ledger
  rows (record which build proved each rung) — lands with the P4 backward column.

**Exit (met):** the generated ledger answers "which ops have a native vs.
placeholder IR adjoint, and does any execute backward natively on this target?"
without reading source, and its `python_reference` column is test-proven not to
diverge from `primitive_coverage`.

### Phase 1 — Public differentiation contract  *(Task #2 — ✅ landed 2026-07-11)*
**Goal:** differentiation becomes an explicit compiler request. Exact spelling
secondary; e.g.:

```python
@tessera.jit(target="cpu", autodiff="reverse", wrt=("x", "weight"))
def loss(x, weight): ...
```

The contract specifies: mode (reverse first; forward/JVP later); differentiated
inputs + returned primals; scalar-loss vs explicit cotangent seed; gradient-accum
numeric policy; `native_required` behavior; fallback policy (reference execution
only when explicitly requested); source diagnostics for unsupported
AST/control-flow/effects. Python owns validation/spans/types/shapes/errors and
**emits `tessera.autodiff = "reverse"` intent into MLIR** (the attribute
`AutodiffPass` already keys on) — it does not reimplement backward execution.
Provenance reuses the existing `CompileResult` vocabulary (`execution_mode`,
`is_native_supported`, `accelerator_proof`), extended with a **backward** facet —
no new enum.

**Exit (met):** Python can request an IR reverse transform and get a typed
`CompileResult` whose backward facet distinguishes **IR-transformed /
artifact-only / native-executable**, mirroring the forward facet.

**Landed:**
- [`python/tessera/compiler/autodiff_request.py`](../../../python/tessera/compiler/autodiff_request.py)
  — `DifferentiationRequest`, `BackwardStatus`, `BackwardProvenance`; decoration-
  time validation (unknown/forward mode, `wrt` not in signature, `wrt` without
  `autodiff`) via the existing `TesseraAutodiffError`; reverse-only for now
  (forward/JVP a named-diagnostic reject).
- `@jit(autodiff="reverse", wrt=(...))` wired in
  [`jit.py`](../../../python/tessera/compiler/jit.py) — validates, stores
  `fn.differentiation_request` / `fn.backward_provenance`, emits the
  `tessera.autodiff = "reverse"` intent, and mirrors the facet onto
  `CompileResult.backward` (new field).
- `native_required=True` → backward `UNSUPPORTED` with a stable reason today
  (call-time enforcement is Phase 4; the `has_native_backward` hook flips it).
- 18 tests in [`tests/unit/test_autodiff_request.py`](../../../tests/unit/test_autodiff_request.py); mypy clean (ratchet 0); 839-test decoration-path regression green.

**Build-verified two things** (full CPU+Apple `tessera-opt`, NVIDIA backend off
per the registration gotcha):
- The Phase 0 F4 smoke test **actually runs and passes** — `tessera-opt
  --tessera-autodiff` rewrites `@train_step` to emit the transposed-matmul
  adjoints (`dA = dY·Bᵀ`, `dB = Aᵀ·dY`) + `arg_cotangents`. Closes the "no XFAIL
  marker but never executed" gap from Phase 0.
- The round-trip caught a **real bug**: the C++ pass reads the `tessera.autodiff`
  marker off the `func.func` ([`AutodiffPass.cpp:106`](../../../src/transforms/lib/AutodiffPass.cpp)),
  but the first wiring put it only on the *module*. Fixed to emit on the
  differentiated function's `fn_attrs` (where the pass reads it), keeping a
  module-level breadcrumb. The full Python→`tessera-opt` round-trip still can't
  execute because the eager path emits shape-free `tensor<*x?>` types that don't
  parse — **that is exactly Phase 3's scope** (static shapes), not a Phase 1 gap.

### Phase 2 — Stable compiler transform (paired program)  *(Task #3 — ✅ first cut landed 2026-07-11)*
**Goal:** give `AutodiffPass` a compilation-grade contract instead of mutating a
user function as its public API. Keep in-place return-expansion as an internal
bootstrap, but define the paired program:

```
forward(inputs)                       -> primals, residuals
backward(inputs, residuals, out_cot)  -> input_cotangents
```

Decisions to lock: residual representation (explicit named/typed SSA outputs);
save-vs-recompute per adjoint; aliasing/mutation rules; accum dtype + reduction
semantics; effect behavior (RNG/state/collectives/memory writes); compile-time
rejection of unsupported differentiable paths; custom-adjoint contract (lower
`custom_adjoint_call` to a callable runtime ABI **or** reject it in native mode).
W5 is the pattern: static tanh/sigmoid → native backward IR; dynamic shapes stay
explicitly runtime-unresolved via `placeholderAdjoint` until a real impl exists
([`AdjointInterface.cpp:141-179`](../../../src/compiler/ir/AdjointInterface.cpp)) —
and a `placeholderAdjoint` result is `ir_adjoint = no` in the ledger, never
silently counted as done.

**Exit (met for the first cut):** a transformed function has a deterministic
forward/backward/residual ABI, verifiable independently of Python tape state — a
lit fixture checks the backward signature + body, no Python needed.

**Landed (first cut — recompute-all residual policy):**
- New **`--tessera-autodiff-paired`** pass
  ([`src/transforms/lib/AutodiffPairedPass.cpp`](../../../src/transforms/lib/AutodiffPairedPass.cpp),
  a `ModuleOp` pass; additive — the in-place `--tessera-autodiff` stays as the
  bootstrap). For each `tessera.autodiff="reverse"` function it emits a **separate
  backward function**:
  ```
  @f__bwd(inputs..., out_cotangents...) -> (input_cotangents...)
    attributes { tessera.autodiff.role = "backward",
                 tessera.autodiff.forward = @f,
                 tessera.autodiff.residual_policy = "recompute_all" }
  ```
  and links the forward via `tessera.autodiff.paired = @f__bwd`. It clones the
  forward ops into the backward body (recompute-all), reverse-walks them, and
  returns one cotangent per input (zero-splat for inputs off the gradient path,
  so the ABI is total for uniform Phase 4 buffer binding).
- **Residual policy = RECOMPUTE_ALL** for the first cut. This is not a toy choice:
  the shipped ROCm gfx1151 flash-attention backward takes `(dO, Q, K, V)` and
  **recomputes** the softmax rather than saving the logsumexp `L` (see §9). A
  SAVE policy (return selected forward values as explicit residual outputs, e.g.
  flash-attn's `L`) is an optimization the same ABI already carries via
  `tessera.autodiff.residual_policy`.
- **The paired form is an ABI, not an implementation.** A hand-emitted backward
  kernel (ROCm WMMA flash-attn bwd) satisfies the same
  `@f__bwd(inputs, out_cotangents) -> input_cotangents` contract and is a
  first-class arbiter candidate (Decision #28); this pass is the *compiler-
  generated* implementation of that contract.
- Verified by two lit fixtures under
  [`tests/tessera-ir/phase2_autodiff/`](../../../tests/tessera-ir/phase2_autodiff/)
  (matmul paired fwd/bwd; non-differentiable-op rejection), plus the Phase 0 F4
  smoke test — all 5 autodiff lit tests pass on the built `tessera-opt`.

**Deferred to Phase 5 (not in the first cut):** SAVE residual policy (explicit
forward residual outputs + the save-vs-recompute cost decision per adjoint);
aliasing/mutation rules for in-place ops; structured control flow (nested regions
are rejected with `[AUTODIFF_NESTED_REGION]`); the `custom_adjoint_call` → runtime
ABI lowering. The recompute-all cut recomputes forward intermediates the gradient
doesn't need (e.g. the forward matmul in `@loss__bwd` is dead) — a follow-on
canonicalize/DCE pass collapses them; correctness is unaffected.

### Phase 3 — One complete CPU vertical slice  *(Task #4 — ✅ IR-oracle cut landed 2026-07-11)*
**Goal:** prove the architecture before broadening coverage. Scope **only**:
static tensors → `matmul` → `tanh`/`sigmoid` → reduction-to-scalar-loss →
gradients for both inputs.

**Landed — the numerical backward-correctness proof (non-circular):**
- [`tests/unit/test_autodiff_paired_cpu_oracle.py`](../../../tests/unit/test_autodiff_paired_cpu_oracle.py)
  runs the **actual built** `tessera-opt --tessera-autodiff-paired` on a shaped
  `act(matmul(x, w))` forward, then **numerically interprets the pass's emitted
  backward IR** (a tiny NumPy interpreter over the emitted op subset) and asserts
  the gradients match an **independent** NumPy VJP oracle — for both `tanh` and
  `sigmoid`, plus seed→grad linearity and the paired-ABI arity. A pass bug
  mismatches; this executes the compiler's output, not a Python re-derivation.
- Ledger gains a **`bwd_cpu_ir_oracle`** column (matmul/tanh/sigmoid = `cpu`),
  reconciliation-tested. This is the **CPU IR-execution rung** — strictly weaker
  than native `oracle_proven` (Phase 4) and never sets it.
- **Fixed a real Phase-0 ledger bug the proof surfaced:** native-adjoint
  detection only caught ops with a `placeholderAdjoint` fallback (tanh/sigmoid),
  so `matmul` — a fully-native `buildAdjoint` — was misreported `ir_adjoint =
  none`. (Corrected in two stages: the first fix keyed off explicit
  `<Op>Op::buildAdjoint` defs, but that over-counted `LayerNormOp` / `SoftmaxOp`
  — hand-written defs that *emit a `CustomAdjointCallOp` placeholder* — as
  native. The classifier is now **body-aware**: a `buildAdjoint` whose body
  constructs a `CustomAdjointCallOp` is placeholder, keyed by the runtime VJP
  string it carries. The ledger shows the true **3** native adjoints — matmul,
  tanh, sigmoid — with layernorm/softmax as placeholder round-trips. Read the
  generated ledger for live counts, Decision #26.)

**Residual (not in this cut, honestly):**
- **Native CPU execution** (LLVM lowering + runtime launch of `@f__bwd`) — that
  is the runtime ABI binding, which is **Phase 4**; only then does the ledger's
  `oracle_proven`/`runtime_bound` flip.
- **Python `@jit` static-shape emission** — the front-end still emits shape-free
  `tensor<*x?>` that `tessera-opt` won't parse (found in Phase 1). This slice
  used hand-shaped forward MLIR; wiring example-arg shapes through Graph IR
  emission so `@jit(autodiff="reverse")` round-trips is the remaining Phase 3
  front-end task.

Original scope notes (retained):

Path: Python AST → canonical Graph IR → `AutodiffPass` → canonicalize/CSE → CPU
lowering → runtime invocation → compare primal **and** gradients to the Python
VJP reference. Concrete artifacts:
- positive MLIR lit fixtures per stage (Graph-IR emit, post-autodiff, post-CSE,
  CPU-lowered);
- negatives: unsupported op, dynamic placeholder under `native_required`, nested
  region → each a named diagnostic (Decision #21);
- a front-end integration test that runs the `@jit(autodiff="reverse")` function;
- an **exact primal+grad oracle compare** registered as the family's backward
  `execute_compare_fixture` (this is what earns `oracle_proven`);
- a repeated-use / gradient-accumulation test;
- a **fallback-provenance test**: an eager/reference run must never report the
  native/`oracle_proven` state.

**Do not broaden native adjoints beyond this family until it executes and matches.**

**Exit:** one user-facing Python function runs its compiler-generated forward and
backward CPU code and passes numerical comparison — and the ledger shows that
family/CPU at `oracle_proven`.

### Phase 4 — Connect compiler output to the runtime ABI  *(Task #5 — ✅ exit met 2026-07-11 via A2+A3; §9a)*
**Goal:** make the slice a product path, not a test-only pipeline. Define an
explicit compiled fwd/bwd artifact bundle; bind input/output/residual/cotangent/
gradient buffers; add runtime ABI entries for backward entry points; add the
**backward column** to `runtime_execution_matrix`; surface the backward
`execution_mode` + provenance in `CompileResult`. `native_required=True` rejects
artifact-only + `placeholderAdjoint` paths. The current `compile_graph_module`
artifact path must **stop** being read as native execution merely because it built
Graph/Schedule/Tile/Target artifacts. Python VJP tape stays reference/fallback only.

**Exit:** `CompileResult` truthfully exposes native-launch / fallback /
unsupported / numerical-proof states for forward and backward **independently**,
and the ledger's `runtime_bound` rung is sourced from the matrix, not asserted.

### Phase 5 — Expand by closed operation families  *(Task #6)*
Promotion order: (1) core tensor algebra (add/mul/broadcast/reductions/GEMM);
(2) pointwise + normalization (tanh/sigmoid/GELU/SiLU/RMSNorm/layernorm);
(3) losses + optimizer primitives; (4) fused matmul epilogues (explicit
auxiliary/residual contracts); (5) one static-shape attention fwd/bwd;
(6) structured control flow (`if`, bounded `for`, then `while`); (7) Graph-IR
rematerialization/checkpointing (replacing eager-only behavior); (8) custom
VJP/JVP lowering + serialization. **Each family repeats the P3/P4 template:**
Python reference rule · IR adjoint · residual policy · target lowering · runtime
binding · direct oracle proof · benchmark evidence *before* fusion/recompute
tuning. A family is not "added" until its ledger row reaches `oracle_proven`.

### Phase 6 — Distributed + accelerator promotion  *(Task #7)*
Order: wire the existing F5 collectives (`DDP`/`FSDP` vs `mock_collective`) into
the paired backward ABI — **wiring, not new insertion logic**; add x86 native
backward proof where a real kernel exists; promote Apple GPU fwd/bwd only with
fresh-process device proof (`accelerator_proof`, `execution_mode ==
"metal_runtime"`); promote NVIDIA family-by-family via hardware
execute-and-compare; add mixed precision / loss scaling only after accum +
residual dtypes are explicit; add checkpointing / fused backward kernels only
when benchmark evidence names the real bottleneck. The ledger's fwd/bwd split
must prevent "forward native" from being reported as "training supported."

> **ROCm is the special case — its backward already runs (see §9).** gfx1151
> executes native matmul/flash-attn/GQA/selective-ssm backward *today*, so the
> ROCm work is **connecting** existing kernels to the paired ABI + the ledger,
> not building new ones. It is the first place `native_required=True` will
> succeed and the first `hardware_proven` backward rows in the ledger.

---

## 5. Non-goals (scope guardrails)

- **Not** replacing the Python numpy tape as the reference/oracle, nor changing
  its VJP/JVP math.
- **Not** a new audit registry parallel to `primitive_coverage.py` (Decision
  #24) — the ledger is a read-only projection (§3).
- **Not** a new provenance enum — Phase 1/4 extend `CompileResult`'s existing
  `execution_mode`/`is_native_supported`/`accelerator_proof` vocabulary with a
  backward facet.
- **Not** higher-order derivatives in Graph IR — the Python F7 surface
  (`grad`/`hvp`/`jacrev`/`jacfwd`) stays; IR-level HOD is out of scope.
- **Not** new effect-aware adjoint collective *insertion* — Phase 6 only wires
  the already-landed F5 collectives into the paired ABI.
- **Not** touching the `@jit` forward frontend or the backend arbiter/emit spine
  (owned by `COMPILER_REFACTOR_PLAN.md` / `OPTIMIZING_COMPILER_PLAN.md`).

---

## 6. Operating rules

- A feature is production-ready **only** when the compiler/LLVM lane executes it
  and an oracle fixture proves it.
- Python VJP/JVP is the semantic reference, **not** evidence of native compiler
  support.
- A `placeholderAdjoint` / `custom_adjoint_call` result is `ir_adjoint = no`
  until it has a defined runtime lowering — never counted as complete.
- Every new native adjoint must state its **residual policy** and
  **dynamic-shape behavior**.
- Documentation updates + generated evidence are part of every milestone's
  definition of done; the ledger reconciles with `primitive_coverage` on every
  regen (drift-gated).

---

## 7. Immediate work queue

| # | Phase | Deliverable | Session task | Status |
|---|---|---|---|---|
| 1 | 0 | Ledger projection + `AUTODIFF_SPEC` §F4 correction | Task #1 | ✅ |
| 2 | 1 | Python differentiation request + diagnostics (backward `CompileResult` facet) | Task #2 | ✅ |
| 3 | 2 | Paired forward/backward/residual IR contract | Task #3 | ✅ first cut |
| 4 | 3 | Static CPU `matmul → tanh/sigmoid → loss` oracle proof | Task #4 | 🟡 IR-oracle cut (native exec → P4; `@jit` static-shape emission remains) |
| 5 | 4 | Runtime ABI binding + backward matrix column + `native_required` enforcement | Task #5 | ✅ A2+A3 (A1 → Phase 5) |

**Next up: Phase 5 (Task #6)** — expand by closed operation families; the first
executing Tier-1 synthesized backward unblocks A1 (register the WMMA backward as
a Tier-3 arbiter candidate, §9a Inc 3). Then Phase 6 (Task #7).

---

## 8. Status at a glance

`✅` proven · `🟡` partial / in flight · `⬜` not started. Rungs per §3; **counts
and per-target truth stay in the generated dashboards** — this table is the skim
surface only, refreshed as phases land, and is *replaced* by the generated ledger
once Phase 0 lands.

| Phase | Item | Status |
|---|---|:--:|
| 0 | Ledger projection + spec correction (F4 smoke build-verified) | ✅ landed 2026-07-11 |
| 1 | `@jit(autodiff=…)` request + backward provenance | ✅ landed 2026-07-11 |
| 2 | Paired fwd/bwd/residual contract (`--tessera-autodiff-paired`, recompute-all) | ✅ first cut landed 2026-07-11 |
| 3 | matmul→tanh/sigmoid→loss — backward IR oracle-proven on CPU (interpreted) | 🟡 IR-oracle cut landed 2026-07-11 (native execution → Phase 4; `@jit` static-shape emission remains) |
| 4 | Compiled backward bound to runtime ABI (ROCm first) | ✅ **exit met** 2026-07-11 via A2+A3 (matrix backward column → ledger `hardware_proven`; `CompileResult.backward` truthful + `native_required` honored for ROCm `flash_attn`; gfx1151-verified). A1 arbiter dispatch deferred to Phase 5 (one executing backward candidate — see §9a finding) |

Per-family × per-target rung truth is now the **generated ledger**, not a hand
table — read [`generated/autodiff_connection_ledger.md`](../generated/autodiff_connection_ledger.md)
for live counts (Decision #26 — don't trust enumerations copied into prose). The
shape it records: `python_reference` broad; the native `ir_adjoint` set is
exactly matmul, tanh, sigmoid (buildAdjoint bodies that emit real Graph IR),
with layernorm/softmax and the pointwise macro ops as `placeholder` round-trips;
`bwd_cpu_ir_oracle` on the matmul/tanh/sigmoid slice (backward IR
interpreted on CPU, oracle-matched). **Phase 4's A2 closed the former "zero
hardware_proven" blind spot:** the native backward rungs are now sourced from the
runtime execution matrix, so the families that genuinely launch a backward on a
real device light up `bwd_runtime_bound` / `bwd_hardware_proven` — today
`flash_attn` (ROCm gfx1151) and `selective_ssm` (ROCm gfx1151 **and** x86
AVX-512), sourced from the matrix, never asserted. Read the generated ledger for
the live set; do not re-copy these into prose (Decision #26).

---

## 9. AMD / ROCm — the backward that already exists

The single most important AMD fact for this plan: **ROCm gfx1151 (Strix Halo,
RDNA 3.5) already executes native backward on real hardware** — ahead of CPU.
Per [`MASTER_AUDIT.md`](../MASTER_AUDIT.md) + the runtime, gfx1151 runs
`matmul`, `flash_attn` (+ additive `attn_bias`), `GQA`/`MQA`, and
`selective_ssm` **forward *and* backward** via `runtime.launch()`. The backward
lanes are real code, not artifacts:

- `_execute_rocm_compiled_flash_attn_bwd` ([`python/tessera/runtime.py`](../../../python/tessera/runtime.py))
  — `compiler_path = "rocm_flash_attn_bwd_compiled"`, three fragment-materialized
  WMMA kernels from a `tessera_rocm.flash_attn_bwd` directive;
- `createGenerateWMMAFlashAttnBwdKernelPass` /
  `createGenerateROCMSelectiveSsmBwdKernelPass`
  ([`.../Tessera_ROCM_Backend/include/TesseraROCM/Passes.h`](../../../src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/Passes.h)).

Three consequences shape the work:

**1. The paired ABI is the arbiter seam, and AMD proves why.** The ROCm backward
is **hand-emitted WMMA, not `AutodiffPass` output**, and is reached today only by
its own `compiler_path` — disconnected from `@jit(autodiff="reverse")`. That is
exactly Decision #28's three-tier model applied to autodiff: the paired
`@f__bwd(inputs, out_cotangents) -> input_cotangents` contract (Phase 2) is the
interface; the ROCm WMMA backward is a **Tier-3 implementation** of it, the
`--tessera-autodiff-paired` output is the **Tier-1** implementation, and a
measured, accuracy-budgeted arbiter picks per `(op, shape, dtype, target)`.
Shared infra must never cap the ROCm ceiling (Decision #28) — the hand-tuned
WMMA backward stays first-class.

**2. AMD's residual policy is `recompute`, which validated the Phase 2 first
cut.** The ROCm flash-attn backward takes `(dO, Q, K, V)` and **recomputes** the
softmax rather than saving the logsumexp `L`. That is the same recompute-all
policy `--tessera-autodiff-paired` emits — so the contract and the shipped kernel
already agree. A future `save-L` ROCm variant (trade memory for compute) is a
different `tessera.autodiff.residual_policy` the ABI already carries; the
per-`(op, target)` residual policy is the knob.

**Concrete AMD work items (in dependency order):**

| # | Item | Phase | Why AMD-specific |
|---|---|---|---|
| A1 | Register the ROCm flash-attn/matmul/GQA/ssm **backward kernels as backward-ABI candidates** keyed to the paired contract, so `@jit(autodiff="reverse", target="rocm")` reaches them (today: only via the standalone `rocm_flash_attn_bwd_compiled` path). | 4 | The kernels exist but are ABI-disconnected. |
| A2 | Wire the ledger's `runtime_bound` / `hardware_proven` **backward** rungs to read `runtime_execution_matrix` — ROCm gfx1151 `matmul`/`flash_attn`/`GQA`/`selective_ssm` backward should light up `hardware_proven`. Removes the ledger's current blind spot. | 4 | ROCm is the first (and today only) real backward. |
| A3 | Flip the Phase 1 `has_native_backward` hook to `True` for `(matmul, rocm)`, `(flash_attn, rocm)`, … so `native_required=True` **succeeds** on ROCm for those families — the first place it is not rejected. | 4 | Enforcement first pays off on AMD. |
| A4 | Record per-`(op, target)` **residual policy** (`recompute` for ROCm flash-attn today); expose the memory/compute tradeoff to the arbiter + a future `save-L` variant. | 5 | Residual policy is target-specific. |
| A5 | Keep the **RDNA-live / CDNA-gated** split: gfx1151 (RDNA 3.5, WMMA, no FP8) is proven; CDNA (MI300, distinct MFMA table + FP4/FP6) stays hardware-gated — same split as forward. | 6 | Distinct ISA; no CDNA device yet. |

Net: on AMD the plan is **connect, don't build** — the hardware backward is
already there; Phases 4–6 attach it to the paired ABI, the ledger, and
`native_required`, so `@jit(autodiff="reverse", target="rocm")` becomes the first
end-to-end native-backward training path in the compiler.

---

## 9a. Phase 4 ROCm — implementation plan (design decided + landed 2026-07-11)

Grounded in a read of the current surfaces. **A2 + A3 landed** (increments below
marked accordingly); A1 deferred to Phase 5. The `ExecutionRow.direction` /
`op_family` fields, the `native_backward_targets()` / `has_native_backward()`
matrix accessors, the ledger sourcing its `bwd_*` rungs from the matrix, and the
`resolve_backward_provenance` wiring at `jit.py` are all in tree and covered by
`test_autodiff_ledger.py` / `test_autodiff_request.py` /
`test_runtime_execution_matrix.py` (45 passed). This section is retained as the
provenance record of how the work was sequenced.

### Current state (what exists vs. what's missing)

| Surface | State today | file:line |
|---|---|---|
| Execution matrix | **One** backward row exists: `("rocm", "rocm_flash_attn_bwd_compiled")` (`execution_kind="native_gpu"`, `executable=True`). `ExecutionRow` has **no `direction`/`op_family`** field and no `hardware_verified` concept. `matmul`/`GQA`/`selective_ssm` backward have **no matrix rows** (they execute on gfx1151 but are reached via other `compiler_path`s / the fwd lane + autodiff). | `python/tessera/compiler/execution_matrix.py:35` (`ExecutionRow`), `:1790` (flash_attn_bwd row) |
| Ledger | `bwd_target_lowered` / `bwd_runtime_bound` / `bwd_oracle_proven` / `bwd_hardware_proven` are **hardcoded `()`** ("empty by construction"). | `python/tessera/compiler/autodiff_ledger.py:144-147` |
| Request | `has_native_backward` is an **unwired bool param**; every `native_required=True` → `UNSUPPORTED`. | `python/tessera/compiler/autodiff_request.py:175-214` |
| Backward-ABI candidate | ROCm backward reached only via the standalone `rocm_flash_attn_bwd_compiled` path, **not** `@jit(autodiff="reverse", target="rocm")`. | `python/tessera/runtime.py` `_execute_rocm_compiled_flash_attn_bwd` |

### Design decision (decided): backward rides on `ExecutionRow`

Extend `ExecutionRow` with two fields — `direction: str = "forward"` and
`op_family: str = ""` — rather than a parallel backward-lane table. Rationale:
the matrix is already the single source of truth for "what `runtime.launch()`
does per `(target, compiler_path)`" and is drift-gated by
`test_runtime_execution_matrix`; a second registry would need its own sync
contract with the matrix (the exact "new source of truth" the ledger design
(§3, Decision #24) forbids). One row per `(target, compiler_path)` stays true;
`direction`/`op_family` are additive with forward-safe defaults, so every
existing row and the drift test are unaffected until rows are tagged.

### Increment 1 — A2: matrix backward column → ledger (foundation) *(✅ landed)*

1. `execution_matrix.py`: add `direction` + `op_family` to `ExecutionRow`
   (defaults `"forward"` / `""`). Tag the existing flash_attn_bwd row
   `direction="backward", op_family="flash_attn"`. Add backward rows for the
   families that genuinely execute on gfx1151 — `matmul`, `gqa`,
   `selective_ssm` — each pointing at its real executor/`compiler_path` (confirm
   each on-hardware first; do **not** add a row for a family that does not
   actually launch a backward — Decision #21).
2. `autodiff_ledger.py`: replace the hardcoded `()` for `bwd_runtime_bound` /
   `bwd_hardware_proven` with a read of the matrix — group `executable`,
   `direction=="backward"` rows by `op_family`, collect their `target`.
   `bwd_hardware_proven` = the subset whose row is device-proven (see the open
   question on the `hardware_verified` source below); `bwd_runtime_bound` = all
   executable backward rows.
3. Regenerate `docs/audit/generated/autodiff_connection_ledger.{md,csv}` via
   `scripts/check_generated_docs.sh --write`; the ROCm `flash_attn`/`matmul`/
   `gqa`/`selective_ssm` rows light up. Update `test_runtime_execution_matrix`
   expectations + the ledger reconciliation test.

**Exit:** the ledger's "0 families hardware_proven" blind spot (§8) closes for
ROCm; counts come from the matrix, never assertion.

### Increment 2 — A3: `has_native_backward` sourced + wired *(✅ landed)*

1. Add `execution_matrix.has_native_backward(op_family, target) -> bool` (a
   backward row exists, is `executable`, and is device-proven).
2. Wire it into the `@jit` path that builds `BackwardProvenance`
   (`resolve_backward_provenance`, `autodiff_request.py:175`) so a
   `(matmul|flash_attn|gqa|selective_ssm, rocm)` request with
   `native_required=True` resolves to `NATIVE_EXECUTABLE` instead of
   `UNSUPPORTED` — the first place enforcement is not rejected.

**Exit:** `@jit(autodiff="reverse", target="rocm", native_required=True)` on a
covered family returns a truthful `NATIVE_EXECUTABLE` backward facet; every other
`(family, target)` still resolves honestly (`UNSUPPORTED`/`IR_TRANSFORMED`).

### Increment 3 — A1: register the ROCm backward kernels as paired-ABI candidates *(deferred to Phase 5 — see finding)*

**Finding (2026-07-11): A1 is premature and correctly deferred.** Two facts,
found while landing A2/A3:

1. **The backward already dispatches natively** — the gfx1151 flash-attn / ssm
   backward is reached through the `autodiff.vjp` rules (`vjp_flash_attn` /
   `vjp_selective_ssm` call `runtime._rocm_*_bwd` directly), not only the
   standalone `compiler_path`. So the reverse-mode path *does* reach the native
   kernel today; A2/A3 make that reality honest in the ledger + `CompileResult`.
2. **There is only one executing backward candidate.** The accuracy-budgeted
   arbiter's job is to *choose* among candidates; the Tier-1 synthesized backward
   (`--tessera-autodiff-paired`) is **IR-only** (oracle-verified on CPU by
   interpretation, not executed — Phase 3). Until it executes (Phase 5), the
   WMMA backward is the sole candidate, so routing backward through the emit
   arbiter would add a duplicative dispatch path with nothing to arbitrate.

Whether the executed Tier-1 backward should flow through the `emit/candidate`
arbiter (a parallel path) or the existing `autodiff.vjp` seam is a real
Decision-#28 architecture choice, **taken when a second executing backward
candidate exists (Phase 5)** — not a mechanical wiring. Registering a
one-candidate arbiter lane now would not be clean.

**Exit (Phase 5):** with the Tier-1 synthesized backward executing, register the
WMMA backward as a **Tier-3** candidate on the paired `@f__bwd(inputs,
out_cotangents) -> input_cotangents` contract (residual policy `recompute`), and
let the accuracy-budgeted arbiter pick per `(op, shape-bucket, dtype, "rocm")`
(Decision #28 — the hand-tuned WMMA stays first-class, displaced only by a faster
in-budget compiled backward).

### Phase 4 status: **exit criteria met by A2 + A3**

The Phase 4 exit (§4) — *"`CompileResult` truthfully exposes native-launch /
fallback / unsupported / numerical-proof states for forward and backward
**independently**, and the ledger's `runtime_bound` rung is sourced from the
matrix, not asserted"* — is satisfied: A2 sources the ledger's
`runtime_bound`/`hardware_proven` backward rungs from the matrix; A3 makes
`CompileResult.backward` resolve `NATIVE_EXECUTABLE` / `UNSUPPORTED` /
`IR_TRANSFORMED` honestly and honors `native_required`. Both are hardware-verified
on gfx1151 (`test_rocm_flash_attn_launch_execute.py`, 13 passed, full CORE+HIP
build). A1 arbiter dispatch moves to Phase 5 per the finding above.

### Verification (all on the gfx1151 box)

- Per increment: `python3 -m pytest tests/unit/test_autodiff_ledger.py
  tests/unit/test_autodiff_request.py` + the execution-matrix drift test +
  `scripts/check_generated_docs.sh` (drift gate).
- Inc 3: an execute-and-compare fixture on real gfx1151 (the flash-attn/matmul
  backward gradients vs. an independent NumPy VJP), mirroring the forward
  `test_rocm_*_runtime_symbol.py` lanes.

### Open questions — resolved during Inc 1

- **`hardware_verified` source. → Resolved.** Rather than reach into
  `backend_manifest` / `rocm_target_map`, `bwd_hardware_proven` reads the
  backward row's own `execution_kind`: a row counts as device-proven iff its
  kind is in `_DEVICE_EXECUTION_KINDS` (`native_gpu` / `native_cpu` /
  `cpu_accelerate`). This keeps the matrix the single truth — no new
  device-proof field, no second registry (`native_backward_targets()` in
  `execution_matrix.py`).
- **matmul/GQA/ssm backward reachability. → Resolved.** Only families that launch
  a *distinct* backward get a row: `flash_attn` (which subsumes `GQA`/`MQA`) and
  `selective_ssm`. `matmul` backward composes the **forward** matmul lane (two
  transposed matmuls), so it is not tagged with a separate backward row — adding
  one would double-count a launch the matrix already records forward (Decision
  #21). If an on-hardware pass later shows a *distinct* matmul-backward launch
  worth crediting, tag it then, not by assertion.
