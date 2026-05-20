# Frontend Substrate Plan

> **Status:** Active. This is the authoritative architecture doc for the
> Tessera frontend substrate work (Phases A–D). Supersedes the
> exploratory framing in earlier "Graph IR vs Middle IR" discussions.
>
> **Last updated:** 2026-05-20

## Decision

Tessera will **strengthen Graph IR as the shared tooling surface**.
Graph IR carries the cross-cutting metadata every frontend lane
benefits from (source positions, lane provenance, value kind,
numeric policy, verification facts) and serves as the common
substrate for audit, explain, and normalization passes.

Tessera will **not** build a new Middle IR tower. Adding another
IR layer between the AST visitors and Graph IR would force
extra translation infrastructure, multiply the test surface,
and risk MIR drifting from Graph IR.

Tessera will **not** force the constrained math lanes
(`@clifford_jit`, `@complex_jit`, `@energy_jit`) to lower into
Graph IR as their source of truth. Those lanes earn their
separate IR types — `CliffordIRProgram`, `ComplexIRProgram`,
`EnergyIRProgram` — from the per-lane verification they enforce
at decoration time (GA whitelist, holomorphicity, energy
whitelist). Folding them into Graph IR risks erasing those
invariants.

Instead, constrained IRs will expose **`to_graph_ir_view()`**
adapters that project them into Graph IR shape for audit /
explain / diagnostics consumption. The constrained IR remains
sovereign over execution; Graph IR becomes the **lingua franca
for cross-cutting tooling**.

## Five Load-Bearing Refinements

The decision above only works if these five rules are enforced
in code, not just in prose. Each refinement names the failure
mode it prevents.

### 1. Optional Metadata Contract + Drift Gate

**Rule:** Every new field on `IROp`, `GraphIRFunction`, or
`GraphIRModule` must have a default value. Adding a required
field is a breaking change that forces every producer (Python
JIT, textual DSL, constrained-lane views) to compute it
immediately.

**Three-line contract:**

1. Producers fill what they know.
2. Consumers must tolerate missing metadata.
3. Metadata semantics are stable when present.

**Enforcement:** `tests/unit/test_optional_ir_metadata_contract.py`
walks `dataclasses.fields()` on the three IR classes and asserts
every field has a `default` or `default_factory`. An explicit
`_GRANDFATHERED_REQUIRED_FIELDS` allowlist exists for the few
genuinely-required fields (`IROp.op_name`, `IROp.operands`,
etc.); adding a new entry requires a rationale comment in the
allowlist.

**Failure mode prevented:** "Optional today, required by
accident in three months" — the slow decay where one producer
needs a field and quietly makes it required.

### 2. `to_graph_ir_view()` Contract

**Rule:** Every constrained-lane IR program type exposes a
`to_graph_ir_view() -> GraphIRModule` method with the following
contract:

| Contract point | Required behavior |
|---|---|
| **Op shape** | 1:1 projection — `view.functions[0].body` maps each constrained-IR op to exactly one `IROp`. Reordering, merging, or splitting is forbidden. The view is a projection, not a normalization. |
| **Op naming** | Canonical names (`mobius`, `complex_mul`, `ebm_inner_step`) — not backend aliases (`complex_mobius`). The audit walker already maintains `_M7_BACKEND_ALIASES` for backend lookup; the view matches what `.explain()` shows users. |
| **Mutability** | Fresh deep copy per call. Downstream passes can mutate freely. Matches the G4 cache semantics. |
| **Lane stamping** | `view.functions[0].lane` is always set to the source lane (`"clifford_jit"` / `"complex_jit"` / `"energy_jit"`). |
| **Verification facts** | `view.functions[0].verification_facts` carries the lane invariants (e.g., `{"holomorphic"}` for `@complex_jit`). |

**Failure mode prevented:** Three lane adapters drifting apart
in subtle ways — one projects 1:1, another collapses adjacent
ops, a third renames mid-flight. Without a written contract,
audit code has to special-case each adapter.

**Spec location:** `docs/spec/COMPILER_REFERENCE.md` section
"Constrained-lane Graph IR views" — written **before** the
first adapter is implemented.

### 3. Normalization Pipeline Ordering

**Rule:** Normalization passes run in a documented, stable
order. Implicit ordering breaks when a new pass lands between
two existing passes that happened to compose correctly by
accident.

**Documented pipeline:**

```python
NORMALIZATION_PIPELINE: tuple[Callable[[GraphIRFunction], None], ...] = (
    canonicalize_op_names,          # strip "tessera." prefix
    propagate_source_positions,     # AST lineno/col_offset → IROp.source_span
    set_lane_provenance,            # function.lane from decorator metadata
    propagate_value_kinds,          # depends on canonical op names
    propagate_numeric_policy,       # keys on canonical names (G3)
    propagate_verification_facts,   # depends on lane + value_kind
)
```

Each pass is **idempotent** (running it twice produces the
same result) and **lane-aware** (no-op when the lane invariant
prevents the change).

**Enforcement:** `tests/unit/test_normalization_pipeline_order.py`
asserts the tuple has a stable element ordering. Re-ordering a
pass requires updating the test, which forces a review.

**Failure mode prevented:** Pass-ordering bugs where pass B
silently depends on pass A having already run, but a refactor
moves B before A.

### 4. Optimization Stays Locked Until Triggered

**Rule:** CSE, dead-binding elimination, constant folding,
and inlining stay **out of the pipeline** until a documented
trigger fires.

**Triggers (any one is enough):**

1. **Benchmark gap.** A specific benchmark in `benchmarks/`
   shows >2× gap between current Tessera and an external
   reference, *and* the gap is attributable to a missing
   optimization pass (proven by manual rewriting).
2. **Correctness need.** A correctness bug requires CSE / DBE
   / inlining as part of the fix — i.e., the bug cannot be
   closed without the pass.
3. **External requirement.** A paper claim, customer ask, or
   conformance test demands optimization-class transforms.

Without one of these triggers firing, Phase D stays deferred.
"Deferred" is not "abandoned" — the substrate (canonical
naming, lane provenance, verification facts) is what lets
these passes be lane-safe when they eventually land.

**Especially watch:** A "dead" `check_cauchy_riemann` op may
look unused by result-use analysis but is semantically
load-bearing for the lane invariant. Any DBE pass must consult
`verification_facts` before deleting.

**Failure mode prevented:** "Add optimization passes because
the substrate exists" — premature optimization with subtle
lane-effect bugs.

### 5. Per-Phase Entry / Exit / Drift Criteria

**Rule:** Every phase has explicit entry conditions, exit
conditions, and drift-gate tests. Phases without exit
criteria become indefinite work streams.

| Phase | Entry (substrate ready) | Exit (proves done) | Drift gate (prevents backsliding) |
|---|---|---|---|
| **A** Optional IR metadata | F1+G2+F2 (shipped 2026-05-19) | `IROp.value_kind` / `verification_facts` + `GraphIRFunction.verification_facts` / `source_hash` ship with defaults; `.explain()` consumes them gracefully | `test_optional_ir_metadata_contract` walks dataclass fields |
| **B** `to_graph_ir_view()` adapters | Phase A complete; view contract written to `COMPILER_REFERENCE.md` | 3 `to_graph_ir_view()` methods (Clifford / Complex / Energy) | Per-lane round-trip test asserting 1:1 op-shape projection |
| **C** Normalization passes | Phases A + B complete | `NORMALIZATION_PIPELINE` tuple + 6 idempotent passes + `@tessera.jit` invokes the pipeline | `test_normalization_pipeline_order` + per-pass idempotence test |
| **D** Optimization passes | A documented trigger fires (benchmark, correctness, external) | Per-pass — no single deliverable | Lane-effect test per pass before merge |

## Why This Plan, Concretely

The honest near-term value is **compiler rigor and
explainability**, not raw codegen speed:

- A developer running `@clifford_jit` on a function that
  accidentally calls `np.dot` gets a typed
  `CLIFFORD_OP_NOT_WHITELISTED` diagnostic with line/col info
  and a "rewrite using `ga.geometric_product`" hint — not a
  free-form Python exception.
- Audit code (`support_table`, `examples_audit`,
  `claim_lint`) consumes any lane through one uniform shape
  (`GraphIRModule`) without per-lane special cases.
- Normalization passes (canonical naming, source-position
  propagation) run once per function regardless of which lane
  produced it.

Raw codegen wins **may** come later (Phase D triggers), but
they are not guaranteed until a backend or benchmark exposes
a real gap. We're not buying speedups; we're buying
**substrate that makes speedups possible without lane-effect
bugs**.

## Open Questions (Deferred)

These are real questions but not load-bearing for Phases A–C.
Address when concrete need arises:

- **Cross-lane composition.** Can a function decorated with
  `@tessera.jit` contain a call to a `@clifford_jit`-decorated
  helper? Today: no clean answer. The lane registry knows
  about it but the IR doesn't bridge.
- **Constrained-lane views with execution.** Phase B
  explicitly says the view is for tooling, not execution.
  But there's an open design question: could a future
  `@tessera.jit` runtime *execute* a constrained-lane view as
  fallback when the lane's native execution path is
  unavailable? Defer until needed.
- **IR versioning.** Adding fields with defaults is
  backwards-compatible. Removing or renaming fields is not.
  Today there's no IR-version attribute. Probably needed
  before any field-removal change.
- **Schedule IR / Tile IR / Target IR metadata propagation.**
  This plan covers Graph IR. The same optional-metadata
  contract should likely apply downstream — but that's
  follow-up work that depends on Phase C landing first.

## Non-Goals (Explicit)

- **A separate Middle IR layer.** Not built. Architecturally
  rejected.
- **Collapsing the constrained lanes into Graph IR.** Not
  built. Constrained lanes keep execution sovereignty.
- **Optimization passes by default.** Deferred until a Phase
  D trigger fires.
- **A "full type checker."** Tessera's type story is
  distributed (constraints + numeric_policy + canonical dtype
  + support_table). This plan improves the *diagnostics*
  around that story, not the type-checking architecture
  itself.
- **LSP / IDE integration.** Source positions land in Phase
  A, which is the substrate an LSP would consume. The LSP
  itself is downstream and depends on editor wiring not
  scoped here.

## References

- Decision conversation logs: 2026-05-20 thread on "Graph IR vs
  Middle IR" in the compiler-architecture review.
- F-cluster substrate that Phase A builds on:
  `python/tessera/compiler/{diagnostics,symbol_table,frontend_lanes,explain}.py`
  (shipped 2026-05-19).
- The audit infrastructure that consumes Graph IR metadata:
  `python/tessera/compiler/{audit,support}.py`.
- The `_M7_BACKEND_ALIASES` precedent for canonical-vs-backend
  naming separation: `python/tessera/compiler/audit.py`.
