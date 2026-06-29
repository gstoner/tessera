---
status: Normative (CF0 slice)
classification: Spec
authority: CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md CF0
last_updated: 2026-06-29
---

<!-- CF1 (2026-06-29): trace-time carry/branch dtype contract + verifier lit
     coverage; pytree-carry tracing carved out to CF1b. -->


# Control-Flow Contract (CF0)

> **Status:** CF0 of `docs/audit/roadmap/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md`.
> Closes the semantic envelope of `tessera.control.{scan, fori_loop, while_loop,
> cond}` **before** any new backend (CUDA/ROCm) control-flow codegen lands.
> This document is descriptive of the code as it exists, not aspirational —
> every claim below is grounded in a named source.

This is a **contract inventory**: what the four control-flow primitives accept,
what effects they may carry, what the trace front-end already enforces, where
they lower today, and — the gap CF1–CF4 close — what happens when a control op
reaches a backend that has no control-flow lowering.

---

## 1. The four primitives (Python reference surface)

`python/tessera/control.py` owns the host-reference semantics. Each primitive
has two modes: an **eager host path** (a plain Python loop / branch — the
reference oracle) and a **traced path** that, under an active trace builder,
emits a first-class Graph IR op instead of unrolling.

| Primitive | Eager (`control.py`) | Traced op emitted | Trace builder method |
|---|---|---|---|
| `scan(fn, init, xs, length, reverse)` | sequential carry loop, stacked `ys` | `tessera.control_scan` | `Tracer.record_scan` |
| `fori_loop(lo, hi, body, init)` | `for i in range(lo, hi)` | `tessera.control_for` | `Tracer.record_for_loop` |
| `while_loop(cond, body, init, max_steps)` | `while cond(v)` (bounded by `max_steps`) | `tessera.control_while` | `Tracer.record_while` |
| `cond(pred, t_fun, f_fun, *ops)` | `t_fun() if pred else f_fun()` | `tessera.control_if` | `Tracer.record_cond` |

The trace seam is `control._active_trace_builder()` →
`compiler._trace_hook.active_tracer()`; when it returns `None` (no JIT/trace in
scope) the eager reference runs. The eager path is **always** the numerical
oracle the lowered paths are validated against.

`associative_scan`, `switch`, `map`, `pmap` exist in `control.py` as host
reference only — they have **no** traced-op emission yet and are out of CF0
scope (tracked as CF-follow-ups).

---

## 2. Supported loop forms (the closed envelope)

CF0 fixes the device-lowerable subset. Forms outside this envelope must be
**rejected at trace time** with a `TesseraTraceError`, never silently unrolled
inside a compiled-backend claim.

### `scan`
- **Static trip count.** `length` (or `xs.shape[0]`) is a Python int at trace
  time (`trace.py:327`).
- **Static carry shape.** `fn` must return `(carry, y)` with
  `carry.shape == init.shape` (`trace.py:338`).
- `xs` must be a single `Tracer` with a leading scan axis; `reverse=True` and
  pytree-`xs` fall back to the host reference (`control.py:114`).
- Stacked output `ys` has static shape `(trip, *y.shape)`.

### `fori_loop`
- **Static bounded trip count.** `trip = upper - lower`, both Python ints
  (`trace.py:243`). Symbolic bounds fall back to the eager loop.
- **Static carry shape.** body must return a `Tracer` of the init carry shape
  (`trace.py:250`). The `control_for` ABI is index-independent: the body is
  traced once at `i=0`.

### `while_loop`
- **Bounded maximum required under trace.** `max_steps` is mandatory
  (`trace.py:290`) and lowers to `control_while`'s `max_iters` (verifier
  requires `max_iters > 0`, `TesseraOps.cpp:2784`). An unbounded traced
  `while_loop` is a hard error — there is no way to bound the device loop
  otherwise.
- **Static carry shape**, single carried `Tracer` (`trace.py:300`); the
  predicate `cond(carry)` must return a `Tracer`.

### `cond`
- **Scalar predicate** carried as the flag arg (a shape-`(1,)` tensor;
  `flag[0] > 0` selects `then`, `TesseraOps.td` `ControlIfOp`).
- **Same-shaped branch results** — `t_fun` and `f_fun` must return `Tracer`s
  with matching shape (`trace.py:272`). Divergent execution (only the taken
  branch runs), not a data-parallel select.

### Known envelope gaps (documented, not silently wrong)
- ~~The trace checks carry/branch **shape** equality but not yet **dtype**
  equality.~~ **Closed in CF1.** The trace now also enforces carry/branch
  **dtype** equality on the for/while/scan/cond paths (`record_for_loop` /
  `record_while` / `record_scan` / `record_cond` in `trace.py`), matching the
  Graph IR verifiers' full-type match (`control_for` result vs carried iter_arg
  type, `TesseraOps.cpp:2751`). A body that drifts the carry dtype (e.g. an
  in-body cast) now fails at trace time with a clear `TesseraTraceError` instead
  of slipping to the later MLIR verifier. Covered by
  `tests/unit/test_cf1_control_flow_dtype.py`.
- Carry is a single `Tracer` under trace — **pytree carries are not yet traced**
  (they fall back to eager). Multi-tensor device carries are a **CF1b** item
  (the structural change: multi-operand control ops + flatten/unflatten +
  `execute_traced` plumbing).

---

## 3. Effects

Device-lowered control bodies are **pure-by-construction** today: all three
Graph IR ops (`control_for`, `control_if`, `control_while`) and `control_scan`
are marked `[Pure]` (`TesseraOps.td`). The contract a body must satisfy:

- **No Python side effects** inside a device-lowered body — the body is traced
  once into a serialized op-list payload (`body_opcodes`/`body_in0`/… in the
  ODS); anything not expressible as a traced op is invisible to the device loop.
- **RNG only through explicit state operands.** A body that needs randomness
  threads an RNG key/counter as a carried operand (the P6 Philox `RNGKey` ABI),
  never via hidden global state — otherwise iterations are not reproducible.
- **Cache mutation only through typed cache handles.** KV/SSM/state mutation
  rides the `KVCacheHandle` / `MemoryStateHandle` ABI as an explicit carried
  operand, not implicit Python object mutation.
- **Speculative rollback via cursor/state handles**, not implicit Python state
  (`speculative.advance_kv` / `advance_ssm` are the reference for the
  accepted-prefix advance; SD1 promotes these to ops).

When a future body op carries a non-pure effect, the op's `[Pure]` trait must be
relaxed and the `EffectLattice` walk (`pure<random<memory<io<top`) must classify
the loop accordingly — tracked with the effect work, not assumed here.

---

## 4. Where control flow lowers **today**

| Target | `control_for` | `control_if` | `control_while` | `control_scan` |
|---|---|---|---|---|
| **Apple GPU** | ✅ `ControlForToAppleGPU` | ✅ (`control_if` lowering) | ✅ `ControlWhileToAppleGPU` | ✅ `run_graph_scan_f32` |
| **CUDA (NVIDIA)** | ❌ none | ❌ none | ❌ none | ❌ none |
| **ROCm (AMD)** | ❌ none | ❌ none | ❌ none | ❌ none |
| **x86** | ❌ none (eager host loop) | ❌ none | ❌ none | ❌ none |

Apple is the only backend with device control-flow lowering today (Phase-G/H);
it is the **research lane** per the acceleration plan. The portable closure
path (CF3 CUDA, CF4 ROCm) is greenfield.

---

## 5. The CF0 diagnostic gap (the deliverable)

**A `control_*` op targeting CUDA or ROCm hits no lowering pattern and no
diagnostic.** The ROCm backend already owns the Decision #21 pattern for
contracts a target cannot lower — `op->emitError("ROCm lowering does not support
TMEM operations")` (`TileToROCM.cpp:343`), `tessera.target.diagnostic` for
KV-cache (`Tessera_ROCM_Backend/README.md`) — but control flow is **not wired
into it**. Per Decision #21 (unsupported lowering must emit a stable diagnostic
naming the op and the target, never silently no-op), CF0 adds:

> `CONTROL_FLOW_UNSUPPORTED_ON_TARGET: '<op>' is not yet executable on target
> '<target>'; device control-flow lowering for this backend lands in CF3 (CUDA)
> / CF4 (ROCm). Only apple_gpu lowers control flow today.`

for `control_for` / `control_if` / `control_while` / `control_scan`, implemented
as `ControlFlowTargetGuardPass`
(`src/transforms/lib/ControlFlowTargetGuardPass.cpp`):

- **Standalone:** `--tessera-control-flow-target-guard=target=<name>` — the
  `target` option names the backend in the message (detection is
  target-independent). Used by the lit fixture
  `tests/tessera-ir/control_flow/cf0_target_guard.mlir` to assert the diagnostic
  for every target including `rocm`, mirroring
  `Tessera_ROCM_Backend/test/rocm/unsupported_tile_features.mlir`.
- **Wired into the Graph-IR lowering pipelines that lack control lowering:**
  `tessera-lower-to-x86` (target `x86`), `tessera-lower-to-gpu` and
  `tessera-nvidia-pipeline*` (target `nvidia_sm90`), right after
  `addGraphIRPreLoweringPasses` so a control-flow program fails before any
  confusing downstream pass. The ROCm pipeline (`tessera-lower-to-rocm`)
  consumes *ROCm Target IR* — Graph-IR `control_*` ops never reach it directly
  today, so its guard attaches when CF4 adds the ROCm Graph-IR entry; the
  standalone pass + lit fixture cover the ROCm diagnostic until then.

This guarantees that until CF3/CF4 land real kernels, a control-flow program on
those targets fails loudly at compile time rather than producing an
executable-backend claim that silently fell back to a host loop.

---

## 6. Negative-test matrix (CF0 acceptance)

Each row is a form **outside** the §2 envelope; the contract is a clear
trace-time rejection (`TesseraTraceError`), proven by a test.

| Rejected form | Where caught | Message intent |
|---|---|---|
| `while_loop` under trace without `max_steps` | `trace.py:290` | "needs a bound: pass max_steps=N" |
| `fori_loop` body changes carry shape | `trace.py:250` | "body must preserve the carry shape" |
| `while_loop` body changes carry shape | `trace.py:300` | "body must preserve the carry shape" |
| `scan` body changes carry shape | `trace.py:338` | "body must preserve carry shape" |
| `cond` branches return different shapes | `trace.py:272` | "branches must share a shape" |
| `cond`/`scan`/loop body returns a non-Tracer (host object capture / data-dependent value) | `trace.py:248/270/334` | "must return a Tracer" |
| `control_*` op lowered to CUDA/ROCm | CF0 diagnostic (§5) | "not yet supported on target '<t>'" |

Acceptance: the eager reference behavior is unchanged (existing
`control.py`-level tests remain the oracle), and the JIT/trace path either emits
a first-class `control_*` op (Apple) or produces the §5 diagnostic
(CUDA/ROCm) — never a silent host-loop fallback inside an executable backend
claim.

---

## 7. What CF1–CF4 build on this

- **CF1** *(done)* — tightened the trace-time dtype check to match the
  verifiers; full verifier lit coverage of the payload (carry_arg_index) form +
  if-payload symmetry (`tests/tessera-ir/control_flow/cf1_control_verifiers.mlir`).
- **CF1b** — pytree-carry tracing (multi-operand control ops + flatten/unflatten
  + `execute_traced` plumbing); optional `tessera.control.yield` helper.
- **CF2** — Schedule/Tile lowering shared by CUDA/ROCm: static-trip loops with
  explicit carry buffers; `while` → bounded loop + device predicate +
  `actual_steps`; `cond` → predicated regions. One backend launch per loop
  wrapper, not one per iteration.
- **CF3 / CF4** — replace the §5 diagnostic with executable CUDA / ROCm
  control-flow kernels (scan/for/while/cond proofs) validated against the §1
  eager reference.
