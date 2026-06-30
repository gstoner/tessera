---
status: Normative (CF0 slice)
classification: Spec
authority: CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md CF0
last_updated: 2026-06-30
---

<!-- CF1 (2026-06-29): trace-time carry/branch dtype contract + verifier lit
     coverage; pytree-carry tracing carved out to CF1b. -->
<!-- CF2 (2026-06-29): control_for → scf.for lowering (LowerControlFlowToSCFPass);
     the legacy all-carried form becomes multi-iter_args scf.for, where pytree
     carries fold in. Wired before the CF0 guard in the named pipelines. -->
<!-- CF2b/CF2c (2026-06-29): control_if → scf.if and control_while → bounded
     scf.while in the same pass; same payload-skip discipline. -->
<!-- CF4e-1 (2026-06-30): control_scan promoted to a first-class Graph IR op
     (ODS + verifier) and lowered to a per-thread ROCm device kernel
     (GenerateROCMControlScanKernel) — the 4th primitive now executes on gfx1151
     (elementwise body; per-step xs in, stacked ys out). §4 + §7 updated. -->


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
| **ROCm (AMD)** | ✅ narrow CF4 rank-1 elementwise proof | ✅ narrow CF4 rank-1 elementwise proof | ✅ narrow CF4 rank-1 elementwise proof | ✅ CF4e-1 rank-1 elementwise proof |
| **x86** | ❌ none (eager host loop) | ❌ none | ❌ none | ❌ none |

Apple is the only backend with device control-flow lowering today (Phase-G/H);
it is the **research lane** per the acceleration plan. ROCm now has CF4 proof
kernels for the no-capture rank-1 elementwise `control_for` / `control_if` /
`control_while` envelope; CUDA and the broader ROCm closure remain greenfield.

---

## 5. The CF0 diagnostic gap (the deliverable)

**A `control_*` form/envelope that a target cannot lower must fail with a stable
diagnostic.** The ROCm backend already owns the Decision #21 pattern for
contracts a target cannot lower — `op->emitError("ROCm lowering does not support
TMEM operations")` (`TileToROCM.cpp:343`), `tessera.target.diagnostic` for
KV-cache (`Tessera_ROCM_Backend/README.md`). Per Decision #21 (unsupported
lowering must emit a stable diagnostic naming the op and the target, never
silently no-op), CF0 adds:

> `CONTROL_FLOW_UNSUPPORTED_ON_TARGET: '<op>' is not yet executable on target
> '<target>'; no lowering exists for this control-flow form/envelope on this
> target.`

for `control_for` / `control_if` / `control_while` / `control_scan`, implemented
as `ControlFlowTargetGuardPass`
(`src/transforms/lib/ControlFlowTargetGuardPass.cpp`):

- **Standalone:** `--tessera-control-flow-target-guard=target=<name>` — the
  `target` option names the backend in the message (detection is
  target-independent). Used by the lit fixture
  `tests/tessera-ir/control_flow/cf0_target_guard.mlir` to assert the diagnostic
  for every target including `rocm`, mirroring
  `Tessera_ROCM_Backend/test/rocm/unsupported_tile_features.mlir`.
- **Wired into the Graph-IR lowering pipelines that lack a lowering for the
  selected control-flow form/envelope:**
  `tessera-lower-to-x86` (target `x86`), `tessera-lower-to-gpu` and
  `tessera-nvidia-pipeline*` (target `nvidia_sm90`), right after
  `addGraphIRPreLoweringPasses` so an unsupported control-flow program fails
  before any confusing downstream pass. ROCm CF4 handles the narrow no-capture
  rank-1 elementwise proof kernels; the guard remains the contract for
  everything outside that envelope and for targets without a device lowering.

This guarantees that unsupported control-flow forms fail loudly at compile time
rather than producing an executable-backend claim that silently fell back to a
host loop.

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
| `control_*` form/envelope lowered to an unsupported target path | CF0 diagnostic (§5) | "no lowering for this form/envelope on target '<t>'" |

Acceptance: the eager reference behavior is unchanged (existing
`control.py`-level tests remain the oracle), and the JIT/trace path either emits
a first-class `control_*` op for a supported device lowering or produces the §5
diagnostic — never a silent host-loop fallback inside an executable backend
claim.

---

## 7. What CF1–CF4 build on this

- **CF1** *(done)* — tightened the trace-time dtype check to match the
  verifiers; full verifier lit coverage of the payload (carry_arg_index) form +
  if-payload symmetry (`tests/tessera-ir/control_flow/cf1_control_verifiers.mlir`).
- **CF1b** — pytree-carry tracing (multi-operand control ops + flatten/unflatten
  + `execute_traced` plumbing); optional `tessera.control.yield` helper.
- **CF2** *(in progress)* — portable, hardware-free lowering shared by
  CUDA/ROCm. **Done:** `control_for` → `scf.for` carrying state in `iter_args`
  (`LowerControlFlowToSCFPass`, `--tessera-control-flow-to-scf`), the body kept
  as a `func.call @body`; one `scf.for` wrapper, not one launch per iteration.
  The legacy all-carried form becomes a **multi-`iter_args`** `scf.for` — where
  **pytree carries fold in** (the CF1b front-end work then just has to flatten a
  pytree carry into the operand list this lowering already threads). Wired into
  the named `tessera-lower-to-x86` / `-gpu` / `-nvidia-pipeline*` builders
  **before** the CF0 guard, so a lowerable loop becomes `scf.for` and never
  trips the guard; the executable-**payload** form (where `@body` is a
  carry-only stub and the real body lives in `body_opcodes`) is **skipped**
  (can't form a correct `func.call` without decoding the payload) and still
  guarded.
- **CF2b/CF2c** *(done)* — `control_if` → `scf.if` (flag `[0] > 0` via
  `tensor.extract` + `arith.cmpf`; branches kept as `func.call`s over the
  non-flag operands) and `control_while` → **bounded** `scf.while` (loop state
  `(counter : index, carry)`; the `before` region checks `i < max_iters` first
  and **short-circuits** — `cond(carry)` is evaluated only inside the
  then-branch of an `scf.if`, so an always-true condition runs at most
  `max_iters` times and never at `i == max_iters`; the `after` region runs the
  body + increments). Same
  payload-skip discipline as `control_for`. All three lower through the named
  `tessera-lower-to-{x86,gpu}` / CUDA13 pipelines before the guard.
- **CF4a** *(done)* — `MaterializeControlPayloadPass`
  (`--tessera-materialize-control-payload`) decodes the frontend's
  executable-payload op-list (`body_opcodes`/`body_in0`/…) on `control_for` into
  a real `@body` `func.func` of `tessera.*` ops (opcode table + tensor-id map +
  result-type inference), then strips the payload attrs — so CF2 lowers the loop
  to `scf.for` with a *real* device body. Unknown opcodes / unresolvable shapes
  are left untouched (two-phase: validate before mutating).
- **CF4a-cont** *(control_if done)* — the same decoder materializes the
  `control_if` `then_opcodes`/`else_opcodes` payload into real `@then`/`@else`
  funcs. `control_if`'s op-list ids are `0..n-1` = operands, branch op j = id
  `n+j` (no carry); CF2 calls the branches with the **non-flag** operands, so the
  materialized branches take the non-flag operands and a payload id `k` maps to
  arg `(k < flag ? k : k-1)`. A branch referencing the flag id is left untouched;
  both branches validate before either is materialized.
- **CF4a-cont-2** *(control_while done)* — the same decoder materializes the
  `control_while` `body_opcodes`/`cond_opcodes` payload. Its ids are `0..n-1` =
  operands, id `n` = live carry, body & cond op j = id `n+1+j`; CF2 calls
  `@body`/`@cond` with only the carry, so both materialize as single-arg funcs
  (id `n` → arg 0). `@body` is `(carry)->carry` (out-type checked == carry);
  `@cond` is `(carry)->pred` with the **predicate type inferred** from the cond
  op-list (`validateOpList` gains a null-`resultTy` infer mode). `@body` and
  `@cond` must be distinct symbols. So **all three control payloads now decode**
  → CF2 → `scf.{for,if,while}`.
- **CF4b** *(done, ROCm/gfx1151)* — `GenerateROCMControlForKernel`
  (`--generate-rocm-control-for-kernel`) lowers an **elementwise-body**
  `control_for` (1-D f32 carry, no captures) to **one** `gpu.func`: grid over the
  carry's elements, each thread runs the loop's `scf.for` (K iterations) with the
  body translated to scalar `arith`/`math`. The whole loop is a single dispatch,
  not one launch per iteration. The chain `generate → convert-scf-to-cf →
  convert-gpu-to-rocdl → rocdl-attach-target{gfx1151} → gpu-module-to-binary →
  hsaco` runs via `hipModuleLaunchKernel` and is **proven on gfx1151** by
  `tests/unit/test_rocm_control_for_exec.py` (`add(c,c)` × K = element·2^K, exact
  for K=1/4/10). Non-elementwise bodies (matmul/softmax/norm) and captures are
  left for the guard / CF4c.
- **CF4c** *(control_if done, ROCm/gfx1151)* — the same
  `GenerateROCMControlForKernel` lowers an elementwise-branch `control_if` to one
  `gpu.func`: grid over the data elements; per thread, `r = (FLAG[0] > 0) ?
  then_scalar(x) : else_scalar(x)` (a per-thread `scf.if`, branch selected once
  by the shape-(1) flag). `@then`/`@else` must be distinct, single-arg, rank-1
  f32, elementwise (shared/non-elementwise branches are left for the guard). The
  `(X, FLAG, O, N)` kernel runs on gfx1151 — proven by
  `tests/unit/test_rocm_control_if_exec.py` (`flag>0 → relu(x)`, `flag<0 → 2x`).
- **CF4c-cont** *(control_while done, ROCm/gfx1151)* — the same kernel-gen
  lowers an elementwise-body `control_while` to one `gpu.func`: grid over the
  carry elements; per thread a **bounded `scf.while`** over `(counter, carry)` —
  `while (i < max_iters AND cond(c) > 0) { c = body(c); i++ }`, with the cond
  **short-circuited** behind the bound (evaluated only inside an `scf.if`, like
  CF2c). `@body` is `(carry)->carry`, `@cond` is `(carry)->pred`, both single-arg
  rank-1 f32 elementwise; exactly one carry operand (no captures). The `(X, O, N)`
  kernel runs on gfx1151 — proven by `tests/unit/test_rocm_control_while_exec.py`
  (`add(c,c)`×`max` with `cond=sigmoid` → `c·2^max`; with `cond=relu` → early
  stop `x>0 ? x·2^max : x`). **All three control constructs now execute on
  device.**
- **CF4e-1** *(control_scan done, ROCm/gfx1151)* — the 4th control primitive now
  executes on device. `tessera.control_scan` is promoted to a first-class Graph IR
  op (ODS + verifier in `TesseraOps.{td,cpp}`: operands `(init, xs)`, results
  `(carry_out, ys)`, `trip`/`carry_arg_index` + optional run_graph_scan payload),
  and `GenerateROCMControlScanKernel` (`--generate-rocm-control-scan-kernel`)
  lowers an **elementwise-body** scan `(carry, y) = body(carry, xs[t]); ys[t] = y`
  (carry/xt rank-1 f32 width K; xs/ys are T×K) to one per-thread `gpu.func`:
  thread g owns carry element g and runs the trip loop locally, reading
  `xs[t*N+g]`, updating the carry, writing `ys[t*N+g]`. The per-step **xs input +
  stacked ys output** is exactly what scan adds over `control_for`; the whole trip
  is one dispatch. Proven on gfx1151 by `tests/unit/test_rocm_control_scan_exec.py`
  (running-sum `carry+xt` and gated `tanh(carry+xt)` bodies, carry **and** stacked
  ys bit-exact).
- **CF4e-2** *(linear SSM scan done, ROCm/gfx1151)* — the scan op gains a variadic
  `captures` operand (loop-invariant weights/biases the body closes over, distinct
  from the per-step `xs`), so the body is `(carry, xt, captures...) -> (carry', y)`.
  `GenerateROCMControlScanGemvKernel` (`--generate-rocm-control-scan-gemv-kernel`)
  lowers the canonical **linear state-space / linear-attention state update**
  `h_t = h_{t-1} @ W + x_t`, `y_t = h_t` (1×K carry, K×K capture `W`, per-step
  `x_t`) — a GEMV body (a reduction over the carry, so cross-element) + a capture +
  the per-step xs. Reuses the CF4d-1 cooperative-workgroup GEMV substrate (h in
  LDS, barrier per step) with the CF4e-1 stacked-ys streaming. Proven on gfx1151 by
  `tests/unit/test_rocm_control_scan_gemv_exec.py` (carry + stacked ys bit-exact vs
  the numpy recurrence).
- **CF4e-3** *(nonlinear RNN-cell scan done, ROCm/gfx1151)* — the full Elman/GRU-
  style recurrent cell as a scan: `h_t = tanh(h_{t-1} @ W + x_t @ U + b)`,
  `y_t = h_t` (1×K carry, two K×K captures `W`/`U`, a 1×K bias `b`, per-step `x_t`).
  `GenerateROCMControlScanRnnKernel` (`--generate-rocm-control-scan-rnn-kernel`)
  fuses the **two GEMVs** — `h@W` over the LDS carry and `x@U` over the per-step
  input — into one per-step reduction, then `+ b`, `tanh`, the barrier handoff and
  stacked-ys store. Builds directly on the CF4e-2 capture substrate (`control_scan`
  already carries variadic captures); the new piece is the second GEMV (over `x_t`)
  + bias + activation. Proven on gfx1151 by
  `tests/unit/test_rocm_control_scan_rnn_exec.py` (carry + stacked ys bit-exact vs
  the numpy `tanh(h@W + x@U + b)` recurrence). This is a real recurrent cell —
  RNN/GRU hidden-state evolution — running as one device dispatch.
- **CF4f** *(cross-element `control_while` done, ROCm/gfx1151)* — power iteration /
  fixed-point: `h = h @ W` **while `Σh > eps`**, bounded by `max_iters`, over a 1×K
  carry and a K×K capture `W`. Both the body (a GEMV) and the continuation cond
  (`Σh`) are reductions over the whole carry, so this is not the per-thread
  elementwise `control_while` (CF4c-cont). `GenerateROCMControlWhileGemvKernel`
  (`--generate-rocm-control-while-gemv-kernel`) generates the kernel **directly**
  (not via CF2's SCF lowering), which dissolves the two reasons this was initially
  deferred:
  1. **Uniform continuation.** A matmul body couples every element each step, so
     the loop can't freeze elements independently. The kernel makes the carry live
     in LDS and has **every thread compute the same `Σ_k lds[k]` reduction and the
     same predicate** (`i < max ∧ Σ > eps`) — so the whole workgroup loops the same
     number of times. That uniformity is exactly what makes the per-iteration
     `gpu.barrier`s safe (a divergent per-element cond would deadlock the barrier).
  2. **Capture threading.** The pass reads the `control_while` op directly and
     wires `W` as a kernel arg (like CF4d-1), so the `W` capture the elementwise
     while-path lacks is threaded here.
  `cond` is `reduce(h){kind="sum"}`; `eps` rides the discardable
  `tessera.while_cond_eps` attr (no change to the shared `control_while` ODS).
  Proven on gfx1151 by `tests/unit/test_rocm_control_while_gemv_exec.py` —
  bit-exact for BOTH an early stop (`Σh` drops below `eps` before `max_iters`) and
  a run to `max_iters`. The cross-element control-flow track now has no remaining
  deferrals.
- **CF3 / cross-element** — `control_while` payload decode (CF4a-cont-2),
  cross-element bodies (matmul/norm), and the CUDA mirror; retire the CF0 guard
  lane by lane.
- **CF3 / broader CF4** — retire the §5 diagnostic lane by lane as executable
  CUDA / ROCm control-flow kernels (scan/for/while/cond proofs) are validated
  against the §1 eager reference.
