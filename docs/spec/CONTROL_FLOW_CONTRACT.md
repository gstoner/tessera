---
status: Normative (CF0 slice)
classification: Spec
authority: INTEGRATED_COMPILER_PLAN.md W4; this file owns the semantic contract
last_updated: 2026-09-06
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

> **Status:** the original CF0 plan is
> [archived](../audit/roadmap/archive/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md);
> active structured-execution work is W4 in the integrated compiler plan.
> This file owns the semantic envelope of the public control primitives.
> Implementation status is layered: traced `control_*` operations, portable
> native SCF lowering, backend packaging, and exact-device execution are
> separate contracts. The integrated compiler plan owns the active queue.

The original CF0 history is retained in §7. Its milestone labels are historical;
§4 is the current status inventory. Native `scf.*` lifetime and AD work does not
by itself establish complete public `tessera.control` support on a backend.

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

These are the current frontend tracing boundaries. The Python wrappers can
choose eager reference execution when tracing is inapplicable; that does not
count as compiled backend execution. Once a compiled path is selected, an
unsupported form must be rejected rather than reported as device execution.

### `scan`
- **Static trip count.** `length` (or `xs.shape[0]`) is a Python int at trace
  time (`trace.py`).
- **Static carry shape.** `fn` must return `(carry, y)` with
  `carry.shape == init.shape` (`trace.py`).
- `xs` must be a single `Tracer` with a leading scan axis; `reverse=True` and
  pytree-`xs` fall back to the host reference (`control.py`).
- Stacked output `ys` has static shape `(trip, *y.shape)`.

### `fori_loop`
- **Static bounded trip count.** `trip = upper - lower`, both Python ints
  (`trace.py`). Symbolic bounds fall back to the eager loop.
- **Static carry shape.** body must return a `Tracer` of the init carry shape
  (`trace.py`). The `control_for` ABI is index-independent: the body is
  traced once at `i=0`.

### `while_loop`
- **Bounded maximum required under trace.** `max_steps` is mandatory
  (`trace.py`) and lowers to `control_while`'s `max_iters` (verifier
  requires `max_iters > 0`, `TesseraOps.cpp`). An unbounded traced
  `while_loop` is a hard error — there is no way to bound the device loop
  otherwise.
- **Static carry shape**, single carried `Tracer` (`trace.py`); the
  predicate `cond(carry)` must return a `Tracer`.

### `cond`
- **Scalar predicate** carried as the flag arg (a shape-`(1,)` tensor;
  `flag[0] > 0` selects `then`, `TesseraOps.td` `ControlIfOp`).
- **Same-shaped branch results** — `t_fun` and `f_fun` must return `Tracer`s
  with matching shape and dtype per result. `record_cond` also accepts equal-arity
  tuples of Tracers; this is not general pytree loop carry support. Divergent execution (only the taken
  branch runs), not a data-parallel select.

### Known frontend boundaries
- ~~The trace checks carry/branch **shape** equality but not yet **dtype**
  equality.~~ **Closed in CF1.** The trace now also enforces carry/branch
  **dtype** equality on the for/while/scan/cond paths (`record_for_loop` /
  `record_while` / `record_scan` / `record_cond` in `trace.py`), matching the
  Graph IR verifiers' full-type match (`control_for` result vs carried iter_arg
  type, `TesseraOps.cpp`). A body that drifts the carry dtype (e.g. an
  in-body cast) now fails at trace time with a clear `TesseraTraceError` instead
  of slipping to the later MLIR verifier. Covered by
  `tests/unit/test_cf1_control_flow_dtype.py`.
- Loop carry is a single `Tracer` under trace — **general pytree loop carries are not yet traced**
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

The explicit-state bullets describe required semantics; they do not assert that
all current payload decoders accept RNG or cache operations. When a body op
carries a non-pure effect, the op's `[Pure]` trait must be
relaxed and the `EffectLattice` walk (`pure<random<memory<io<top`) must classify
the loop accordingly — tracked with the effect work, not assumed here.

---

## 4. Current lowering and execution boundaries

### Portable native lowering

`src/transforms/lib/LowerControlFlowToSCFPass.cpp` lowers supported
`control_for`, `control_if`, bounded `control_while`, and `control_scan` forms
into native `scf.for` / `scf.if` / `scf.while`. Scan carries recurrent state and
stacked outputs; SAVE mode materializes residual state. Registered payload
materialization and branch/carry contracts must succeed first. Unknown payloads
or unsupported envelopes are not made executable by SCF registration.

The named pipeline builders in `src/transforms/lib/Passes.cpp` order supported
control lowering before the target guard. This is shared compiler capability,
not proof that every resulting tensor body can be packaged for every backend.

### Backend execution evidence

| Backend | Existing execution paths | Boundary still requiring work |
|---|---|---|
| Apple GPU | Direct `ControlForToAppleGPU`, `LowerControlIfToAppleGPUPass`, `ControlWhileToAppleGPU`; runtime `run_graph_scan_f32` path | These are distinct native/runtime routes with bounded payload support. They do not establish general native tensor tape execution or all public control forms. |
| NVIDIA CUDA | `emit/nvidia_cuda.py` has narrow `run_control_{for,if,while,scan}_f32` CUDA-source helpers (source/ABI tests in `test_nvidia_control_flow_contract.py`, not new device proof here). Native GPU SCF executes in the storage/AD, ring and resident attention packages recorded in `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` | The old table's “none” is wrong. Handwritten CUDA helpers remain a separate historical route from the MLIR/LLVM foundation. These proofs do not establish complete frontend `control_for/if/while/scan` → production package coverage. General persistent nested tensor tapes remain open. |
| AMD ROCm | CF4 elementwise for/if/while/scan; cooperative GEMV and norm loops/branches, linear and nonlinear recurrent scans, bounded GEMV while; paired bounded state-machine kernels (`test_rocm_state_machine_exec.py`); native storage/lifetime packages | Each generator accepts its own shape/body/capture envelope. Neither the rank-1 CF4 proofs nor CUDA results imply general ROCm closure. See `tests/unit/test_rocm_control_*_exec.py`. |
| x86 | Shared control-to-SCF lowering; recorded bounded state-machine forward/backward execution through linalg → bufferization → LLVM → ORC JIT on the AVX-512 host (`test_x86_state_machine_exec.py`, W4-PRODUCT-1) | This bounded native compute proof is distinct from the host sizing companion. Full public control packaging, general persistent host tapes and clean physical timing remain separate requirements. Eager Python loops remain reference execution. |

Apple is **not** the only backend with device control flow. Avoid a single
boolean “supported” column: it conflates graph syntax, native lowering, package
binding and measured execution. Backend-specific schedules and device evidence
must not be transferred between architectures.

### AD and lifetime boundary

`AutodiffForwardPass` and `AutodiffPairedPass` consume supported native structured
regions. The paired `export-product=forward|backward` option preserves full
residual types and common lineage, including nested SAVE-loop and saved if/while
artifacts. CUDA/HIP invocation snapshots and CUDA resident attention LSE have
separate ownership implementations. These are not a general persistent nested
control-flow tape implementation. Dynamic iteration paths, residual allocation
capacity, branch selection and completion-owned lifetimes require a real native
consumer and backend-specific execution proof.

Current work and evidence: [integrated compiler plan](../audit/compiler/INTEGRATED_COMPILER_PLAN.md)
and [native storage follow-up](../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

---

## 5. Unsupported compiled forms

**A `control_*` form/envelope that a target cannot lower must fail with a stable
diagnostic.** The ROCm backend already owns the Decision #21 pattern for
contracts a target cannot lower — `op->emitError("ROCm lowering does not support
TMEM operations")` (`TileToROCM.cpp:343`), `tessera.target.diagnostic` for
KV-cache (`Tessera_ROCM_Backend/README.md`). Per Decision #21 (unsupported
lowering must emit a stable diagnostic naming the op and the target, never
silently no-op), the registered guard emits:

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
  `tessera-nvidia-pipeline*` (target `nvidia_sm90`), after supported portable lowering (the precise pass order is owned by
  `Passes.cpp`) so an unsupported control-flow program fails
  before any confusing downstream pass. ROCm CF4 handles the narrow no-capture
  rank-1 elementwise proof kernels; the guard remains the contract for
  everything outside that envelope and for targets without a device lowering.

This guarantees that unsupported control-flow forms fail loudly at compile time
rather than producing an executable-backend claim that silently fell back to a
host loop.

---

## 6. Frontend and target rejection checks

Each row is a form **outside** the §2 envelope; the contract is a clear
trace-time rejection (`TesseraTraceError`), proven by a test.

| Rejected form | Where caught | Message intent |
|---|---|---|
| `while_loop` under trace without `max_steps` | `trace.py` | "needs a bound: pass max_steps=N" |
| `fori_loop` body changes carry shape | `trace.py` | "body must preserve the carry shape" |
| `while_loop` body changes carry shape | `trace.py` | "body must preserve the carry shape" |
| `scan` body changes carry shape | `trace.py` | "body must preserve carry shape" |
| `cond` branches return different shapes | `trace.py` | "branches must share a shape" |
| `cond`/`scan`/loop body returns a non-Tracer (host object capture / data-dependent value) | `trace.py` | "must return a Tracer" |
| `control_*` form/envelope lowered to an unsupported target path | CF0 diagnostic (§5) | "no lowering for this form/envelope on target '<t>'" |

Acceptance: the eager reference behavior is unchanged (existing
`control.py`-level tests remain the oracle), and the JIT/trace path either emits
a first-class `control_*` op for a supported device lowering or produces the §5
diagnostic — never a silent host-loop fallback inside an executable backend
claim.

---

## 7. Lifecycle and evidence ownership

The original CF0–CF4 milestone sequence is historical. Its original plan is
[archived](../audit/roadmap/archive/CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md).
Current structured execution and tape work is owned by W4 and W2.4a / CAKE / SO-2
in the integrated compiler plan, with backend-specific work in the four backend
`todo.md` queues. A parser fixture, SCF rewrite or host sizing function may close
its own compiler contract; none alone closes a device execution item.

When adding a form, update its frontend acceptance, ODS/effect contract, native
lowering, backend package mapping, AD residual requirements and numerical
reference tests. Unsupported forms must remain explicitly rejected on compiled
paths. Reference execution must remain identified as reference execution.

### Persistent counted-while normalization

For physical split-tape materialization, paired AD optionally normalizes a
provably counted `scf.while` into `scf.for`: the counter begins at zero, advances
by one, and is compared with a constant positive bound (2–1024), with no other
condition-region operations. The declared capacity must cover the actual bound.
Checkpoint policy is retained; SAVE positions become the interior iteration
ordinals. The resulting native tensor products execute through the bounded
CUDA/HIP tape path. This does not admit data-dependent termination, unknown
counter increments, scalar predicate residuals or unchecked maximum annotations.


### Persistent data-dependent exits (2026-09-07)

The optional native paired-AD `normalize-data-while` path accepts a pure
single-block while only when its zero-origin, unit-increment counter has an
actual signed constant upper-bound conjunct. A fixed-capacity for loop freezes
all carried state after the first false predicate. The existing checkpoint
machinery then records discrete and differentiable state separately. This
supports data-dependent exit within a proven capacity; it does not infer a bound
from `max_iters`, recover arbitrary source CFGs, or admit shape-varying device
residuals. The physical export's `box-product-scalars` option stores index and
predicate residuals as i64/i8 tensors without assigning them cotangents.


The native counter proof also accepts a false-else short-circuit `scf.if` or
`arith.select`; it rejects a true or unknown fallback. Native x86 exported tapes
now execute bounded shape-varying slices with saved logical extents and dynamic
adjoint zeros. This does not change the frontend's shape-preserving carry
contract or enable dynamic CUDA/HIP persistent slots. See the integrated plan's
2026-09-07 shape-tape increment for exact evidence and remaining CFG boundaries.
