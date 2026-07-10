# Apple GPU control-flow lowering (Phase G) — design + ladder

> Status: **Rungs 0 + 1 + 2 landed; Rung 3 first slice via MSL.** Rung 0
> (bounded `scan` → MPSGraph `forLoop`), Rung 1 (the Gumiho serial draft as one
> `forLoop`), Rung 2 (predicate-driven `while` generation with EOS early-stop),
> and Rung 3's dynamic speculative-accept control flow as a single MSL kernel
> (`tessera_apple_gpu_msl_spec_accept`) — the part MPSGraph can't express. This
> is the on-Apple slice of the Phase-G "single-kernel `@jit` of a control-flow
> loop" frontier.

> **Native Metal validation note:** run hardware proof tests outside sandboxed
> command environments and in a fresh Python process with a fresh runtime dylib.
> In particular, Codex sandboxed exec can hide `MTLCreateSystemDefaultDevice()`
> even on Metal-capable Apple Silicon, causing false negatives such as
> `DeviceTensor.is_metal() == False` and skipped Metal stress tests. Use
> `TESSERA_APPLE_GPU_RUNTIME_LIB` or the freshly built/temp
> `libTesseraAppleRuntime.dylib` when validating native Metal behavior.

## The problem

`tessera.control.{scan, fori_loop, while_loop, cond}` are **Python-reference
only** today — they run as host loops in `python/tessera/control.py`. There is
no Graph-IR control-flow op and no per-backend lowering, so a `@jit` of a loop
executes its body op-by-op from the host: every iteration pays a
host↔GPU dispatch + sync per op. "Single-kernel `@jit` of the whole loop" means
lowering the control flow itself onto the device so the loop body and its trip
control run in **one dispatched graph**.

The canonical motivating case is the Gumiho speculative-decoding loop
(`examples/advanced/gumiho`), whose inner serial draft is a bounded
autoregressive recurrence.

## What the Apple GPU backend can express

The Apple backend is **MPSGraph** (a static-shape dataflow graph compiled to one
GPU executable) + custom **MSL** kernels. MPSGraph ships real control-flow ops:

| MPSGraph op | Maps to |
|---|---|
| `forLoopWithLowerBound:upperBound:step:bodyBlock:` | `fori_loop` / bounded `scan` (static trip count) |
| `whileWithInitialInputs:beforeBlock:afterBlock:` | `while_loop` (predicate-driven, **static carry shapes**) |
| `ifWithPredicateTensor:thenBlock:elseBlock:` | `cond` / `switch` |

The hard constraint: **every tensor in the loop body and carry must have a
static shape.** The body is an ordinary MPSGraph subgraph (matmul, reductions,
`gatherWithUpdatesTensor`, `reductionArgMaximum`, …), so anything expressible as
fixed-shape dataflow can be a loop body — but **dynamic allocation,
dynamic-length sequences, and data-dependent shapes cannot.**

## What lowers, and what does not

| Speculative-loop construct | Shape | Lowerable to one Apple GPU graph? |
|---|---|---|
| Serial draft (autoregressive, `serial_tokens` steps) | fixed trip count, fixed carry `[d]`+token | ✅ `forLoop` (Rung 1) |
| Parallel heads | fixed, no loop | ✅ already one graph chain |
| Per-position target verify | fixed `[C+N, …]` | ✅ static (no control flow) |
| Greedy/while decode to `max_new_tokens` | fixed trip count | ✅ `forLoop` (Rung 2 if carry shapes fixed) |
| Acceptance (`cumprod` prefix + argmax) | fixed `[P, depth]` | ✅ static |
| **FTA top-k path selection** | variable combo set | ❌ no GPU top-k; host control flow |
| **Prefix-trie construction** | dynamic node count, pointer-chasing | ❌ dynamic allocation — not graph-expressible |
| **Variable accepted-prefix length** | data-dependent trip | ⚠️ expressible only via fixed-upper-bound `while` + masking |

So the **full** single-kernel speculative loop is **not** achievable without
algorithmic restriction: the trie and top-k are dynamic-allocation /
pointer-chasing control flow that no static-shape graph (MPSGraph or otherwise)
can express. A fully on-device speculative step would require either (a) a
fixed-shape "treeless" draft (enumerate a fixed candidate set, no trie, no
top-k — accept by masking) or (b) a hand-written monolithic MSL kernel. Both are
research-grade and out of scope here. What *is* tractable and worth doing is the
**bounded, static-shape inner loops** — which is where this ladder starts.

## The ladder

- **Rung 0 — control-flow primitive (this PR).** Prove MPSGraph control flow
  lowers to one dispatch: a bounded `scan` /
  `carry_{i+1} = act(carry_i @ W_h + x_i @ W_x)` over a fixed trip count, built
  with `forLoopWithLowerBound:…` and emitting per-step outputs via an
  index-scatter accumulator. Runtime symbol
  `tessera_apple_gpu_cf_scan_f32`; validated bit-close against a numpy scan.
  This is the reusable machinery (body subgraph + carry threading + per-step
  accumulation) every higher rung needs.
- **Rung 1 — serial draft as a forLoop (landed).** Reuses Rung 0's machinery
  with the serial-block body (`fc_in → [RMSNorm → value-attn → SwiGLU →
  residual]×L → RMSNorm → LM head → argmax → embed-gather`), carry =
  `(hidden, token)`, accumulating per-step tokens + hiddens via index-scatter.
  `tessera_apple_gpu_cf_serial_draft_f32` runs the whole autoregressive serial
  draft in **one** `forLoop` dispatch (vs ~`(2+10L+1)·T` per-op host
  dispatches), validated token-for-token against the host `SerialHead`
  (hidden err ~6e-7). Exposed as `gumiho.validate_serial_forloop` /
  `serial_draft_forloop`; `demo.py --mode forloop`.
- **Rung 2 — predicate-driven `while` generation (landed; MSL since 2026-06-04).**
  A greedy-generation loop with a **data-dependent** trip count:
  `token = argmax((hidden = tanh(hidden @ W)) @ lm)` looped until the EOS token
  or `max_steps`. The predicate is `(step < max) AND (last_token != eos)`.
  `tessera_apple_gpu_cf_while_generate_f32` returns the tokens + the actual
  count. This is the variable-trip control-flow primitive — distinct from the
  fixed-trip `forLoop`. The body is intentionally small (the dynamic
  verify/accept of a real speculative step stay host-side).
  **Originally an MPSGraph `whileWithInitialInputs:before:after:`, it was moved
  to a single hand-written MSL kernel with a native `while` loop (the same
  single-thread pattern as the Rung-0 `cf_scan_msl` and Rung-3 `spec_accept`
  kernels).** The MPSGraph `while` route crashed (SIGSEGV) inside MPSGraph's own
  `GPU::WhileOpHandler` constructor during lazy graph specialization — it ran in
  isolation but faulted once enough MPSGraph executables had churned through the
  process (reproduced by `tests/unit/test_apple_gpu_control_flow_stress.py`,
  which interleaves `bmm` + while-generate dispatches). Because the loop is
  bounded and the per-step work is tiny (`d ≤ 256`), the whole sequential
  generation now runs in one MSL thread, dispatched through the stable
  `commit_and_wait_with_timeout` path. argmax streams over the vocabulary, so
  `V` is unbounded; the hidden carry lives in a `[256]` thread-local array,
  matching the documented `d ≤ 256` control-flow envelope (`d > 256` falls back
  to the host numpy loop).
  Native confirmation of this rung requires Metal access in the current process;
  a sandboxed command runner may report no default Metal device even on a Metal
  4-capable M1/M2/M3/M4 host.
- **Rung 3 — dynamic frontier, via MSL (first slice landed).** Out of scope for
  the MPSGraph route, but the MSL route reaches it (see the mapping below).
  `tessera_apple_gpu_msl_spec_accept` runs the **dynamic speculative-verify
  control flow** — per-path accept-while-match with an early `break` (a
  data-dependent trip count), cross-path longest-prefix selection, and the bonus
  token — as a **single MSL kernel over fixed-capacity candidate buffers** (no
  trie, no heap). This is exactly the variable-trip / data-dependent control
  flow MPSGraph's static-shape graph can't express, validated against a numpy
  reference. The remaining dynamic pieces (full draft-model forward feeding it,
  top-k path *generation*) compose the static MPSGraph kernels with this MSL
  accept kernel.

## Mapping to MSL 4.0 (Metal 4) — a second lowering route

There are **two** ways to put a control-flow loop on the Apple GPU, and they
have very different expressiveness:

1. **MPSGraph control-flow ops** (Rungs 0–2). The loop is a graph node
   (`forLoop` / `while` / `if`); the body is a static-shape dataflow subgraph.
   Mature, handles the matmul/reduction bodies for free — but **every tensor in
   the body and carry must have a static shape**, which is what puts dynamic
   allocation / data-dependent shapes out of reach.

2. **Hand-written MSL with native in-kernel control flow** (Metal 4, M2). The
   loop is an ordinary `for` / `while` / `if` *inside one MSL kernel*, dispatched
   through the MTL4 command model. Only the kernel's **I/O buffer** shapes are
   fixed; everything inside the kernel — trip counts, indices, branches — is
   ordinary, fully data-dependent code.

`tessera_apple_gpu_mtl4_scan_f32` (M2) implements the Rung-0 recurrence the
*second* way — the scan loop is a literal MSL `for` loop, one thread runs the
whole sequence — and it matches the MPSGraph `forLoop` scan bit-close (~1e-7).
So the two routes are interchangeable for the static cases:

| Phase-G construct | MPSGraph route | MSL 4.0 route |
|---|---|---|
| `scan` / `fori_loop` (Rung 0/1) | `forLoopWithLowerBound:` | native `for` loop ✅ (M2) |
| `while_loop` (Rung 2) | `whileWithInitialInputs:` | native `while` loop |
| `cond` / `switch` | `ifWithPredicateTensor:` | native `if` |
| **variable trip count / early break** | fixed upper bound + masking only | **native — real data-dependent loop** |
| **data-dependent indexing** | gather/scatter, static shapes | **native pointer/index arithmetic** |
| **dynamic allocation (trie)** | ❌ impossible | ⚠️ via a fixed-capacity preallocated buffer + in-kernel cursor |

The key consequence: **MSL native control flow is strictly more expressive than
MPSGraph's static-shape loops.** The Rung-3 blockers — variable accepted-prefix
length, data-dependent FTA path selection, the prefix trie — are *control flow*,
not arithmetic, and MSL expresses arbitrary control flow inside a kernel. The one
thing MSL still can't do is heap allocation, but a draft tree of bounded size
fits a **fixed-capacity preallocated buffer** with an in-kernel write cursor — so
a *treeless / bounded-tree* speculative step becomes a single MSL kernel rather
than the "research-grade" item the MPSGraph route declared it. That is the path
to Rung 3, now that the Metal 4 MSL-kernel dispatch (M2) exists. The trade-off is
that the MSL route is hand-written (no automatic shape/dtype coverage, no free
matmul fusion), so it earns its place only where the MPSGraph route can't go —
i.e. the dynamic frontier.

## Compiler-track tie-in

Rung 0 ships the **runtime capability**. The Graph-IR side — a
`tessera.control.scan` op that lowers to this executable through the
`tessera-lower-to-apple_gpu` pipeline — is the follow-on once a second consumer
exists (Decision #19's hardware-free Target IR layer is where a
`tessera_apple.gpu.scan` op would sit). Until then the runtime symbol is the
contract, exercised directly (the same path the MPSGraph Tier-1 ops took before
their dialect ops landed).

## Acceptance

- `tessera_apple_gpu_cf_scan_f32` runs a fixed-trip recurrence in **one**
  MPSGraph executable (one `runWithMTLCommandQueue`), validated against a numpy
  reference to f32 tolerance, with per-step outputs recovered.
- Non-Darwin stub parity keeps the symbol callable everywhere.
- A unit test locks the numeric contract and the single-dispatch property.
