# Apple GPU control-flow lowering (Phase G) — design + ladder

> Status: **Rungs 0 + 1 + 2 landed** — Rung 0 (bounded `scan` → MPSGraph
> `forLoop`), Rung 1 (the Gumiho serial draft as one `forLoop`), and Rung 2
> (predicate-driven `while` generation with EOS early-stop). Rung 3 (the
> dynamic frontier) remains out of scope. This is the on-Apple slice of the
> Phase-G "single-kernel `@jit` of a control-flow loop" frontier.

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
- **Rung 2 — predicate-driven `while` generation (landed).** A
  `whileWithInitialInputs:before:after:` greedy-generation loop with a
  **data-dependent** trip count: `token = argmax((hidden = tanh(hidden @ W)) @
  lm)` looped until the EOS token or `max_steps`. The `before` block returns the
  scalar predicate `(step < max) AND (last_token != eos)`; the `after` block
  runs the body and scatters each token into a fixed-capacity `[max_steps]`
  buffer. `tessera_apple_gpu_cf_while_generate_f32` returns the tokens + the
  actual count. This is the variable-trip control-flow primitive — distinct from
  the fixed-trip `forLoop`. The body is intentionally small (the dynamic
  verify/accept of a real speculative step stay host-side).
- **Rung 3 — frontier.** The dynamic trie/top-k/variable-accept parts. Needs a
  treeless fixed-shape reformulation or a monolithic MSL kernel. Explicitly
  out of scope until an algorithm change makes it static-shaped.

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
