# Target IR Membership — does an op *require* its contract?

**Generated. Do not hand-edit.** Regenerate with
`python -m tessera.compiler.generated_docs --write`.

Decision #19 (amended 2026-08-30) says the `tessera_<backend>` layer
exists for **contract carriage**. This measures whether that holds.

**Corrected 2026-08-30: the first published totals were wrong.**
This audit reported "38 of 138" before two defects were found in
its own parser: the contract vocabulary could not see `memory_scope`
or `leading_dim` (real contracts, uncounted), and a brace-less
`def X : Base<...>;` matched forward into the next operation's body
and **consumed that operation**, so ops vanished with no error.
Both produced a confident number rather than a failure, which is why
each now has a named regression test.

`optional-only` is the row to read. Those ops declare their contract
attributes as `OptionalAttr`, almost always inherited from the
dialect's op base class, which hands the same bag to every op. The
contract is *expressible* but not *enforced* — the op verifies with
`arch`, `accum` and `dtype` all unset. That **fails open**: it is
Decision #32 information loss (silently absent rather than declared
dropped) and Decision #21a (a semantic key never defaults), violated
by construction rather than by mistake. Such an op satisfies #19's
membership test on paper and carries nothing in practice.

**45 of 147 ops require the contract they carry.**

| Backend | requires | optional-only | no contract |
|---|---|---|---|
| `nvidia` | 5 | 21 | 2 |
| `rocm` | 31 | 49 | 0 |
| `x86` | 5 | 0 | 8 |
| `apple` | 4 | 12 | 10 |

## What this means for the operator expansion

#19's amendment invites Apple and x86 to grow their Target IR, and
expansion is exactly when this regresses: the cheapest way to add an
op is to derive it from the base class and inherit the optional bag,
which adds surface that *looks* contract-carrying and is not.
**A new op should declare its contract attributes as required.**

Apple is the sharpest case — it requires nothing at all, which is the
same gap as its missing machine primitives seen from the contract
side: a dialect of dispatch containers has no contract to enforce.

## Ops requiring no contract at all

| Backend | Op |
|---|---|
| `nvidia` | `philox` |
| `nvidia` | `func` |
| `x86` | `amx_tile_zero` |
| `x86` | `amx_dpbf16ps` |
| `x86` | `amx_dpbusd` |
| `x86` | `elementwise` |
| `x86` | `kernel` |
| `x86` | `kv_cache_read` |
| `x86` | `unsupported` |
| `x86` | `profiler_probe` |
| `apple` | `gpu.dispatch` |
| `apple` | `gpu.mps_dispatch` |
| `apple` | `diagnostic` |
| `apple` | `gpu.control_loop` |
| `apple` | `gpu.control_if` |
| `apple` | `gpu.control_while` |
| `apple` | `cpu.kv_cache_read` |
| `apple` | `cpu.moe_solver` |
| `apple` | `cpu.profiler_probe` |
| `apple` | `gpu.profiler_probe` |
