---
status: Normative
classification: Backend contract
audience: backend authors (Apple proven; NVIDIA / ROCm inherit)
last_updated: 2026-06-03 (Apple Value Target IR sprint)
---

# Value Target IR Contract

Tessera Target IR has two complementary shapes per backend:

1. **Artifact / inspection ops** — attribute-only (no SSA operands or results).
   They are metadata for dashboards, audits, and lit inspection. They do *not*
   preserve dataflow; a pipeline that emits only artifact ops is a *projection*,
   not a semantics-preserving lowering. Artifact pipelines may use `ub.poison`
   as an honest husk for any Tile-op results that must be consumed to keep the
   module valid.

2. **Value ops** — carry the original SSA tensor operands and results, so the
   lowering can `replaceOp(tileOp, valueOp.getResults())` and the rest of the
   function (including `func.return`) keeps consuming real values. Value ops are
   what make Graph → Schedule → Tile → Target an *executable proof spine*: the
   value op names the runtime C ABI `symbol`, and the runtime dispatches to it.

This contract is **proven first on Apple** (CPU is the executable proof lane via
Accelerate/LAPACK symbols that already exist). NVIDIA and ROCm **inherit the
same shape** — copy the pattern, swap the backend-specific attrs — rather than
each redesigning Target IR.

## Value-op shape (backend-neutral)

Each backend defines value ops with:

- **operands**: `Variadic<AnyType>` — the original SSA tensor inputs.
- **results**: `Variadic<AnyType>` — the original SSA result types
  (multi-result ops, e.g. `svd → U,S,V`, are first-class: the value op produces
  all results directly).
- **attributes**:
  - `op_kind` (required) — the logical op (`"cholesky"`, `"matmul"`, …).
  - `symbol` (required) — the C ABI entry the runtime dispatches to.
  - `status` (required) — `"executable"` iff a runtime dispatcher exists for
    this op on this backend; otherwise the value-only `-full` pipeline must
    **fail with a named diagnostic** rather than silently emit an artifact.
  - `abi`, `dtype`, `framework`, optional `argument_layout` — dispatch detail.
- **assembly**: `operands attr-dict `:` functional-type(operands, results)`.

### Apple reference ops (`tessera_apple` dialect)

| Value op | Lane | Example symbol |
|----------|------|----------------|
| `tessera_apple.cpu.call` | Accelerate / LAPACK / BNNS | `tessera_apple_cpu_cholesky_f32` |
| `tessera_apple.gpu.kernel_call` | custom MSL kernel | `tessera_apple_gpu_cholesky_f32` |
| `tessera_apple.gpu.package_call` | authored `.mtlpackage` (PK8) | `tessera_apple_gpu_svd_f32` |

The attribute-only artifact ops (`cpu.vector_op`, `gpu.metal_kernel`,
`gpu.dispatch`, …) remain unchanged for dashboards / compatibility.

## Pipeline intent

- `tessera-lower-to-<target>` — **artifact** projection (inspection). May emit
  metadata ops + `ub.poison` husks.
- `tessera-lower-to-<target>-full` — **value-preserving**. Emits value ops only.
  The final module must contain **no** `ub.poison`, **no** `tensor.empty`, and
  **no** surviving `tile.*`. An op with no value lowering ⇒ named diagnostic +
  pass failure (never a silent degrade to artifact).

In the Apple `TileToApple` pass this is a single `valueMode` flag (the `-full`
pipelines pass `valueMode=true`); a backend may instead use distinct passes.

## Front door + runtime

- `canonical_compile` / JIT tag the lowered module:
  `driver.classify_apple_target_ir(ir) → "value_target_ir" | "target_ir_artifact"`.
- The runtime dispatcher reads the dispatch tuple straight off each value op
  (`driver.extract_apple_value_calls(ir)`): for a CPU call it invokes the named
  Accelerate/LAPACK `symbol`; for a GPU call it reports native execution only
  when `driver.apple_value_call_is_executable(call)` (status `"executable"` +
  a symbol). This is the **seam-closure** contract — the executed result is
  produced by the `symbol` named *in the IR*, not by a parallel matcher.

## NVIDIA / ROCm follow-on (not in this sprint)

Inherit the shape:

- `tessera_nvidia.call` / `tessera_rocm.call` (and kernel/package variants) with
  the same value operands/results + `{op_kind, symbol, status, abi, dtype,
  framework}` attrs.
- A `valueMode` (or `-full`) lowering that `replaceOp`s the Tile op and fails
  loudly on unsupported ops.
- Runtime dispatchers that read the value-op attrs and only report native
  execution when `status == "executable"`.

Execution conversion for NVIDIA/ROCm is gated on real hardware; the value
Target IR *shape* is backend-neutral and ready to copy today.
