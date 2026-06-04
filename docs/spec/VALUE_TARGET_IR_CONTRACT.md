---
status: Normative
classification: Backend contract
audience: backend authors (Apple proven; NVIDIA / ROCm inherit)
last_updated: 2026-06-03
note: Sprint 11 — Apple value Tile verifier hardened the -full handoff; native sparse attention added as a strict Apple GPU value envelope when the real Metal executor is active.
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

## Tile IR is strict in Apple value pipelines (Sprints 9 + 11)

The value-mode tiling pass emits **registered** Tile IR ops in the `tile`
dialect (`tile.matmul`, `tile.gemm`, `tile.batched_gemm` with rank/shape
verifiers; `tile.cholesky`/`tri_solve`/`cholesky_solve`/`lu`/`qr`/`svd`). The
Apple value `-full` pipelines therefore run with **no
`--allow-unregistered-dialect`**. That is **necessary but not sufficient**:
Sprint 11 adds an Apple-value-only Tile verifier immediately after value-mode
tiling and before `TileToApple`. It fails if any `tile.*` op is unregistered,
and it fails if a registered Tile op is outside the value allowlist:
the linalg family, rank-2 `tile.matmul`/`tile.gemm`, and rank-3
`tile.batched_gemm`. A guard test injects an opaque `tile.fake_value_op` into
the value lane and proves the verifier rejects it, so the handoff is no longer
dependent on parser behavior alone. The artifact/runtime pipelines are
unchanged for now.

A guard test
(`test_apple_value_lowering_uses_no_unregistered_dialect_flag`) fails if the
flag is reintroduced in the driver, the lit value fixtures, or the test
runner. The `tile` dialect still allows unknown operations so the *artifact*
lane's remaining transient tile ops (`tile.mma`/`async_copy`/`kv_cache`/debug)
stay opaque until their own ODS lands — a documented follow-on, not a hidden
gap for the value handoff.

## Reasoning-model attention prologue (Sprint 10)

The Apple value `-full` pipelines (`tessera-lower-to-apple_{cpu,gpu}-full`) run
the Graph IR attention-family **recognizer** passes — SwiGLU / MLA /
DeepSeek NSA / Ling-Kimi hybrid / Lightning / DeltaNet-Kimi — **before**
distribution and tiling, exactly like the NVIDIA `tessera-nvidia-pipeline`.
This makes reasoning models *compiler-visible* on the Apple spine: the
DeepSeek MLA decode chain (`latent_kv_compress -> expand_k/v -> flash_attn`)
fuses into `tessera.mla_decode_fused`, and the Lightning / Delta / hybrid pass
slots run in a stable position for a future backend rewrite.

**Honesty:** compiler-visible is **not** executable. MLA decode, Lightning,
Delta/Kimi, and hybrid ops have no Apple value lowering yet, so they pass
through the value lane as Graph IR ops — there is no
`tessera_apple.gpu.kernel_call`, no MSL/MPS symbol, and no `ub.poison` husk.
The runtime envelope (`driver._APPLE_GPU_RUNTIME_OPS`) does **not** contain
them, so they are never claimed executable.

Sprint 11 promotes exactly one new reasoning envelope:
`tessera.native_sparse_attn_fused` lowers to
`tessera_apple.gpu.kernel_call` with symbol
`tessera_apple_gpu_native_sparse_attn_f32` only for static fp32 rank-4 Q/K/V/O,
rank-4 gate logits shaped `[B,H,S,S/block_size]`, positive
`window_size`/`block_size`/`top_k`, `S % block_size == 0`, and
`top_k <= S/block_size`. Runtime execution is additionally gated on the real
Darwin/Metal value executor being active; the non-Darwin stub exports the
symbol but zero-fills, so it is rejected for value execution.

The other strict executable reasoning envelope remains the MLA-style block
built from executable primitives (`matmul -> softmax -> matmul` + MPS
projections), proven numerically by `tests/unit/test_apple_gpu_mla_e2e.py`.
Lit: `apple_reasoning_attention_prologue.mlir`. Benchmark + honesty guards:
`benchmarks/apple_gpu/benchmark_reasoning_attention.py`
+ `tests/unit/test_apple_gpu_reasoning_benchmark.py` (reports route / target /
executor / correctness / timing as separate fields; never fabricates a number
for an op it did not run).

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
  `CompileResult.to_runtime_artifact()` records `apple_target_ir_kind`, the
  extracted `apple_value_calls`, and — for the value lane — sets
  `compiler_path = "apple_value_target_ir"` (preserving the prior path as
  `apple_previous_compiler_path`).
- `driver.extract_apple_value_calls(ir)` reads the dispatch tuple off each value
  op with a **brace-safe scanner** (anchors on the mnemonic, walks to the
  matching top-level `}` while skipping braces inside quoted strings — so an
  `argument_layout` whose value is a JSON object survives intact).

### What "runtime dispatches" means today (Sprints 2–11)

This is a **narrow, honest** executable path, not a blanket claim. The boundary
is **CPU linalg + CPU rank-2 matmul (f32/f16/bf16) + CPU fp32 rank-3 batched
matmul + Apple GPU rank-3 batched matmul (f32/f16/bf16) + Apple GPU native
sparse attention (strict fp32 rank-4 envelope, real Metal executor only) =
executable; all other GPU value calls and non-linalg value calls = classified +
gated**.

Two executor allowlists, one per backend lane:
- **CPU** (`test_value_envelope_executable_allowlist_exact`): the six CPU linalg
  symbols, `tessera_apple_cpu_gemm_f32` (rank-2 matmul), `…_gemm_f32_batched`
  (rank-3 batched), and `…_gemm_f16` / `…_gemm_bf16` (rank-2 matmul).
- **GPU** (`test_gpu_value_executor_allowlist_exact`):
  `tessera_apple_gpu_bmm_{f32,f16,bf16}` (rank-3 batched matmul, Sprint 8)
  plus `tessera_apple_gpu_native_sparse_attn_f32` (native sparse attention,
  Sprint 11; active only when the real Metal executor is available).

- **The full Apple CPU linalg family is executable now (Sprint 3).** The matrix
  row `(apple_cpu, apple_value_target_ir)` resolves to the
  `_execute_apple_value_target_ir_artifact` executor, which reads
  `metadata["apple_value_calls"]`, requires a single `tessera_apple.cpu.call`
  with `status == "executable"` and a symbol on the CPU allowlist, and invokes
  that C ABI entry via ctypes (numpy alloc, f32-contiguous, shape-validated).
  The allowlist is all six LAPACK-backed symbols:
  `tessera_apple_cpu_{cholesky,tri_solve,cholesky_solve,lu,qr,svd}_f32`.
  Single-result ops return one ndarray; multi-result ops return a tuple in SSA
  order (`lu→(LU,pivots)`, `qr→(Q,R)`, `svd→(U,S,V)`). The linalg semantic attrs
  `lower`/`trans`/`unit_diag`/`full_matrices` ride the value op and parameterize
  the ABI. The executed result is produced by the `symbol` named *in the IR*,
  not by a parallel op-name matcher.
- **CPU fp32 rank-2 matmul is the first non-linalg executable value op
  (Sprint 5).** `tessera.matmul` in the **static rank-2, f32** envelope lowers —
  via the value-mode `TilingPass` (which preserves the dense contraction as a
  single `tile.matmul` instead of tiling it into `scf.for`) — to a
  `tessera_apple.cpu.call` carrying the symbol `tessera_apple_cpu_gemm_f32`
  (`status="executable"`, `op_kind="matmul"`). The runtime dispatches Accelerate
  `cblas_sgemm`. Note `tessera.matmul` is the only **registered** Graph IR
  spelling — `gemm` is a vocabulary alias, *not* a distinct registered op, so
  there is no `tessera.gemm` lowering path today (the Tile→Apple value pass
  emits `op_kind="gemm"` only if a `tile.gemm` ever arrives, and the runtime
  reuses the one `tessera_apple_cpu_gemm_f32` symbol for both kinds). **Out of
  envelope is gated, never silently dispatched:** the registered `MatmulOp`
  verifier rejects result-shape mismatches (`(4×8)@(8×16)→(5×5)` fails with a
  named result-dimension diagnostic — it never reaches a value call);
  dynamic/non-rank-2 matmul reaches the value lowering as a raw `tessera.matmul`
  (not the vetted `tile.matmul`) and fails with a named diagnostic; rank-3
  batched matmul is its own op (`tessera.batched_gemm`, Sprint 6 below).
  (f16/bf16 rank-2 matmul became executable in Sprint 7 — see below.)
  - **Runtime dispatch contract:** the value executor requires the **exact**
    operand count for the symbol (a matmul value call takes exactly 2 inputs;
    an extra operand is rejected as `invalid_artifact`, never silently dropped).
    Inputs are **coerced** to contiguous fp32 at the ABI boundary (`_as_f32_2d`)
    — the dtype gate is upstream in the compiler, so a non-f32 array reaching
    the runtime is cast, not rejected.
- **CPU rank-2 f16/bf16 matmul is executable (Sprint 7).** The static rank-2,
  non-transposed matmul value envelope now covers `f16` and `bf16` alongside
  `f32`. `TilingPass.TileMatmulValue` accepts a single shared float element type
  (f32/f16/bf16); `TileToApple` selects the dtype-specific GEMM symbol from the
  result element type: `f32 → tessera_apple_cpu_gemm_f32` (`cblas_sgemm`),
  `f16 → …_gemm_f16` (`bnns_matmul_f16`), `bf16 → …_gemm_bf16`
  (`bnns_matmul_bf16`). The frontend capability gate was widened so f16/bf16
  rank-2 matmul reaches the value lane (per-op matmul dtype tuple on the `cpu`
  and `apple_cpu` capabilities; the rank-2 / no-transpose / no-dynamic envelope
  is still enforced by `TileMatmulValue`). Runtime: `f16` uses `np.float16` +
  uint16 ABI pointers; `bf16` uses `ml_dtypes.bfloat16` and **fails with a named
  unsupported-dependency error if `ml_dtypes` is missing — never a silent fp32
  fallback**. The result keeps its honest dtype (f16→f16, bf16→bf16); tests
  compare through an fp32 view at dtype-appropriate tolerance. Batched f16/bf16,
  transposed, and dynamic matmul remain gated (no batched f16/bf16 yet).
- **CPU fp32 rank-3 batched matmul is executable (Sprint 6).** `tessera.batched_gemm`
  is a registered Graph IR op (`BatchedGemmOp`, rank-3 `B×M×K @ B×K×N → B×M×N`,
  verifier-checked for batch/K/M/N consistency). In the **static rank-3, f32**
  envelope it lowers — via the value-mode `TilingPass` (`TileBatchedMatmulValue`,
  preserving it as a single `tile.batched_gemm`, no `scf.for`) — to a
  `tessera_apple.cpu.call` with symbol `tessera_apple_cpu_gemm_f32_batched`
  (`op_kind="batched_gemm"`, `abi="cblas_sgemm_batched_loop"`). The runtime
  validates exactly 2 rank-3 operands with matching batch + K (no broadcasting),
  coerces to contiguous fp32, and dispatches the Accelerate batched-GEMM C ABI
  (per-batch `cblas_sgemm`, `beta=0`), returning the `(B,M,N)` result.
  **Out-of-envelope is gated, never silently executed:** broadcast batch
  (`1 vs B`) and result-shape mismatch are rejected by the `BatchedGemmOp`
  verifier; rank-4+ is rejected (the rank-3 contract is strict); dynamic shapes
  and non-fp32 reach the value lowering as a raw `tessera.batched_gemm` (not the
  vetted `tile.batched_gemm`) — now *collected* by the Apple pass and failed
  with a named diagnostic rather than leaking through as an unlowered op.
- **The front door is environment-free (Sprint 4).** `apple_target_ir_mode =
  "value"` runs the `-full` pipeline via `driver._resolve_tessera_opt()`, whose
  precedence is `TESSERA_OPT` → `PATH` → the in-repo
  `build/tools/tessera-opt/tessera-opt` (located by a repo-root parent walk that
  finds `python/tessera` + `src/compiler`). A source checkout needs no env
  setup. When the `-full` lowering can't run or fails, the front door keeps the
  artifact IR and records `apple_value_target_ir_error` — the failure is
  observable, never silent.
- **Apple GPU rank-3 batched matmul is executable (Sprint 8).** The
  `(apple_gpu, apple_value_target_ir)` row is now **executable** for one narrow,
  honest lane: static rank-3 `tessera.batched_gemm` (f32/f16/bf16). The GPU
  `-full` pipeline runs value-mode tiling, `TileBatchedMatmulValue` preserves the
  strict static rank-3 envelope as a single `tile.batched_gemm`, and `TileToApple`
  lowers it to a `tessera_apple.gpu.kernel_call` carrying the dtype-specific
  symbol `tessera_apple_gpu_bmm_{f32,f16,bf16}` (MPSGraph bmm; `bf16` keeps an
  honest bf16 ABI — uint16 boundary, internal f32 upcast — *not* an f32 alias).
  The dedicated `apple_gpu_value_target_ir` executor accepts **exactly one**
  `gpu.kernel_call` with `op_kind=="batched_gemm"`, `status=="executable"`, a
  symbol on the GPU allowlist, and exactly two rank-3 operands with matching
  batch + K (no broadcasting); everything else (`cpu.call`, `package_call`,
  multi-op, off-allowlist, extra operands, non-executable status) is
  `invalid_artifact`. Broadcast/rank-4 batched are verifier-rejected;
  dynamic/non-shared-dtype reach the lowering as a raw op and fail with a named
  diagnostic.
- **Apple GPU native sparse attention is executable in one strict envelope
  (Sprint 11).** The Graph-level `tessera.native_sparse_attn_fused` op lowers
  to a `tessera_apple.gpu.kernel_call` with symbol
  `tessera_apple_gpu_native_sparse_attn_f32` only for static rank-4 fp32
  Q/K/V/O tensors with identical shape, rank-4 fp32 gate logits
  `[B,H,S,S/block_size]`, and fixed positive `window_size`, `block_size`, and
  `top_k` attrs satisfying `S % block_size == 0` and
  `top_k <= S/block_size`. The runtime executor validates the same envelope,
  reads the symbol from `metadata["apple_value_calls"]`, and refuses the
  non-Darwin zero-fill stub. Benchmark rows report
  `executor="apple_gpu_value_target_ir"`, correctness, and timing only when
  that runtime probe succeeds; otherwise they report `executor=None`,
  `correctness=None`, and `timing_ms=None` with a named skip reason.
- **Everything else is classified + gated, never silently dispatched:**
  - Other GPU `gpu.kernel_call`s (`cholesky`/`tri_solve` linalg, rank-2 matmul)
    and `gpu.package_call` are **non-executable** on the value lane — classified
    + recorded, `launch` returns a structured non-success. GPU value execution
    beyond batched matmul and native sparse attention waits on a broader GPU
    value-call adapter.
  - **Other non-linalg value calls are not value-executable.** `softmax`,
    `gelu`, `conv2d`, transposed rank-2 matmul, **CPU** batched f16/bf16, and
    dynamic/rank-4 matmul keep their default artifact/runtime path;
    requesting value mode for them never yields an executable `cpu.call` — the
    `-full` pipeline either declines (no value op) or fails with a recorded
    `apple_value_target_ir_error`. They are **not advertised as value-executable**.
    (Sprint 5 promoted fp32 rank-2 `tessera.matmul`; Sprint 6 added CPU fp32
    rank-3 `tessera.batched_gemm`; Sprint 7 added f16/bf16 rank-2
    `tessera.matmul`; Sprint 8 added the Apple GPU rank-3 batched-matmul lane
    (f32/f16/bf16) — each only in its strict static envelope.) Transposed rank-2
    matmul
    (`transposeA`/`transposeB`) stays gated until the value ABI carries transpose
    attrs and the runtime honors them — it is *not* silently computed as the
    non-transposed product.
  - Multi-op programs, multi-symbol CPU value calls beyond the allowlist, GPU
    `kernel_call`, and `package_call` raise `invalid_artifact` (named
    follow-ons), so the runtime reports a clear reason instead of falling back.

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

### Next optimization rungs (foundation noted, not yet built)

The value lane's `TileMatmulValue` / `TileBatchedMatmulValue` are the
*preserve-eligible-contraction* optimization: they keep a statically-
shaped, dtype-eligible matmul as a single `tile.matmul` so it reaches one
GEMM value call instead of dissolving into `scf.for`. Future rungs build on
this seam: (1) **cast folding** — absorb `f32→f16/bf16` casts on matmul
operands into the value call's dtype selection; (2) **matmul+bias/activation
fusion** — recognize `matmul → add(bias) → gelu/relu` chains and emit a
single fused value call; (3) **shape-envelope analysis** — widen the static
envelope (tile-size-aware acceptance, then dynamic-shape value calls once a
runtime shape ABI exists). Each remains gated until implemented.

### Backend inheritance order (normative)

Each new backend adopts the lane in exactly this order — no step skips ahead of
the one before it:

1. **Value Target IR first** — define the value ops (operands/results +
   `{op_kind, symbol, status, …}`) and a value-mode `-full` lowering that
   `replaceOp`s the Tile op and fails loudly on unsupported ops. Pure IR; no
   hardware needed. (Apple: done.)
2. **Executor adapter second** — add a matrix row + an executor that reads the
   value-call tuple and invokes the backend C ABI symbol, with an explicit
   allowlist and `invalid_artifact` for everything outside it. (Apple CPU
   cholesky: done; Apple GPU: pending its adapter.)
3. **Hardware proof third** — numeric conformance on real silicon flips the
   row/symbol from gated to executable. Until then the op stays classified +
   gated, never silently dispatched.

The rule keeps every backend honest: an op is only ever advertised as
executable once a real adapter *and* (where applicable) hardware proof exist.
