---
status: Ratified
classification: Design / Roadmap
authority: Production MLIR/LLVM compiler
last_updated: 2026-06-05
---

# Tessera Production Compiler Plan (MLIR/LLVM)

> **Status:** Ratified architecture (decisions D1–D5 locked).
>
> * **Phase 0 landed 2026-06-05** — boundary proof on `tessera.add`.
> * **Phase 1 Sprint 1.1 landed 2026-06-05** — JIT harness generalized
>   (`tessera_jit_invoke(handle, name, void** descs, int n)` via direct c-iface
>   dispatch; arity 1–8); generic rank-N f32 descriptor packing in Python;
>   `tessera.matmul` → `linalg.fill(0) → linalg.matmul` (first non-elementwise
>   op); binary elementwise family expanded to `add/sub/mul`.
> * **Phase 1 Sprint 1.2 landed 2026-06-05** — `tessera.reduce` (one
>   parameterized op, `kind` ∈ {sum,max,min,mean} × `axis`) →
>   `linalg.fill(identity) → linalg.reduce`, mean = sum + 1/N scale. **First op
>   whose result rank differs from its input rank.**
> * **Phase 1 Sprint 1.3 landed 2026-06-05** — `tessera.softmax` →
>   numerically-stable `max → (x−m) → exp → sum → (e/d)` decomposition. **First
>   broadcast** (reduced `(…)` tensor applied against full `(…,N)` input via an
>   affine map dropping the reduced axis — `emitBroadcastBinary`) and **first use
>   of the `math` dialect** (`math.exp`, via `convert-math-to-llvm`). Elementwise
>   family completed with `tessera.div`.
> * **Phase 1 Sprint 1.4 landed 2026-06-05** — `tessera.rmsnorm` /
>   `tessera.layer_norm` (unweighted, innermost axis). Pure composition over
>   Sprint 1.3 (`emitMean` + `emitBroadcastBinary` + precise `math.sqrt`).
> * **Phase 1 Sprint 1.5 landed 2026-06-05** — **bf16 boundary (ABI §12.5).**
>   `ml_dtypes.bfloat16` Python side / raw-i16 at the memref boundary; matmul
>   accumulates in f32 then `truncf` to bf16 storage. Descriptor packing went
>   dtype-generic (`_resolve_elem`); mixed-dtype rejected, not promoted. A test
>   proves the f32-accumulate policy actually engaged (beats naive bf16-accumulate
>   on K=512).
> * **Phase 1 Sprint 1.6 landed 2026-06-05** — activations
>   `relu/sigmoid/tanh/silu/gelu` (unary `math` family; gelu = tanh approximation
>   to avoid the unlowerable `math.erf`).
> * **Phase 1 Sprint 1.7 landed 2026-06-05** — `tessera.transpose` (rank-2, via
>   `linalg.transpose`) and `tessera.matmul` `transposeA/transposeB` (operand
>   transposed before a plain matmul — the `Q @ Kᵀ` shape). **A full single-head
>   attention block — `softmax(Q Kᵀ / √d) V` — now composes from production-lane
>   primitives and matches the numpy oracle.** **109/109 production-lane tests
>   green** across `tests/unit/test_production_jit_*.py`.
>
> * **Phase 1 Sprint 1.8 landed 2026-06-05** — **multi-op graph compilation.**
>   `GraphFn` (Python) builds a whole multi-op `tessera` function compiled as ONE
>   JIT'd unit — intermediates never cross the boundary, the lowering can fuse.
>   The invocation counter proves it (N-op graph ⇒ +1). **A LLaMA-style
>   single-head transformer decoder layer (rmsnorm + attention + SwiGLU MLP +
>   residuals) compiles as one function and matches numpy** — the "model layer
>   end-to-end" milestone. C ABI invoke went arity-unlimited via libffi.
> * **Phase 1 Sprint 1.9 landed 2026-06-05** — `tessera.batched_gemm` (rank-3,
>   `C[i]=A[i]@B[i]`) → `linalg.batch_matmul`, f32-accumulate for bf16. Unblocks
>   the batch/head dimension; batched per-head attention composes in one graph.
> * **Phase 1 Sprint 1.10 landed 2026-06-05** — **compilation cache** (S14
>   direction). Compiled handles cached on MLIR text; repeated same-(op,shape)
>   calls skip parse→lower→JIT. Transparent (`compile_module` cache-backed,
>   `destroy` no-op for cached, freed at exit). A C++ compile-counter proves cache
>   hits don't recompile while each invoke still runs. **129/129 production-lane
>   tests green.**
>
> **Phase 1 op coverage:** `add/sub/mul/div, matmul (±transpose, ±bf16/f32-acc),
> batched_gemm, reduce(sum/max/min/mean), softmax, rmsnorm, layer_norm,
> relu/sigmoid/tanh/silu/gelu, transpose`, plus **multi-op graph compilation** and
> a **compilation cache** — all oracle-tested through real codegen; f32 + bf16.
> Capstone proof: a transformer decoder layer compiles+runs as one function.
> Remaining Phase-1 polish (deferred, optional): dynamic shapes, reconcile
> `tessera_jit` ↔ `tsrCompileArtifact` (§12.7).
>
> ### Phase 2 — control flow (state is next)
> * **Sprint 2.1 landed 2026-06-05** — data-parallel conditional: `tessera.select`
>   (`cond!=0 ? a : b`) and `tessera.masked_fill` (`mask!=0 ? x : value`). The
>   masked_fill path is the **causal-attention masking primitive**; a causal
>   attention block (`softmax(masked_fill(Q Kᵀ, mask, -1e9)) V`) composes in one
>   compiled function.
> * **Sprint 2.2 landed 2026-06-05** — **`scf.for` control flow.** A bounded loop
>   with a tensor carry, compiled as one function through
>   tessera→linalg→scf→cf→llvm. `GraphFn.for_loop(count, init, body)`; proven with
>   power iteration and **N iterated transformer FFN blocks (shared weights)**.
>   Foundation work: registered scf bufferization, `allowReturnAllocsFromLoops`,
>   and switched the DPS rewrite to **`memref.copy`** (the redirect-and-erase trick
>   silently killed control flow — `memref.copy` is correct for any producer and
>   lowers to `memcpy` for the identity-layout boundary).
> * **Sprint 2.3 landed 2026-06-05** — **`scf.if` conditional control flow.** A
>   shape-(1,) runtime flag drives an `scf.if` (only the taken branch executes,
>   vs select). `GraphFn.cond(flag, then, else)`; **nests with `for_loop`**.
> * **Sprint 2.4 landed 2026-06-05 — state; Phase 2 COMPLETE.**
>   `tessera.write_row` (functional KV-cache update → `tensor.insert_slice`) +
>   **multi-result functions** (DPS out-param per result, so a decode step returns
>   `out` *and* the updated caches as ONE compiled function). **Capstone (the
>   Phase-2 DoD): a stateful incremental-decode loop** threads the KV cache across
>   T steps through the production lane and matches a full causal-attention numpy
>   oracle (and the accumulated cache equals K/V). ABI hardening: function inputs
>   are marked `bufferization.writable = false` so bufferization can never write
>   in-place into a caller's input buffer (write_row stays value-semantic).
>   **154/154 production-lane tests green.**
>
> **Phase 2 COMPLETE** — control flow (select/masked_fill, scf.for, scf.if) +
> state (write_row, multi-result, stateful decode). Control flow lives at the
> builder + bufferization level (no bespoke region-carrying dialect ops).
>
> ### Phase 3 — Apple GPU end-to-end (production on real silicon)
> There is **no upstream MLIR Metal/AIR backend**, so Apple GPU does NOT use the
> CPU lane's `linalg→LLVM→ORC` path — it's a **bespoke Metal back-half** (D2). The
> shared part is the `tessera` graph structure + the CPU lane as oracle; execution
> routes to hand-tuned MPS/MSL kernels.
> * **Sprint 3.1 landed 2026-06-05** — Apple GPU back-half + **cross-target
>   oracle**. `python/tessera/_apple_gpu_backend.py` reuses the existing
>   `tessera_apple_gpu_*` *kernel* C ABI (not `runtime.py`'s dispatch) via the
>   on-the-fly runtime loader. matmul / softmax / **fused matmul→softmax** / gelu
>   run on the real Apple GPU and match the **compiled CPU production lane**
>   (which matches numpy). The fused `matmul→softmax` (one Metal kernel) equals the
>   un-fused CPU composition — the **D2 fused-chain target override**, proven.
>   **12/12 GPU tests green** (`tests/unit/test_production_jit_phase3_apple_gpu.py`).
>
> * **Sprint 3.2 landed 2026-06-06** — **Apple GPU kernel coverage toward the
>   full transformer block.** Wired into the production back-half
>   (`_apple_gpu_backend.py`), each cross-target oracle-matched vs the compiled
>   CPU lane: `gpu_rmsnorm` / `gpu_layer_norm` (unweighted — GPU kernel called
>   with γ=1, β=0 to match the CPU lane's unweighted norms), `gpu_silu`
>   (MPSGraph unary opcode 4), the **fused single-head attention block**
>   `gpu_attention` (`softmax(A@B)@C` in ONE Metal kernel — the D2 fused-chain
>   override, == CPU's un-fused matmul→softmax→matmul), and the fused MLP chains
>   `gpu_matmul_gelu` / `gpu_matmul_rmsnorm`. Capstone: a **pre-norm
>   self-attention sub-block** (rmsnorm → QKV proj → softmax(QKᵀ/√d)V → residual)
>   composes entirely from production GPU kernels and stays oracle-clean against
>   the same composition on the CPU lane. **189/189 production-lane tests green**
>   (+23; `tests/unit/test_production_jit_phase3_kernels.py`).
>
> * **Sprint 3.3 landed 2026-06-06** — **`GraphFn(target="apple_gpu")`
>   graph-level dispatch.** A whole multi-op graph now routes to the Metal
>   back-half as one `run()`: since there is no MLIR Metal backend (D2), the
>   recorded straight-line graph is interpreted op-by-op against the
>   `_apple_gpu_backend` kernels (numpy intermediates threaded), and the
>   canonical chains **auto-fuse** — matmul→softmax→matmul, matmul→softmax,
>   matmul→gelu, matmul→rmsnorm collapse to single fused Metal kernels. Fusion is
>   conservative (only single-use, non-returned intermediates) so it never
>   changes observable values, and `GraphFn.last_dispatch()` exposes which
>   kernels fired (an attention graph fires ONE `matmul_softmax_matmul`).
>   Oracle: the same graph built `target="cpu"` (compiled linalg→LLVM→ORC).
>   Capstone: a **full pre-norm attention block** (rmsnorm → QKV proj →
>   softmax(QKᵀ)V → residual) is expressed as one graph, routed to the GPU, and
>   matches the CPU lane — the attention chain auto-fuses to one kernel while the
>   3 projections + rmsnorm + residual run as their own GPU kernels. Control flow
>   and non-f32 are rejected with clear diagnostics (deferred to later sprints).
>   **199/199 production-lane tests green** (+10;
>   `tests/unit/test_production_jit_phase3_graph.py`).
>
> * **Sprint 3.3 follow-on landed 2026-06-06 — SwiGLU DAG fusion + the full
>   transformer block.** Wired the `swiglu_f32` kernel (`gpu_swiglu`) and taught
>   the GPU graph executor to recognize the **SwiGLU MLP DAG**
>   `(silu(X@Wg) ⊙ (X@Wu)) @ Wd` — five primitive ops (two gate/up matmuls, silu,
>   elementwise mul, down matmul) collapse to ONE fused Metal kernel. `_fuse_for_gpu`
>   became a two-pass matcher (SwiGLU DAG anchored at the gate-multiply, then the
>   linear chains), still conservative (only single-use, non-returned
>   intermediates). **Phase-3 block milestone:** a full pre-norm transformer block
>   — `h = x + attention(rmsnorm(x)); out = h + swiglu(rmsnorm(h))` — is expressed
>   as ONE `GraphFn`, routed end-to-end on the Apple GPU back-half (attention →
>   one `matmul_softmax_matmul`, MLP → one `swiglu`, plus 2 rmsnorm + 2 residual
>   GPU kernels), and matches the same graph on the CPU lane. **206/206
>   production-lane tests green** (+7; `tests/unit/test_production_jit_phase3_swiglu.py`).
>
> * **Sprint 3.3 perf-fusion landed 2026-06-06 — custom `rmsnorm_matmul` Metal
>   kernel (pre-norm + projection).** Authored a NEW C ABI kernel
>   `tessera_apple_gpu_rmsnorm_matmul_f32` in `apple_gpu_runtime.mm` — an
>   in-memory **MPSGraph** that computes `O = (rmsnorm(X)*γ) @ W` as ONE fused
>   dispatch (graph-cached + buffer-pool acquired; CPU reference fallback; stub
>   parity in `apple_gpu_runtime_stub.cpp`; `_SENTINEL_SYMBOL` bumped to it).
>   Wired `gpu_rmsnorm_matmul` (γ=1 → unweighted, matches the CPU lane) and a
>   GraphFn fusion pass that folds a single-use `rmsnorm(x)→matmul` into the
>   kernel (conservative: a norm shared by ≥2 projections, the QKV shape, stays
>   unfused). Numerically exact (max err 4.8e-7 vs numpy). **214/214
>   production-lane tests green** (+8;
>   `tests/unit/test_production_jit_phase3_rmsnorm_matmul.py`). First custom Metal
>   kernel authored end-to-end in the production lane (compiles on-the-fly from
>   `apple_gpu_runtime.mm`, no CMake rebuild needed locally).
>
> * **Sprint 3.3 perf-fusion landed 2026-06-06 — QKV-concat (+ pre-norm fold).**
>   When ≥2 plain matmuls share one input X (the Q/K/V projection shape), the
>   GraphFn GPU executor concatenates their weights `[Wq|Wk|Wv]`, issues ONE
>   matmul, and column-splits the result back — 3 GEMM dispatches → 1, with **no
>   new Metal kernel** (host-side weight concat + existing `gpu_matmul` + split).
>   It's a **multi-output** synthetic node (the executor writes Q,K,V from one
>   dispatch). When X is a single-use pre-norm of exactly that group, the rmsnorm
>   **folds in** (one `gpu_rmsnorm_matmul` on the concat weight) — so a full
>   `rmsnorm → QKV` collapses to a SINGLE `qkv_concat_prenorm` kernel; the fold
>   declines (plain `qkv_concat` + standalone norm) when the norm output escapes.
>   Handles GQA/MQA unequal widths via per-projection column splits. In the full
>   transformer block, the attention pre-norm + 3 projections now collapse to one
>   kernel (dispatch: `qkv_concat_prenorm → matmul_softmax_matmul → add → rmsnorm
>   → swiglu → add`). **220/220 production-lane tests green** (+6;
>   `tests/unit/test_production_jit_phase3_qkv.py`).
>
> * **Sprint 3.3 — whole-graph compile (`GraphFn.run_mlpkg`) landed 2026-06-06.**
>   The architectural leap: instead of `run()`'s per-kernel interpreter, the WHOLE
>   straight-line graph is authored into ONE serialized MPSGraph package and
>   dispatched as a SINGLE Metal ML pass — MPSGraph fuses globally. New C ABI
>   `tessera_apple_gpu_mlpkg_author_graph` (PK8c) in `apple_gpu_runtime.mm` walks
>   a flat op-list (args→placeholders, op j→tensor id n_args+j), builds the
>   MPSGraph (reusing `mpsg_unary_node`/`mpsg_binary_node`; matmul/softmax/norms
>   inline), and hands it to `_mlpkg_compile_and_write`. Python: `apple_mlpkg.
>   author_graph_package` + `GraphFn.run_mlpkg()` (serializes `_ops`, authors to a
>   `*.mtlpackage`, compiles once + caches the pipeline, fills inputs, dispatches,
>   reads output). **The full ~13-op transformer block compiles to ONE MPSGraph
>   dispatch**, matching both the CPU lane and the per-kernel interpreter to 1.1e-6.
>   Op set: matmul(±transpose)/add/sub/mul/div/softmax/rmsnorm/layer_norm/relu/
>   sigmoid/tanh/silu/gelu; single output, straight-line, f32; needs the
>   packaged-ML dispatch lane (macOS 26+). Stub parity + `_SENTINEL_SYMBOL` bumped.
>   **7 tests** (`tests/unit/test_production_jit_phase3_mlpkg.py`).
> * **Sprint 3.3 — Metal-4 resident-weight MLP session landed 2026-06-06.** Wired
>   the existing `mtl4_mlp_session_*` C ABI as `_apple_gpu_backend.Mtl4MlpSession`
>   — `Y = act(X@W+bias)` with `W[K,N]` uploaded once and kept resident; per decode
>   step uploads only `X` (f16/bf16) and dispatches one fused matmul+activation
>   epilogue (act ∈ none/relu/gelu/silu). Amortizes the per-call MTL4 overhead that
>   keeps routing off at small-M decode. `mtl4_mlp_available()` gate; matches an
>   f16-rounded/f32-accumulate oracle. **15 tests**
>   (`tests/unit/test_production_jit_phase3_mtl4_mlp.py`). **242/242 production-lane
>   tests green.**
> * **Sprint 3.4 — bf16 across the GPU back-half + GraphFn landed 2026-06-06.**
>   The back-half kernels are now **dtype-polymorphic** (auto-detect from the input
>   dtype): an `ml_dtypes.bfloat16` input routes to the **native bf16 Metal kernel**
>   (matmul / softmax / gelu / attention / matmul_gelu / matmul_rmsnorm / swiglu,
>   all f32-accumulate per the ABI) or, for ops with no native bf16 variant
>   (rmsnorm / layer_norm / rmsnorm_matmul / silu+unary / elementwise-binary), to an
>   **f32-compute-then-round** path (bf16-storage / f32-compute). `GraphFn(target=
>   "apple_gpu", elem="bf16")` runs the WHOLE graph in bf16 through the interpreter
>   — fusion still fires (`qkv_concat_prenorm`/`matmul_softmax_matmul`/`swiglu`).
>   The full pre-norm transformer block runs end-to-end in bf16 and matches the f32
>   lane to ~0.5% rel (bf16's ~8-bit mantissa). bf16 boundary = raw 16-bit
>   (`ml_dtypes.bfloat16`), matching RUNTIME_ABI_SPEC §12.5 / the CPU lane's Phase
>   1.5 policy. The whole-graph `run_mlpkg()` lane stays f32 (the `author_graph`
>   MPSGraph is f32-hardcoded; bf16 there is a documented follow-on — use `run()`
>   for bf16). **256/256 production-lane tests green** (+14;
>   `tests/unit/test_production_jit_phase3_bf16.py`).
>
> * **Sprint 3.5 — bf16 gap closure (post-Phase-3 follow-on) landed 2026-06-06.**
>   Closed the two bf16 gaps Sprint 3.4 left open. **(#2 bf16 in `run_mlpkg`):**
>   `author_graph` gained an `io_bf16` boundary flag (bf16 placeholders → cast to
>   f32 → f32 body → cast output to bf16), but the mlpkg reflection /
>   `prepare_tensors` path (`MTLTensorDataTypeFromMPSDataType`) hard-asserts on
>   bf16 *bindings* today, so `run_mlpkg(elem="bf16")` authors an **f32 package and
>   converts at the Python boundary** (bf16 in/out, f32 internal compute) — the
>   `io_bf16` C path is kept for when bf16 bindings become reflectable. Full block
>   in bf16 matches the f32 package to ~1.3e-2 and the interpreter to ~2.3e-2.
>   **(#3 native bf16 MPSGraph kernels):** added `tessera_apple_gpu_{mpsgraph_unary,
>   mpsgraph_binary,rmsnorm_gpu,layer_norm}_bf16` (native via the dtype-parameterized
>   `mpsg_run_*` helpers; bf16 supported per `mpsgraph_bf16_supported()` probe;
>   upcast→f32-extern→round host fallback) + stub parity. The back-half's
>   rmsnorm/layer_norm/silu+unary/elementwise bf16 paths now run **native bf16**
>   instead of host-upcast. **256/256 production-lane tests green**; mypy clean
>   both platforms. bf16 is now uniform across the GPU back-half AND both GraphFn
>   engines (run + run_mlpkg).
>
> **Phase 3 COMPLETE** — Apple GPU end-to-end on real silicon: per-kernel
> interpreter (`run()`, any Metal macOS) + whole-graph one-dispatch (`run_mlpkg()`,
> macOS 26+) + all four perf-fusions (rmsnorm_matmul / QKV-concat / mlpkg
> whole-graph / MTL4 MLP session) + f32 **and** bf16 (uniform after Sprint 3.5).
> The full transformer block is production-grade and oracle-matched on this Mac's
> GPU.
>
> ### Phase G — control flow on GPU (bounded loops)
> * **G-A — GraphFn bounded for-loop landed 2026-06-06.** `GraphFn(target=
>   "apple_gpu").for_loop(count, init, body)` now authors the bounded loop as ONE
>   MPSGraph `forLoop` and runs it in a single dispatch (vs the host per-iteration
>   interpreter). New C ABI `tessera_apple_gpu_run_graph_loop_f32` generalizes
>   `cf_scan`'s fixed RNN body to an arbitrary recorded body op-list, reusing a
>   factored `mpsg_build_graph_op` node-builder (shared with the PK8c straight-line
>   author). **Executed DIRECTLY via `runWithMTLCommandQueue`, not the package
>   path** — the MTL4 `MLEncoder` rejects control-flow ops ("Unsupported Ops or
>   shapes for MLEncoder"), the same way it rejects bf16 bindings. v1: f32, single
>   static-shape carry, init must be a function arg, body of args+carry only
>   (matmul/elementwise/softmax/norms/activations). Oracle: the same graph on the
>   CPU `scf.for` lane — `carry = silu(carry@W)` and `carry = carry + rmsnorm(carry)@W`
>   match to ~6e-10. **265/265 production-lane tests green** (+9;
>   `tests/unit/test_production_jit_phase3_loop.py`).
> * **G-A.2 — GraphFn `cond` (if/else) landed 2026-06-06.** `GraphFn(target=
>   "apple_gpu").cond(flag, then_fn, else_fn)` authors the divergent branch as ONE
>   MPSGraph `if` (`ifWithPredicateTensor:thenBlock:elseBlock:`, predicate =
>   flag[0] > 0) and runs it directly (one dispatch; only the taken branch
>   executes). New C ABI `tessera_apple_gpu_run_graph_cond_f32` + a factored
>   `mpsg_build_branch` helper (free function — composes inside the ObjC if-blocks
>   without lambda-capture hazards); the loop body-serialization was unified into a
>   shared `GraphFn._serialize_branch`. v1: f32, flag is a function arg, each branch
>   a straight-line op-list over the args (both same shape). Oracle: CPU `scf.if`
>   lane — `silu(a)` vs `relu(b)`, `a@W` vs `b@W`, branch-returns-arg all match.
>   Mixing for_loop + cond is rejected (v1). **272/272 production-lane tests green**
>   (+7; `tests/unit/test_production_jit_phase3_cond.py`).
> * **G-A.3 — GraphFn bounded `while` landed 2026-06-06.** `GraphFn(target=
>   "apple_gpu").while_loop(max_iters, cond, body, init)` authors a max-iter-capped
>   while as ONE MPSGraph `forLoop` with **select-masking** (MPSGraph's native
>   `while` SIGSEGVs under churn): each step computes `next=body(carry)`,
>   `pred=cond(carry)>0`, `carry=select(pred, next, carry)` — once the predicate
>   goes false the carry freezes. New C ABI `tessera_apple_gpu_run_graph_while_f32`
>   (reuses `mpsg_build_branch` with the carry as `extra`; `selectWithPredicateTensor:`
>   + `greaterThanWithPrimaryTensor:`, both grounded in the SDK header). No CPU
>   `scf.while` lane, so the oracle is a numpy masking-while — decay-stops,
>   runs-all-iters, and stops-immediately cases are **bit-exact**. v1: apple_gpu
>   only, f32, init a function arg. **280/280 production-lane tests green** (+8;
>   `tests/unit/test_production_jit_phase3_while.py`).
>
> **G-A series COMPLETE** — the in-graph control-flow primitives (bounded
> `for_loop`, `cond`, bounded `while`) all lower to single MPSGraph dispatches via
> direct `runWithMTLCommandQueue` (the package/MLEncoder path can't run control
> flow).
> * **G-B — Graph-IR control-flow op + Apple lowering landed 2026-06-06.** The
>   compiler-track lift (C++ MLIR): a new `tessera.control_for` Graph-IR op
>   (`src/compiler/ir/TesseraOps.td`) — a value-semantic bounded loop carried by
>   `iter_args`, with the body as a **symbol-referenced `func.func`** (not a
>   region — sidesteps region ODS/terminator/region-rewrite hazards; `tessera` had
>   no region ops). A new `tessera_apple.gpu.control_loop` Target-IR op
>   (`TesseraAppleOps.td`, value-preserving) records the body symbol + static range
>   + the runtime `symbol = tessera_apple_gpu_run_graph_loop_f32`,
>   `status = "artifact"` (Decision #19 hardware-free Target IR; IR-only). A
>   lowering pass `tessera-control-for-to-apple_gpu` (manual module walk — NOT the
>   greedy driver, whose region-simplification would DCE unrelated result-less
>   artifact ops like an unlowered `tile.cholesky`) wired into the
>   `tessera-lower-to-apple_gpu` pipeline + registered standalone. Builds clean
>   under MLIR 22; lit fixture `tests/tessera-ir/phase8/apple_gpu_control_for.mlir`
>   (phase8 lit 55 PASS / 7 UNSUPPORTED / 0 FAIL); `verifier_coverage` dashboard
>   regenerated. So `@jit`-emitted IR (not just the Python `GraphFn`) can now carry
>   a control-flow op that lowers through the Apple pipeline. MLIR-driven execution
>   off the op (calling run_graph_loop) + `control_if`/`control_while` IR ops are
>   follow-ons.
>
> * **G-B.2 — MLIR-driven execution landed 2026-06-06.** The lowered Target IR
>   op now *executes*, not just lit-checks. `tessera.control_for` /
>   `tessera_apple.gpu.control_loop` gained optional **executable-payload attrs**
>   (the loop body serialized to the run_graph_loop op-list ABI: `body_opcodes`/
>   `body_in0`/`body_in1`/`body_iattr` `DenseI32ArrayAttr` + `body_fattr`
>   `DenseF32ArrayAttr` + `body_out_id`/`carry_arg_index`), copied through by the
>   lowering — so the lowered op is a self-contained executable contract.
>   `GraphFn.run_via_target_ir()` emits the `control_for` MLIR, lowers it through
>   `tessera-opt --tessera-control-for-to-apple_gpu`, and `apple_mlpkg.
>   execute_control_loop_mlir()` reads the lowered op's payload + dispatches off
>   its recorded `symbol` (`tessera_apple_gpu_run_graph_loop_f32`). End-to-end
>   (GraphFn → control_for MLIR → C++ lowering → control_loop → execute) is
>   **bit-identical to `run()`** and matches numpy (~6e-10). **286/286
>   production-lane tests green** (+6; `tests/unit/test_production_jit_phase3_target_ir_exec.py`).
>
> * **G-C — `@jit`-style bounded-loop front-end landed 2026-06-06.** A natural
>   one-call front-end (`tessera._jit_boundary.{build_fori_loop,jit_fori_loop}`)
>   *traces* a bounded loop body into `tessera.control_for` and executes it:
>   `jit_fori_loop(trip, body, init=…, consts=[…], target="apple_gpu")` runs
>   `for _ in range(trip): carry = body(g, carry, *consts)` on Apple GPU through
>   the Target-IR path (control_for → `tessera-opt
>   --tessera-control-for-to-apple_gpu` → `tessera_apple.gpu.control_loop` →
>   `run_graph_loop_f32`), or compiles `scf.for` natively on `cpu`. `body` is
>   `(g, carry, *consts) -> carry` written against the GraphFn build protocol, so
>   the front-end reuses the entire G-A/G-B/G-B.2 machinery end-to-end (no new C
>   ABI / ODS surface). Single tensor carry, f32, v1. `build_fori_loop` returns
>   the un-executed GraphFn for IR inspection. **295/295 production-lane tests
>   green** (+9; `tests/unit/test_production_jit_phase3_fori_frontend.py`).
>
> * **Close-out Phase A — AST `@tessera.jit` → `control_for` bridge landed
>   2026-06-06.** A plain Python `for` loop in `@jit(target="apple_gpu")` now
>   lowers to the GraphFn `control_for` path and executes on Apple GPU — no
>   explicit `jit_fori_loop` call needed:
>   ```python
>   @jit(target="apple_gpu")
>   def f(x, w):
>       for _ in range(N):
>           x = ts.ops.silu(ts.ops.matmul(x, w))
>       return x
>   ```
>   New `python/tessera/compiler/graphfn_bridge.py` does an **IR-to-IR
>   translation**: `detect_loop_fn` reads the `@jit` graph_ir op-list (loop body
>   inline between `tessera.scf.for.{begin,end}` markers), recovers the single
>   tensor carry structurally (the one arg both read in and re-bound by the body),
>   and `build_graphfn` replays the body ops through the GraphFn builder into
>   `for_loop` → `run_via_target_ir`. Wired into `JitFn.__init__` (decoration-time
>   detect, cached per arg-shape) + `__call__` (dispatch before the existing
>   apple_gpu branch). **Dispatch policy (Decision #21):** auto-route any matching
>   loop; if the shape matches but a body op has no GraphFn builder (e.g.
>   `tessera.sqrt`/conv/einsum) raise a stable diagnostic naming the op — never
>   silent host fallback. Reuses the entire G-A/G-B/G-C machinery; no new C ABI /
>   ODS surface. Single tensor carry, loop-is-the-whole-function, f32, v1.
>   **+9 tests** (`tests/unit/test_jit_apple_gpu_loop_bridge.py`: detect, numeric
>   vs numpy, control_loop dispatch + per-shape cache, hard diagnostic,
>   non-matching → existing path); broad jit/compiler sweep (843) + production
>   lane (315) green.
>
> * **Close-out Phase B — bf16 control flow (host upcast) landed 2026-06-06.** A
>   bf16 bounded loop (via `jit_fori_loop` or the AST `@jit` bridge) now runs:
>   `GraphFn._coerce_loop_args` upcasts bf16 args → f32 for the f32
>   `run_graph_loop_f32` executor, the loop computes in f32 (f32 carry — more
>   accurate than per-step bf16 rounding), and `_finalize_loop_out` downcasts the
>   result back to bf16. `_serialize_loop_spec`'s `f32-only` gate relaxed to
>   `{f32, bf16}` (the op-list is dtype-agnostic); `jit_fori_loop` infers `elem`
>   from the carry dtype. No new C symbol (native `run_graph_loop_bf16` is a
>   perf/exact-rounding follow-on). cond/while stay f32-only until C/D. **+4 tests**
>   (`tests/unit/test_production_jit_phase3_control_flow_bf16.py`); the obsolete
>   `test_loop_rejects_bf16` retargeted to assert bf16 now executes.
>
> * **Close-out Phase C1 — `control_if` Target-IR op + lowering + bf16-for-cond
>   landed 2026-06-06.** New `tessera.control_if` Graph-IR op (`TesseraOps.td`) +
>   `tessera_apple.gpu.control_if` Apple Target-IR op (`TesseraAppleOps.td`),
>   carrying the then/else branches serialized to the run_graph_cond op-list ABI
>   (then/else `opcodes`/`in0`/`in1`/`iattr`/`fattr`/`out_id` + `flag_arg_index` +
>   `out_shape`). Lowering `ControlIfToAppleGPU.cpp` (`--tessera-control-if-to-apple_gpu`,
>   manual module walk; symbol `tessera_apple_gpu_run_graph_cond_f32`), wired into
>   the `tessera-lower-to-apple_gpu` pipeline. `GraphFn._serialize_cond_spec` /
>   `_emit_control_if_mlir` / `run_cond_via_target_ir`; `apple_mlpkg.
>   execute_control_if_mlir` reads the lowered op's payload + dispatches. cond also
>   gained bf16 (host upcast, via `_coerce_loop_args`/`_finalize_loop_out`). E2E:
>   both branches select correctly, ~8e-9 vs numpy, drives `control_if`. **+lit
>   fixture** `apple_gpu_control_if.mlir` (phase8 56 PASS / 0 FAIL); **+tests**
>   `test_production_jit_phase3_cond_exec.py`; obsolete `test_cond_rejects_bf16`
>   retargeted. verifier_coverage regenerated for the new op.
>
> * **Close-out Phase C2 — AST `if` bridge landed 2026-06-06.** A plain Python
>   `if/else` in `@jit(target="apple_gpu")` now lowers to `control_if` and executes:
>   ```python
>   @jit(target="apple_gpu")
>   def f(flag, x, w):
>       if flag:
>           y = ts.ops.silu(ts.ops.matmul(x, w))
>       else:
>           y = ts.ops.relu(ts.ops.matmul(x, w))
>       return y
>   ```
>   `graphfn_bridge.detect_cond_fn` reads the `@jit` op-list (then-/else-ops
>   between `tessera.scf.if.{begin,else,end}` markers, dynamic flag = the begin
>   marker's operand arg), recovers the single carried result (the one base
>   re-bound in both branches), and `build_cond_graphfn` replays both branches
>   through `GraphFn.cond` → `run_cond_via_target_ir`. Wired into `JitFn` alongside
>   the loop bridge (`_cond_shape`; dispatch after the loop check). E2E both
>   branches ~8e-9 vs numpy, drive `control_if`. **+tests** in
>   `test_jit_apple_gpu_loop_bridge.py`.
>
> * **Close-out Phase D — `control_while` Target-IR op + lowering landed
>   2026-06-06.** New `tessera.control_while` Graph-IR op + `tessera_apple.gpu.
>   control_while` Apple op, carrying the body+cond op-lists (body/cond
>   `opcodes`/`in0`/`in1`/`iattr`/`fattr`/`out_id` + `carry_arg_index` +
>   `max_iters`; carry is an arg so the shape is recoverable — no `out_shape`).
>   Lowering `ControlWhileToAppleGPU.cpp` (`--tessera-control-while-to-apple_gpu`),
>   reusing the existing `run_graph_while_f32` executor (MPSGraph forLoop +
>   select-masking; native `while` is unstable). `GraphFn._serialize_while_spec` /
>   `_emit_control_while_mlir` / `run_while_via_target_ir`; `apple_mlpkg.
>   execute_control_while_mlir`. while also gained bf16. Also fixed empty
>   `DenseArrayAttr` emission (an empty cond/branch op-list is `array<i32>`, not
>   the invalid `array<i32: >`) via shared `_i32_array`/`_f32_array`/`_i64_array`
>   helpers + parser tolerance. E2E ~1.5e-8 vs numpy, early-stop freezes the carry,
>   drives `control_while`. **+lit** `apple_gpu_control_while.mlir` (phase8 57 PASS
>   / 0 FAIL); **+tests** `test_production_jit_phase3_while_exec.py`; obsolete
>   `test_while_rejects_bf16` retargeted.
>
>   *AST `while` bridge:* deliberately deferred — a Python `while` carries no
>   `max_iters` but `GraphFn.while_loop` is bounded, so the bounded-while *user*
>   surface is the explicit `jit_while_loop` front-end (Phase E), not an AST
>   bridge.
>
> * **Close-out Phase E — `jit_while_loop` front-end landed 2026-06-06.**
>   `jit_while_loop(max_iters, cond, body, init=, consts=)` traces a bounded while
>   into `tessera.control_while` and executes it on Apple GPU (one MPSGraph forLoop
>   + select-masking; the carry freezes once the predicate goes false). `cond`/
>   `body` are `(g, carry, *consts) -> …` on GraphFn handles. `build_while_loop`
>   returns the un-executed GraphFn. This is the bounded-while *user* surface
>   (vs. an AST bridge — a Python `while` has no `max_iters`). **+tests**
>   `test_production_jit_phase3_while_frontend.py`.
>
>   *`jit_scan` deferred (runtime-ABI follow-on, not a front-end wrapper):* a
>   single-dispatch fused scan needs two ops the current GraphFn set lacks —
>   **gather** to index `xs[t]` per step and **scatter**/dynamic-update to write
>   each step's `y` into a `(trip, *y)` buffer — plus a `run_graph_scan_f32`
>   executor that threads `(carry, ys_buffer, index)`. That is a distinct runtime
>   project (new MSL/MPSGraph kernels + a ys-collection C ABI), tracked separately
>   from the control-flow bridge.
>
> **Phase-G control-flow frontier is closed** for `for`/`if`/`while`: AST `@jit`
> bridges (for, if) + bf16 + the three Target-IR ops (`control_for`/`control_if`/
> `control_while`) with MLIR-driven execution + the `jit_fori_loop`/`jit_while_loop`
> front-ends. Open follow-ons: AST `while` ergonomics, `jit_scan` (ys-collection),
> native bf16 control-flow C symbols.
>
> * **Phase F1 — abstract-interp tracing lift (straight-line core) landed
>   2026-06-06.** The general successor to the narrow AST bridge: interpret a
>   function *once by running it* with abstract `Tracer` values that record
>   graph_ir, instead of pattern-matching its AST. New
>   `python/tessera/compiler/trace.py` (`Tracer`, `TraceBuilder`, shape rules,
>   `trace`/`to_graphfn`/`run_traced`) + `compiler/_trace_hook.py` (a neutral
>   `_ACTIVE_TRACER` contextvar). The enabling hook: `autodiff/tape.py::_make_wrapper`
>   — already installed on every `tessera.ops.*` — now checks `active_tracer()`
>   first and, under trace, records an `IROp` + returns a `Tracer` (shape from a
>   rule), skipping numpy. Layer-2 reuses `graphfn_bridge._apply_op`/`_OP_TABLE` to
>   replay the straight-line trace into an executable GraphFn. Proof:
>   `run_traced(mlp_residual)` — a straight-line `silu(matmul)` + residual
>   `add(matmul, x)` + `rmsnorm`, including the cross-reference to the original
>   input the AST bridge can't express as one shape — matches numpy at 1.2e-7.
>   **+12 tests** `tests/unit/test_trace_f1.py`; 813 autodiff/tape/production/vjp/jvp
>   tests green (no eager/tape regression from the shared hook).
>
> * **Phase F2 — control-flow trace-awareness landed 2026-06-06.** Under a trace,
>   `tessera.control.fori_loop` / `cond` / `while_loop` (control.py) detect the
>   active tracer and run their body/branches in a **nested trace**, emitting a
>   `tessera.control_for` / `control_if` / `control_while` IROp that carries the
>   sub-trace (body/branch op-lists + carry/flag SSAs) in its kwargs. `to_graphfn`
>   replays each region through `GraphFn.for_loop`/`cond`/`while_loop`. The carry
>   is captured by natural Python variable flow (the body's returned Tracer) — no
>   SSA/AST heuristics. E2E: traced fori (7e-10), cond both branches (~1e-8), while
>   (0.0) vs numpy. `while_loop` needs an explicit `max_steps` bound under trace.
>   **+10 tests** `tests/unit/test_trace_f2.py`.
>
> * **Phase F3 — general Layer-2 concrete interpreter landed 2026-06-06.** The
>   genuine capability beyond the AST bridge: `trace.execute_traced` walks a traced
>   function with a concrete env, running straight-line ops as per-op Apple GPU
>   kernels (`agb.gpu_*`) and each control region as ONE fused `run_graph_*`
>   dispatch whose "args" are the live concrete values the region's body
>   references. This **lifts the GraphFn-executor constraints** (return ==
>   construct result, loop init == function arg) — so a control construct can sit
>   anywhere with straight-line code **before** (computing its carry/inputs, which
>   need NOT be function args) and **after** (consuming its result), plus residuals
>   to pre-construct values. `run_traced` routes control-flow traces through
>   `execute_traced` and pure-straight-line traces through the fused GraphFn path.
>   Key fix: the carry occupies id `len(args)` (a slot past all args incl. the
>   init), and region OUTPUT ssas (a branch/cond can return a bare external arg)
>   join the live set. E2E: loop+pre+post+residual, cond/while with surrounding
>   code, computed loop consts — all ~1e-7 vs numpy. **+6 tests**
>   `tests/unit/test_trace_f3.py`. (Nested control flow — a control op inside a
>   region body — remains future: `run_graph_*` bodies are flat.)
>
> * **Phase F4 — `@jit(target="apple_gpu")` trace-by-running landed 2026-06-06.**
>   Behind a flag (`TESSERA_JIT_TRACE` env / `trace.jit_trace()` context manager /
>   `set_jit_trace`) while parity is oracled; `JitFn.__call__` routes apple_gpu
>   calls through `trace.run_jit_traced` (trace `self._fn` with the call's args →
>   `execute_traced` for control flow, fused GraphFn for straight-line). Default
>   **off** → zero change to existing @jit behavior. **The tracer's domain (the
>   JAX contract):** straight-line `tessera.ops`; a Python `for _ in range(N)` over
>   a static N (**unrolls**); and explicit `tessera.control.*` for data-dependent
>   control flow. A raw Python `if`/`while` on a traced value now raises via
>   `Tracer.__bool__` ("use tessera.control.cond / while_loop") instead of silently
>   taking one branch — the abstract-trace hazard. Parity: `loop_whole`'s raw
>   `for range(4)` gives the same result through the AST bridge (→ `control_for`)
>   and the tracer (→ unrolled) at 8e-10; `ctrl_cond` selects the right branch;
>   straight-line matches numpy. **+8 tests** `tests/unit/test_trace_f4.py`.
>
>   *Refined "supersede" (vs the original plan):* the tracer supersedes the AST
>   bridge for **explicit-style** code (`ts.ops` + `ts.control.*` + static-`for`).
>   Raw *data-dependent* `if`/`while` is fundamentally incompatible with
>   run-by-tracing, so F5's retirement of the AST bridge is paired with a
>   migration note (rewrite raw data-`if`/`while` to `ts.control.cond`/`while_loop`)
>   rather than a silent swap; raw static-`for` needs no change (it unrolls).
>   Next: F6 (execution-vocab parity — the prerequisite for the F5 flip), then F5.
>
> * **Phase F6 — execution-vocab parity landed 2026-06-06.** F4 revealed the
>   blocker: the tracer's execution vocab is the ~15 `_OP_TABLE` ops, but the
>   apple_gpu `@jit` surface uses `rope`/`flash_attn`/`qkv`/`mla`/`cholesky`/the
>   MPSGraph lane/fused chains — handled by the **canonical compile/runtime path**
>   (straight-line bypasses the bridge), not the tracer. F6 reaches parity without
>   re-implementing every op's GPU execution: **(1) concrete tracing** — the op
>   wrapper passes the original numpy op to `record_op`, which runs it on the
>   inputs' concrete values, so shape/dtype come from real execution for **any**
>   op (no per-op shape rule; `Tracer` carries the `value`; shape rules remain the
>   value-less fallback). **(2) smart dispatch** — `run_jit_traced` traces (full
>   vocab), routes control flow to `execute_traced` and **straight-line to the
>   canonical apple_gpu path** (`_apple_gpu_fast_call`); control-flow-ness is
>   cached (structural). **Validated:** 117 broad apple_gpu @jit tests
>   (mla/mpsgraph-lane/llama-decoder/backend-roadmap) pass with
>   `TESSERA_JIT_TRACE=1` (the F5-default simulation) — straight-line defers
>   bit-identically; flash_attn traces concretely. **+6 tests**
>   `tests/unit/test_trace_f6.py`; default still **off** → zero regression. The 4
>   bridge-internal tests that fail under trace-on are the precise F5 migration
>   list (`divergent_if` raw-`if` → `ts.control.cond`; `last_dispatch==control_loop`
>   / `_bridge_cache` assertions). Next: F5 (flip default + retire
>   `detect_loop_fn`/`detect_cond_fn`/`run_bridged_*` + relax decoration + migrate
>   those 4 tests).
>
> * **Phase F5 — surgical supersede landed 2026-06-07.** The tracer is now the
>   default apple_gpu control-flow front-end and the AST bridge is retired. The key
>   correction over the original "flip a global flag" plan: a global default-on was
>   too blunt (it bypassed the package-lane/auto_batch/exec-mode dispatch that
>   lives *after* the hook, breaking ~15 feature tests). Instead, a **surgical
>   gate**: `trace.function_needs_tracer(graph_ir, fn)` flags control-flow
>   functions at decoration (raw `for`/`if` → `tessera.scf.*` markers, or an
>   explicit `tessera.control.*` call); `JitFn.__call__` routes **only those**
>   (`self._needs_trace`) through the tracer, and **pure straight-line falls
>   through to the existing package/auto_batch/canonical path untouched**. Default
>   on (`TESSERA_JIT_TRACE=0` to disable). **Retired** `detect_loop_fn`/
>   `detect_cond_fn`/`run_bridged_loop`/`run_bridged_cond`/`build_*graphfn`/
>   `LoopShape`/`CondShape` from `graphfn_bridge.py` (kept `_strip`/`_OP_TABLE`/
>   `_apply_op` — the tracer's translation core) + the `JitFn` `_loop_shape`/
>   `_cond_shape`/`_bridge_cache` plumbing. **Migration:** `test_jit_apple_gpu_loop_bridge.py`
>   rewritten to test the tracer (loop functions unroll; `divergent_if` raw-`if` →
>   `ts.control.cond`); bf16-loop + `_needs_trace` assertions retargeted. Validated:
>   the ~15 feature tests now pass (straight-line untouched); 52 migrated
>   bridge/bf16/trace tests green; mypy clean host + linux.
>
>   *F5 follow-on — decoration relaxation landed 2026-06-07.* An AST Graph-IR
>   emission failure no longer hard-fails `@jit(target="apple_gpu")` decoration:
>   the `_decorate` except branch (`jit.py`) catches it for apple_gpu, builds a
>   deferred state (empty module + `compile_bundle=None` + a
>   `JIT_APPLE_GPU_TRACE_DEFERRED` warning), and forces `_needs_trace=True` so the
>   tracer runs the function at call time. So a body the AST can't emit (e.g.
>   surrounding straight-line code + a residual + a shape-conditional around a raw
>   loop) now decorates *and runs* through the `@jit` decorator itself (matched
>   numpy at 0.0). Non-apple_gpu targets still raise. **+3 tests**
>   `tests/unit/test_jit_apple_gpu_decoration_relax.py`; new diagnostic enum member
>   `JitDiagnosticCode.APPLE_GPU_TRACE_DEFERRED`.
>
> **The abstract-interp tracing lift (F1–F6) is complete:** the tracer is the
> default apple_gpu control-flow front-end (concrete tracing → full vocab; control
> flow → fused `run_graph_*`; straight-line → canonical), superseding the AST
> bridge, with `tessera.control.*` as the data-dependent control-flow contract.
>
> * **Phase H1 — nested control flow landed 2026-06-07.** A control op inside a
>   region body used to raise (`run_graph_*` bodies are flat op-lists).
>   `execute_traced` (`compiler/trace.py`) is now recursive: `exec_op` dispatches a
>   flat region to the fused `run_graph_*` path, and a region whose body contains a
>   nested control op is **host-orchestrated** — the outer construct runs as a
>   Python loop threading the concrete carry, recursively calling `exec_op` (so the
>   inner construct still fuses; only the outer per-step dispatch is host-side).
>   Cond on the host evaluates the flag and runs the taken branch; while matches
>   the fused freeze-on-false semantics. `_region_flat` is the classifier. E2E:
>   loop-in-loop / cond-in-loop / loop-in-cond match numpy ≤1e-9. Python-only, no
>   C symbol. **+8 tests** `tests/unit/test_trace_h1_nested.py`; 47 trace + 336
>   production/apple control-flow tests green.
>
> * **Phase H2 — native f16 control flow landed 2026-06-07.** f16 control flow
>   (previously unsupported — only f32 + bf16-via-upcast) now runs natively.
>   Grounded finding: **MPSGraph has no bf16 type** (runtime + SDK headers), so
>   bf16 stays correctly host-upcast; **MPSGraph DOES support f16**, so new
>   `tessera_apple_gpu_run_graph_{loop,cond,while}_f16` C symbols
>   (`apple_gpu_runtime.mm`) run f16 graphs over the f16-bit ABI (uint16_t I/O,
>   2-byte buffers). A shared `run_mpsgraph_cf(dt, esz, …, build_block)` helper owns
>   the placeholder/feed/run/readBytes plumbing (buffer-pool RAII preserved); each
>   entry supplies the loop/cond/while build block — the **f32 functions are left
>   untouched** (zero risk). Python: `apple_mlpkg.run_graph_*_f16` wrappers +
>   non-Darwin stub parity; `_jit_boundary` routes `elem == "f16"` to the f16 lane
>   natively (no upcast) via `_loop_elem_np`/`_coerce_loop_args` + the loop/cond/
>   while dispatch; `jit_fori_loop`/`jit_while_loop` infer `f16` from an
>   `np.float16` carry and use the direct lane (the Target-IR path records the f32
>   symbol). `_SENTINEL_SYMBOL` bumped; `runtime_abi` regenerated. E2E: f16 loop
>   3.4e-6 / cond / while 5.4e-5 vs f32 ref. **+6 tests**
>   `tests/unit/test_apple_gpu_control_flow_f16.py` (incl. ABI probe + a
>   no-native-bf16 regression guard); buffer-pool + bf16/f32 regression green.
>
> * **Phase H3 (core) — fused scan landed 2026-06-07.** `(carry, ys) =
>   scan(body, init, xs)` runs as ONE MPSGraph `forLoop` carrying `[carry, ys]`:
>   per step `x_t = gatherWithUpdatesTensor(xs, [index], axis 0)` (slice),
>   `(carry, y) = body(args, carry, x_t)`, `ys = scatterND(ys, [y], [[index]],
>   Set)`. New `tessera_apple_gpu_run_graph_scan_f32` C symbol
>   (`apple_gpu_runtime.mm`, body-op ids: consts `0..nc-1` / carry `nc` / x_t
>   `nc+1` / op j `nc+2+j`; xs/ys are rank-3 `(trip, *2D-inner)`, consts/carry
>   rank≤2) + stub parity + `apple_mlpkg.run_graph_scan_f32` wrapper. Front-end:
>   `_jit_boundary.{GraphFn.scan, _serialize_scan_spec, _run_apple_gpu_scan,
>   build_scan, jit_scan}` — `jit_scan(trip, body, init=, xs=, consts=) ->
>   (carry, ys)`, `body = (g, carry, x_t, *consts) -> (carry, y)`. E2E: cumsum
>   exact (0.0); an RNN-style matmul-body scan (rank-2 carry, rank-3 xs/ys, const
>   W) at ~2e-8 vs numpy. `_SENTINEL_SYMBOL` bumped; `runtime_abi` regenerated.
>   **+6 tests** `tests/unit/test_apple_gpu_scan.py`; buffer-pool RAII preserved.
>   *Remaining H3b:* `tessera.control.scan` trace-awareness (`record_scan` +
>   `execute_traced` branch) so `@jit(apple_gpu)` scan routes through the tracer
>   like fori/cond/while; `jit_scan` is the working API today.
>
> Next: H3b (scan tracer-awareness). Then Phase 4 (NVIDIA correctness-first).
>
> **Fusion opportunities surveyed (grounded in `apple_gpu_runtime.mm`):** the
> runtime already carries deeper fusion infra — (1) ~~`rmsnorm_matmul`~~ **DONE**;
> ~~QKV-concat~~ **DONE**; ~~`mlpkg_*` whole-graph → one dispatch~~ **DONE**;
> ~~MTL4 MLP session~~ **DONE** (all Sprint 3.3 above). (2) historical `mlpkg_*`
> Metal-4 op-chain authoring API (`author_chain`/`compile`/`dispatch`) which could
> compile an *arbitrary* graph to one dispatch (the "graph as one fused unit"
> ideal, vs. today's per-kernel interpreter); (3) an MTL4 MLP session. A QKV-concat
> fusion (3 projections sharing the pre-norm input → one matmul + split) is also
> open. None are blockers for the block milestone (already met); they are the
> perf-fusion backlog.
>
> Phase 3 remaining toward the DoD: bf16 across the GPU back-half + GraphFn
> (Sprint 3.4). Optional perf-fusion follow-ons above. Control-flow on GPU
> (iterated blocks / decode loop) stays a later sprint — the GPU GraphFn lane is
> straight-line tensor algebra.
> **Scope:** Evolve Tessera from a Python-interpreted prototype into a production
> MLIR/LLVM-IR compiler, while retaining the Python compiler as the
> experimentation lane. This document is the committed decision record; it gates
> all sprint work below.

---

## 0. Two lanes

- **Python lane = experimentation.** Fast prototyping of new ops and
  programming-model ideas. Eager numpy/Accelerate/ctypes execution. The registries
  and the eager interpreter live here. Allowed to be loose — it is a lab.
- **MLIR/LLVM lane = production.** Real codegen, real execution on real silicon.
  This is what ships.

The lanes are connected by **oracle testing** (D4): the Python lane's numpy
reference is the production lane's test oracle. Nothing is promoted to production
without an oracle test that matches within tolerance.

Apple macOS (CPU + Metal 4 GPU) is the **production-grade end-to-end proving
ground** — the silicon on hand, used to expose ABI/dtype/shape/fallback/runtime
mistakes under real GPU conditions before NVIDIA/AMD add their own complexity.

---

## 1. Ratified decisions

### D1 — Keep `tessera` Graph IR as the stable apex; do not TOSA-ize the spine
`tessera` ops are `[Pure]` on `AnyRankedTensor` (`src/compiler/ir/TesseraOps.td`),
i.e. the value-semantic subset that lowers cleanly. The internal spine is
**`linalg` (on tensors) + `math` + `arith` + `tensor` + `scf`/`cf`**. TOSA is an
*ingestion-only* dialect (opinionated quant semantics, rank ceilings, fixed op
menu) — acceptable for *importing* external models, never a lowering target for
Tessera's own ops.

### D2 — Dialect-target map is per op-category, not uniform

| Category | Examples | Target | Notes |
|---|---|---|---|
| Pure tensor algebra | matmul, conv, norms, gelu/silu, reductions, reshape/transpose | `linalg` named + `linalg.generic` + `math`/`arith` | value-semantic; upstream → vector → llvm/gpu |
| Control flow | scan, while, fori, cond, cf_while | `scf` / `cf` | — |
| Stateful / effectful | KV-cache append/read, memory_read/write/evict, RNG state | **stay `tessera` ops w/ `MemoryEffects`**, lower late | → `memref` + `tsr*` runtime calls. **Never become `linalg.generic`.** |
| Scheduling / distribution | mesh, pipeline stages, sharding, collectives | `schedule` dialect (above linalg) | → collective runtime calls |
| Custom attention family | flash_attn, MLA, NSA, lightning, delta | high-level `tessera.attn` op | **structured op + generic fallback + target override** (correctness via tiled linalg/scf; performance via FA-4/MPS/MSL) |

### D3 — The compiled-function ABI is a first-class artifact, designed before nontrivial lowering
- **Calling convention:** MLIR C-ABI wrappers (`-llvm-request-c-wrappers` →
  `_mlir_ciface_<fn>`) taking packed **memref descriptors**
  `{alloc_ptr, aligned_ptr, offset, sizes[], strides[]}`. Descriptor carries
  shape/stride, so dynamic shapes are additive later, not a rewrite.
- **Ownership = Destination-Passing Style (DPS), caller-allocated.** Outputs are
  passed as `outs` memrefs. Aligns with linalg bufferization; avoids callee
  allocation becoming a permanent ABI wart. Callee-allocation exceptions are
  made **explicit later**, never baked into v1.
- **Layout/dtype contract:** boundary memrefs are identity-layout, C-contiguous,
  or the boundary inserts a copy.
- **bf16/dtype policy (ABI rule, not implementation accident):** `ml_dtypes` on
  the Python side, **raw 16-bit at the MLIR/runtime boundary**, copy/convert on
  mismatch.
- **Integration:** a *new* compiled-codegen ABI alongside the existing
  `tsrLaunchHostTileKernel` shim. `canonical_compile(target="cpu")` returns a
  callable bound through `mlir::ExecutionEngine`; the `tsr*` malloc/stream/event
  surface remains the device-memory/async layer underneath.

### D4 — Two lanes connected by oracle testing
Production lane is **green iff its codegen output matches the Python lane within
tolerance.** This makes experimentation *feed* production rather than fork from it.

### D5 — Production apex is a verified *subset* of the Tessera dialect; promotion is explicit
The Python Graph IR may emit a **superset** of what production accepts. The
production-accepted set is partitioned by op category (D2). **Promotion across
the boundary is explicit** and recorded — each promoted op has an oracle test
that admitted it. The coverage registry's job becomes the **promotion ledger**
(is op X in the production subset; what test admitted it), replacing aspirational
status-tracking.

---

## 2. Honest hard-problems register

Upstream MLIR provides correctness primitives and host-side plumbing; it does
**not** provide a competitive kernel. Each hard problem is assigned a phase so it
is confronted, not hidden inside "swap the back-half."

| Hard problem | Reality | Phase |
|---|---|---|
| Bufferization + ownership | Tractable on CPU with DPS; discipline starts at the ABI | 0/1 |
| Memory spaces (shared/global/register) | linalg bufferization is space-agnostic; needs promotion passes | 4 |
| Async copy / `cp.async` / TMA | `nvgpu` ops exist; double-buffer pipelining is hand-assembled | 5 |
| mbarrier / barrier sequencing | `nvgpu.mbarrier` exists; correct sequencing is manual | 5 |
| Target matmul forms (WGMMA/MMA/MFMA) | `nvgpu.warpgroup.mma` / `amdgpu.mfma` exist; shape selection + fragment layouts are real work (current `NVWGMMALoweringPass` stops at `func.call`) | 5 |
| Performance legality (occupancy, bank conflict, swizzle, reg pressure) | Entirely ours | 5 |
| Dynamic shapes | Descriptor carries it; lowering + guards are work | post-1 |
| Apple has no upstream Metal/AIR target | Metal back-half is permanently bespoke | 3 |

---

## 3. Phased roadmap

**Phase 0 — The Boundary.** Op: elementwise `add` (deliberately trivial so the
sprint is ONLY the ABI). DoD: ABI spec written; `tessera.add` → `linalg`/`arith`
→ llvm → ExecutionEngine; `canonical_compile(target="cpu")` returns a callable;
DPS round-trips; oracle test vs Python lane passes. NOT in scope: matmul, tiling,
GPU, dynamic shapes, state.

**Phase 1 — CPU coverage via linalg.** matmul, reductions, norms, elementwise,
softmax → linalg → vectorize → llvm. DoD: ~15 structural patterns covering the
bulk of the ~100 Python ops; all oracle-tested; bf16 boundary works. Performance
reasonable, not tuned. NOT: GPU, attention fusion, state, perf tuning.

**Phase 2 — State & control flow, honestly.** KV-cache/memory/RNG as effectful
ops → `memref` + `tsr*` calls; scan/while/cond → `scf`. DoD: a stateful decode
step runs end-to-end on CPU through the production lane, oracle-matched. NOT:
perf, GPU.

**Phase 3 — Apple GPU end-to-end (production milestone on real silicon).** linalg
front-half + bespoke Metal back-half (MTL4 GEMM / MPS / MPSGraph / MSL) as target
override; attention via D2 override. DoD: a full transformer block runs
production-grade on this Mac's GPU, oracle-matched, hand-tuned Metal kernels as
fast-path. **First point at which "functional AND production-grade end-to-end"
is true.** NOT: NVIDIA/AMD.

**Phase 4 — NVIDIA correctness-first.** linalg → `gpu` → `nvgpu`/`nvvm` → PTX via
the generic pipeline; confront memory spaces + bufferization-to-GPU. DoD: kernels
run correctly on NVIDIA, oracle-matched, even if slow.

**Phase 5 — Performance legality + target matmul forms + AMD.** WGMMA/MFMA,
async-copy/TMA double-buffering, mbarrier, occupancy/swizzle tuning; AMD
ROCDL/MFMA back-half. DoD: competitive MFU on headline kernels.

**Critical path:** D3 (ABI) gates everything → Phase 0 → Phase 1. Phases 3 (Apple)
and 4 (NVIDIA) depend only on Phase 1's front-half, so after CPU they are
sequenced by **priority, not dependency**. Ratified priority: **Apple (3) before
NVIDIA (4)** — Apple is the silicon that can be continuously proven.

---

## 4. Relationship to existing code

- `python/tessera/compiler/graph_ir.py` — Python-lane IR producer (superset, D5).
  Stays. Must emit canonical MLIR syntax so the production lane can parse without
  the current `driver.py` regex patch.
- `python/tessera/runtime.py` — Python-lane executor (numpy/ctypes). Stays as the
  experimentation executor + oracle. Does **not** grow into the production path.
- `canonical_compile.py` — becomes the single front door whose `executable`
  answer is a real JIT'd function once Phase 0 lands.
- C++ `src/transforms/lib/*` — fusion/verifier passes are real and reused; the
  hand-written per-op `*ToAppleGPU.cpp` lowerings become **target overrides**
  (D2), not the foundation.
- The coverage registry (`primitive_coverage.py`) — repurposed as the **promotion
  ledger** (D5).
