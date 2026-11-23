---
audit_role: plan
plan_state: open
owner: Compiler foundation (model programs)
last_updated: 2026-11-22
---

# Compiled autoregressive decode: one program per step, then per loop

Discovery: [compiler map](README.md). Global sequencing defers to
[INTEGRATED_COMPILER_PLAN.md](INTEGRATED_COMPILER_PLAN.md); this plan owns only
the acceptance contract for its slices and proposes no competing priority.

Synchronization key: `AR-DECODE-PROGRAM-1` (proposed; not yet a live-queue
record). Parent items it extends rather than replaces: E2E-REAL-6 (one compiler
authority, resident tensor programs), W1.1 (typed producer composition),
W4-PRODUCT-1 (bounded native loops and state), FRONTEND-IR-MEDIUM-1 (capture).

## 1. Why this plan exists

An evidence survey on 2026-11-22 found that Tessera compiles most of the
operations an autoregressive (AR) language model needs, but not the model:

- The only working prefill → KV-cached decode → greedy loop is the NumPy runtime
  in `python/tessera/models/moe_transformer_runtime.py` (`prefill`,
  `decode_step`, `greedy_generate`; the token loop is a Python `for` with
  `np.argmax`, lines 434–446).
- `python/tessera/compiler/phase_specialization.py` says so itself: "a model
  hand-writes a prefill/decode loop … the phase distinction is not a compiler
  contract." `PhaseSpecializedProgram.generate` drives Python callables.
- No queue item names a transformer block, a decode step or a generate loop as a
  compiled program. MASTER_AUDIT §2 lists general structured programs as open.

The same gap blocks coarse-to-fine generators (block-diffusion refinement in
`models/decode.py` is explicitly "reference orchestration … gated on Phase G")
and the EBM training loop. The one iterative loop that *does* compile end to end
is EBM Langevin sampling ([architecture](../domain/EBM_NATIVE_LOOP_ARCHITECTURE.md)),
which is the pattern this plan generalizes.

## 2. What already exists (2026-11-22, read from generated rows)

Per-op native execution is the floor, not the gap. Rows are from
[`runtime_execution_matrix`](../generated/runtime_execution_matrix.md) and
[`e2e_fleet`](../generated/e2e_fleet.md); evidence never transfers between
targets.

| Need in a decode step | sm_120 (RTX 5070) | gfx1201 / gfx1151 | Notes |
|---|---|---|---|
| RMSNorm, matmul, residual | native; multi-op DAGs via `canonical_native_tensor_program` (retained C++ frame, owned FP16/FP32 intermediates; MLP of two matmuls + ReLU proved) | native; resident RMSNorm→matmul with bounded dynamic M/N/K on gfx1201 | The DAG program is the seed of a "layer program". |
| RoPE | `nvidia_posenc_compiled` (f32) | composed | f16/bf16 RoPE inside a DAG not yet admitted. |
| Paged KV **read** | `nvidia_kv_cache_compiled`; `paged_kv` family `release_ready` | gfx1151 `release_ready` | i32 logical page table, `[start,end)` rows. |
| KV **append** | none | composed from scatter/gather with host glue | `kv_cache_append`: Tile/Target `not_applicable` (`e2e_op_coverage.md:401`). **Gap.** |
| Attention / MLA decode | `nvidia_flash_attn_compiled`, `nvidia_mla_decode_fused_compiled` | composed; gfx1201 attention scheduled packages | Single-query decode shape must be admitted inside the layer program. |
| Embedding gather | none named | none named | `tessera.gather` is cataloged (`op_catalog.py:124`). **Gap.** |
| argmax / top-k | **none** | `rocm_argreduce_compiled`, `rocm_sort_compiled` | x86 has both. **Gap on sm_120.** |
| Bounded loop with device-resident state | `nvidia_control_flow_compiled` — static rank-1 f32 state only | gfx1151/gfx1201 control flow; EBM Langevin row program | State must widen from one f32 vector to a program state (token, position, cache view). |
| Prefill→decode handoff | `CacheHandoff` / `verify_phase_split` exist as Python types and oracle | same | Must become a typed IR boundary. |

### 2a. Optimization readiness per IR level (audited 2026-11-22)

The sm_120 route these programs use runs:
1. `tessera-graph-to-schedule` → `tessera-schedule-to-tile` → `tessera-lower-to-nvidia-sm120` (buffer reuse and arena, Tile→NVIDIA, NVVM).
2. `mlir-opt` loop-invariant code motion and LLVM conversion.
3. `llc -O3`, then `ptxas` (`native_sm120_tensor_program.py:68-70`, `nvidia_native.py:2942-3000`).

The richer `tessera-nvidia-pipeline` (canonicalize, SwiGLU fusion, async copy) is the sm_90 chain and is not on this route. Findings were spot-checked against source:

| Level | Need | State on the sm_120 route | Slice |
|---|---|---|---|
| Graph | Effectful ops inside a native program | **Refused.** `NativeTensorProgram.h:357` admits only `isMemoryEffectFree` ops. | S1/S2 |
| Graph | KV append with position and page table | **Missing.** `tessera.kv_cache.append` takes `(cache, k, v)` only, returns a new handle (`TesseraOps.td:2996`), and has no lowering. | S1 |
| Graph | Order the append before the dependent read | `GraphDataflowAnalysis` (`GraphDataflow.h:57-63`) exists but has no consumer on this route. | S2 |
| Graph | Canonicalize/CSE/DCE before Schedule | Not in the route; runs only in projection checks. | S3 |
| Graph | SwiGLU and epilogues inside a program | Matmul epilogues are standalone-only (`NativeTensorProgram.h:343`); `SwigluFusionPass` is not on the route. | S2 |
| Graph | Effect pass naming | `pass_metadata.py:725` and `pipeline_registry.py:138` name `tessera-effect-annotate`; the C++ pass registers `tessera-effect-annotation`. **Defect.** | S0 follow-up |
| Schedule | M=1 GEMV | **None.** f16/bf16 always uses m16n8k16 MMA; at M<32, one warp with global staging wastes 15/16 of MMA rows. No sm_120 split-K. | perf, after S4 |
| Tile | In-place KV update without copy (DPS / one-shot bufferization) | `tessera-opt` does not register the vector bufferization model (`tessera-opt.cpp:538-543`); DPS exists only in side lanes. | S1 |
| Tile | Warp-shuffle reductions (norms, argmax) | Shared-memory tree with a barrier per step; no `shfl`. | S1 (argmax), perf |
| Tile / Target | cp.async, 16-byte vector loads, shared staging | Present, but enabled only for macro producers (M≥32). `ldmatrix`: not found in `src`. | perf |
| Target | IEEE sqrt in RMSNorm | `__nv_fsqrt_rn` is pinned only in the row-program emitter; the native norm uses `math.sqrt` → libdevice, which may be approximate. | S2 gate |
| Target | argmax / gather / append / RoPE inside a program | No sm_120 lowering; paged read needs an isolated two-argument entry (`NativePagedKV.h:15-24`). | S1 |
| LLVM | Device middle-end | `llc -O3` with no `opt` run on device IR. | perf |
| Runtime | Launch overhead | The frame issues one `cuLaunchKernel` per stage, about 10 per layer; CUDA graph capture exists only for profiling. HIP graphs exist. | S4 |
| Runtime | Resident buffers and arena | Present (`NativeTensorFrameRuntime.h`). | — |

**Consequence for the slices:** S1 is larger than "three ops". It must first admit a resource-scoped memory effect into the native program, and it must redesign the append op to carry position and page-table operands with an in-place, destination-passing lowering. The other two ops (argmax with warp reduction, embedding gather) follow. Performance gaps (GEMV schedule, split-K, `ldmatrix`, device `opt`, graph capture) are not correctness blockers and wait for S4.

## 3. Architecture

### 3.1 What "one program" means on a GPU

A decode step is too large for one kernel, and nothing here requires that. The
unit of compilation is a **compiler-owned program**: one verified MLIR module
whose Graph→Schedule→Tile→Target descent produces a set of kernels plus a
**native frame** that owns buffers, launch order and completion (the
`canonical_native_tensor_program` retained-frame ABI, `docs/spec/NATIVE_TENSOR_JIT.md`).
The acceptance property is not "one launch"; it is:

1. **No semantic decision on the host between kernels or between steps.** The
   host may submit work and observe completion; it never computes logits,
   picks a token, edits the cache, or advances position.
2. **No per-step host transfer of model state.** Weights, KV pages, the current
   token and position stay device-resident; the only host traffic per step is
   the emitted token (optional, for streaming) and completion status.
3. **One lowering authority** (Decision #31). The NumPy runtime becomes a
   declared differential oracle (#31a), never a second production path.

### 3.2 Program shape

```
decode_step(state: DecodeState, weights) -> DecodeState
  DecodeState = { token: i32[B], pos: i32[B], kv: PagedKV (resident, effectful),
                  rng: Philox key (sampling only), done: i1[B] }
  x  = gather(embed, token)                         # new: native gather
  for l in layers (static unroll):
     h  = rmsnorm(x); q,k,v = matmul(h, Wqkv_l)
     q,k = rope(q, k, pos)
     kv  = kv_cache.append(kv, l, k, v, pos)         # new: native in-place append
     a   = attention_decode(q, kv.read(l, 0, pos+1)) # existing paged read + attention
     x   = x + matmul(a, Wo_l)
     x   = x + mlp_l(rmsnorm(x))                     # existing DAG program pattern
  logits = matmul(rmsnorm(x), W_lm)
  token' = argmax(logits) | sample(logits, rng)      # new on sm_120: native arg-reduce
  return {token', pos+1, kv, rng', done | token'==eos}
```

`generate` is then a bounded `while` over `decode_step` with `DecodeState` as
the loop-carried value and `max_new_tokens` as the bound — the same structure
as the EBM K-step loop, with a structured state instead of one f32 tensor.

### 3.3 Design decisions that must hold

- **The KV cache is an effectful, resident operand, not a value.** Append is a
  `memory` effect (Decision #5) on a buffer the frame owns; Graph IR must not
  model it as copy-and-return, or the loop carries the whole cache by value.
  Use destination-passing style through bufferization so the cache is updated
  in place, and let `GraphDataflowAnalysis` (Decision #30) order append before
  the dependent read.
- **Static capacity, runtime length.** Shapes are bucketed by
  `max_seq_len`/page capacity (Decision #28's `static | bucket | dynamic`
  policy); the live length is a runtime scalar in `DecodeState`. This is the
  bounded-dynamic pattern already proved for resident M/N/K.
- **Sampling randomness follows Decision #18.** Philox stream assignment and
  counter advance mirror the EBM lowering; greedy decoding needs no RNG and is
  the first acceptance target.
- **Semantic keys fail closed** (Decision #21a): dtype, `numeric_policy`
  (accumulator), RoPE base/scaling and EOS handling are required attributes,
  never defaults.
- **Prefill is a second program over the same cache layout**, joined by a typed
  `CacheHandoff` boundary in IR; `verify_phase_split` (prefill ▸ decode loop ≡
  full forward) becomes its device gate.

## 4. Slices and acceptance

Each slice lands with its producer, its consumer and its test together
(Decision #29). The first device is **sm_120 on The-Super-Bear**, because it
already has the retained multi-op frame, paged KV read and fused MLA decode;
every other target needs its own proof.

| Slice | Deliverable | Start | Acceptance (all fail closed outside the envelope) |
|---|---|---|---|
| **S0** (landed, §4a) | Oracle and fixture. Freeze a tiny dense config (2 layers, d=64, GQA 4/2 heads, vocab 256, page 16) and a seeded weight set; declare `moe_transformer_runtime` greedy decode the #31(a) oracle; write the Graph-level `decode_step` contract and envelope. | host-free | Oracle reproduces cached ≡ recompute on the fixture; contract doc names every semantic attribute; negative fixtures for missing dtype/policy/EOS. |
| **S1** (S1a landed, §4b) | Three missing ops on sm_120 through Graph→Schedule→Tile→Target: row **argmax** (first-occurrence tie-break, i32 index), **embedding gather**, **kv_cache.append** into paged storage (in-place, DPS). | device (sm_120) | Bitwise equality with the oracle per op; append+read round-trip across a page boundary; lit fixtures for each lowering; execution-matrix rows. |
| **S2** | **One decoder layer** at M=1 as a native tensor program: RMSNorm, QKV matmul, RoPE, append, paged attention read, O-proj, residual, MLP — one Graph function, one retained frame. | device (sm_120) | Per-layer output and updated KV pages match the oracle (f32 bitwise or declared f16/bf16 tolerance with fp32 accumulation); zero host contact inside the layer; replay byte-identity of the package. |
| **S3** | **Whole decode step**: L layers + final norm + LM head + argmax; `DecodeState` resident across calls; host loop calls the step program. | device (sm_120) | Token sequence identical to the oracle for 64 steps; per-step logits parity (not only argmax, so ties cannot hide errors); host contact per step = one submit + completion (+ optional token readback), counted by the frame. |
| **S4** | **Device-resident loop**: `generate` as a compiled bounded `while` over `decode_step` with EOS early exit. Widen the control-flow lane from rank-1 f32 state to structured state; an explicit, IR-visible graph-capture ABI is an acceptable physical route if it is declared, not hidden. | device (sm_120) | One program invocation produces N tokens identical to S3; no host round-trip between steps; EOS stops the loop on device. |
| **S5** | **Prefill program** sharing cache layout; typed `CacheHandoff` in IR. | device (sm_120) | `verify_phase_split` passes on device: prefill ▸ decode loop ≡ full forward for prompts that cross page boundaries. |
| **S6** | **Sampling**: temperature / top-k with Philox; deterministic per Decision #18. | device (sm_120) | Fixed-key token streams bit-identical to the oracle; distribution test on a large draw. |
| **S7** | **Sibling targets**, each with its own owning-device proof: gfx1201 (Tajasarus), gfx1151 (Princess-Luna), x86 AVX-512, Apple GPU. ROCm already has native arg-reduce and resident dynamic M/N/K. | device (each) | The S3/S4 acceptance re-run on that target; no transferred claims. |

### 4a. S0 — landed 2026-11-22 (host-free, Mac)

- **Oracle:** `python/tessera/models/ar_decode.py`, the declared #31(a)
  oracle. `reference_decode` (KV-cached) and `recompute_decode` (full forward
  on the growing prefix) run over the fp64 reference runtime and return chosen
  tokens, the logits that chose them and top-1/top-2 margins.
  `paged_kv_pages` defines the device KV layout: `[pages, page_size, Hkv, D]`,
  token `t` at page `t // page_size`, slot `t % page_size`, keys stored after
  RoPE, zero past the live length.
- **Fixture:** `AR_DECODE_TINY` (2 layers, H=64, 4/2 GQA heads, D=16,
  vocab 256, dense SwiGLU 128) with `TINY_PROMPT`. Weights come from a
  counter-based splitmix64 hash rather than `numpy.random`, so every host and
  every device upload reproduces the same bytes; `TINY_WEIGHTS_SHA256` pins
  them. Norm scales are not all-ones, so a dropped scale cannot pass.
- **Contract:** `DecodeStepContract` has no defaults. Its semantic keys are
  storage dtype, accumulator dtype, RoPE variant and theta, RMSNorm eps, EOS
  (explicit `None` = never stop), sampling, tie-break, page size and capacity.
  `validate_decode_contract` raises the registered code
  `AR_DECODE_CONTRACT_INVALID`, naming the field. It also refuses a model
  outside the S0 envelope (MLA, MoE layers, sparse or windowed attention) and
  a contract whose theta or eps disagrees with the model.
- **Acceptance met** (`tests/unit/test_ar_decode_contract.py`):
  - Cached decode equals recompute over 32 steps: identical tokens, per-step
    logits within 1e-10, 21 distinct tokens.
  - Every fp32 step is decisive: minimum margin 0.043.
  - EOS stops both paths at the same token.
  - The paged layout holds independently recomputed RoPE'd keys.
  - Nineteen invalid contracts, three out-of-envelope models and four invalid
    requests each fail closed.
- **Defect found and fixed:** the reference runtime ignored
  `config.rope_theta`, so every RoPE call used base 10000, including GLM-5
  (8e6) and MiniMax-M3 (5e6). It now passes the model's theta at all eight GQA
  and materialized call sites. A regression test proves theta reaches the
  logits.
- **Found, not fixed (follow-up, outside the S0 envelope):** the MLA absorb
  path in `stdlib/attention.py` never rotates the cached key. Its
  `compress_latent` docstring says `k_rope = RoPE(x·W_kr)`, but the function
  returns `x·W_kr` unrotated. MLA also has no RoPE-base parameter. Cached ≡
  recompute cannot see this, because both sides are equally position-blind on
  keys. It needs its own fix before any MLA slice uses this oracle.

### 4b. S1a — in-place paged KV write on sm_120 (2026-11-22)

S1 is split into sub-slices because §2a showed the append was the hard part.
S1a lands the physical write; S1b (argmax), S1c (embedding gather) and S1d
(admitting a resource-scoped memory effect into the native tensor program)
follow.

- **Op design.** `tessera.paged_kv_write(pages, page_table, values, position)
  -> pages` is a raw-tensor op, the write-side twin of `tessera.paged_kv_read`.
  - The result type must equal the pages type. Native lowering binds it to the
    input pages buffer, so the write is destination-passing and copies no page.
  - `position` is an `i32[1]` tensor read on the device, not a static
    attribute, so a decode loop can advance it without a host round trip.
  - The handle-level `tessera.kv_cache.append` is unchanged.
- **Route.** `pagedKVWriteContract` (in `NativePagedKV.h`) hashes the contract.
  The route then runs Graph → `schedule.paged_kv_write` → `tile.paged_kv_write_kernel` → `materializeSm120PagedKVWriteKernel`.
  The Python side (`scheduled_paged_kv.lower_scheduled_paged_kv_write`,
  `nvidia_native.package_paged_kv_write`) only binds and replays; every launch
  field is re-read from the replayed Tile contract. Only sm_120 is admitted;
  any other target refuses at Schedule by name.
- **Fail closed on device data.** The kernel checks the signed position, the
  logical capacity and each physical page index. A violation skips that
  element and sets an `i32` status word. The launcher then returns
  `paged KV write refused`, does not copy the pages back, and leaves the
  caller's cache untouched.
- **Evidence** (The-Super-Bear, RTX 5070, NDEBUG LLVM 23.1.1, worktree at the branch head):
  - Lit: three `tests/tessera-ir/ar_decode` fixtures (schedule/tile, five verifier refusals, ROCm refusal) and `sm120_paged_kv_write_kernel.mlir`.
  - `tests/unit/test_paged_kv_write_contract.py`: 9 passed (replay, tamper and relabel refusal, envelope refusals).
  - `tests/device/nvidia/test_paged_kv_write.py`: 7 passed.
    - 24 token-by-token writes through a permuted page table, bitwise against `paged_kv_pages` of the S0 oracle.
    - One 24-token prefill write across a page boundary.
    - An in-place write touches only its rows.
    - Five refused writes (negative position, past capacity, physical page out of range, negative page, two written logical pages aliasing one physical page) each leave the pages bitwise unchanged with status 1; an alias outside the written range is not refused (review on #1073).
- **Host-copy ABI, not yet resident.** This launcher still copies pages host↔device per call, like the read. Resident pages, a device-side position across calls, and composition with the paged read inside one program are S1d/S2 work.

Performance is out of scope until S4 passes. Timing rows follow the
device-clock admission rules and must carry their route (Decision #12); no
promotion is claimed by correctness slices.

## 5. What this unlocks next (not in scope here)

- **MoE and MLA models**: swap the dense MLP for the existing MoE transport and
  the attention for fused MLA decode inside the S2 layer program.
- **Speculative / block-diffusion decoding** (DFlash, DiffusionGemma): the
  refinement loop is a bounded `while` over a fixed canvas with the same
  resident-cache machinery; it needs S4 plus a native canvas-commit op.
- **EBM training loops** (persistent chains, contrastive divergence): the same
  structured loop state, with the chain as resident state instead of KV.

## 6. Risks named now

- **Cache-by-value in Graph IR.** If append is modelled functionally, the loop
  carries the cache and every step copies it. Mitigation: effectful operand +
  DPS from S1, with a negative test that fails if bufferization introduces a
  cache-sized copy.
- **Scope creep into a model framework.** The slices stop at a tiny dense model;
  real checkpoints, tokenizers and MoE are follow-ons.
- **Tie-breaking and reduced precision.** Greedy equality over many steps is
  brittle under f16/bf16; per-step logits parity with declared tolerance is the
  primary gate, token equality the secondary one.
- **A hidden second compiler.** A Python orchestrator that sequences kernels
  "for now" is exactly the E2E-REAL-6 failure mode; the frame, not Python, owns
  launch order from S2 onward.
- **Host-evidence traps.** Device suites on a host without the device skip
  rather than fail; source `scripts/_nvidia_env.sh` and the sm_120 device-lane
  variables before any run, and report the host that produced each result.
