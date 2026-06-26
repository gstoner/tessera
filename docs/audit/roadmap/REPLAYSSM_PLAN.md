---
last_updated: 2026-06-15
audit_role: plan
plan_state: landing
---

# ReplaySSM — SSM decode-state ABI plan

> **Status:** Phases 0–6 + 5-bench + block-decode + f16/bf16 + big-GDN +
> size-heuristic prefill default ✅ (2026-06-15) — Track-R complete on Apple.
> Sibling to the Track-L delta-rule work (`python/tessera/stdlib/delta_rule.py`).
> **Remaining — now in active backend development (no longer hardware-gated).**
> Both target boxes are in hand (Strix Halo gfx1151 / RDNA3.5; RTX 5070 Ti
> sm_120 / Blackwell), so the CUDA/ROCm work is a kernel-enablement task, not a
> wait on silicon. Kernels to enable, then `execute-and-compare` against the
> Apple fused-decode reference before closing out each backend:
>   - **`selective_state_update_replayssm_output_only`** — fused decode,
>     buffer-replay path (no state writeback).
>   - **`selective_state_update_replayssm_state_and_output`** — fused decode +
>     flush path (materialize `S_t` → new `S_0`, clear the ring buffer).
> Both must keep `S_0` resident across the ring buffer (that residency is the
> bandwidth win). Lower to ROCm (MFMA/WMMA via the Strix Halo lane) and NVIDIA
> (sm_120 `mma.sync` via the Blackwell lane). This plan stays `plan_state:
> landing` until both backends have a green execute-and-compare row; close it
> out once they do.
>
> **Thesis:** ReplaySSM ([Dao AI Lab, 2026](https://tridao.me/blog/2026/replayssm/);
> [repo](https://github.com/Johnny-Liou/ReplaySSM)) is a *route-selection-over-a-
> state-ABI* problem, not a kernel port. The right Tessera response is a
> first-class **SSM decode-state contract** — checkpoint state + ring-buffered
> replay inputs + cursor + flush/rollback policy + output-only vs
> state-and-output route — that the compiler can lower per-backend, with the
> replay≡eager identity proven by the existing evaluator oracles.

---

## 1. What ReplaySSM is (grounded)

ReplaySSM exposes an algebraic identity on the SSM recurrence as two routes:

- **Summary route (baseline):** `S_t = a_t·S_{t-1} + Δ_t·(v_t k_tᵀ)` — eagerly
  materialize and write the full `[d, n]` state every token.
- **History route (replay):** `S_t = Σ_{i≤t} (Π_{i<j≤t} a_j)·Δ_i·(v_i k_iᵀ)` —
  keep a checkpoint state `S_0` plus a ring buffer of recent `(Δk, v, a)`, and
  reconstruct outputs on demand from the buffer.

From the identity fall three behaviors:

1. **Output-only path** (most decode steps):
   `y_t = ā_t·(S_0 q_t) + Σ_j s_{j,t}·v_j·(k_jᵀ q_t)`. Forming the scalar
   `k_jᵀ q_t` first, then scaling `v_j`, avoids the outer product (~64× fewer
   FLOPs at d=n=128) **and writes no state**.
2. **State-and-output path** (flush): materialize `S_t` from `S_0` + buffer,
   write it back. Triggered only when the buffer is near-full.
3. **Rollback = cursor move.** Speculative draft tokens live in the same ring
   buffer; on rejection, rewind the cursor — no per-position state snapshot.

**Reported results (CUDA):** dominant state traffic `8dn → ~4dn` (halved);
1.43–1.48× kernel / 1.2–1.48× e2e standard decode; 1.87–1.96× spec-decode
throughput (2.14× over baseline spec); 3.0–3.3× concurrency under fixed HBM.
Repo kernel taxonomy: Mamba2 `selective_state_update_replayssm_{output_only,
state_and_output,spec}`; GDN `fused_recurrent_gated_delta_rule_replayssm` /
`gdn_replayssm_spec_decode`; cursor ops `commit_replayssm_spec` /
`reset_replayssm_spec_cursors`; `--replayssm-route` flag.

---

## 2. What Tessera already has (grounded)

The history-route **algebra already lives in the codebase**:

| Building block | Location | Note |
|---|---|---|
| History-route reconstruction (scalar-A, bit-exact ~1e-15) | [`python/tessera/_mamba_ssd.py:47`](../../../python/tessera/_mamba_ssd.py) | `seg = exp(Dcum_t − Dcum_s)·[s≤t]` **is** the `s_{j,t}` decay weights; `M = Cl@Blᵀ` + einsum is the buffered-input reconstruction. ReplaySSM's output-only single step is the `L=1`-query / `h`-key special case. |
| Scalar-A detect + Metal bmm dispatch | [`python/tessera/runtime.py:4604`](../../../python/tessera/runtime.py) | `_apple_gpu_dispatch_selective_ssm` routes scalar-A through the parallel form. |
| Graph-IR op + envelope | [`apple_gpu_envelope.py:191`](../../../python/tessera/compiler/apple_gpu_envelope.py) | `_APPLE_GPU_SSM_OPS = {"tessera.selective_ssm"}`. |
| `selective_ssm` reference + VJP + JVP | [`__init__.py:2953`](../../../python/tessera/__init__.py), `autodiff/{vjp,jvp}.py` | VJP+JVP registered (Phase D3). The "forward-only in v1" docstring is **stale** → fixed in Phase 0. |
| State-handle patterns to clone | [`cache/memory_state.py`](../../../python/tessera/cache/memory_state.py), `cache/handle.py` | `MemoryStateHandle`: `read/write/evict/clone/checkpoint/restore`. `KVCacheHandle`: `append/read/prune/evict_oldest` + `current_seq` cursor. |
| Speculative scaffolding | [`speculative.py`](../../../python/tessera/speculative.py) | `DraftTree`, `batch_verify` (Leviathan), `advance_kv`. **KV-only** — never touches SSM state. |
| True gated delta rule (Track-L) | [`stdlib/delta_rule.py`](../../../python/tessera/stdlib/delta_rule.py) | Recurrent (decode) + chunked UT-transform (prefill) forms; "chunk ≡ recurrent" oracle. The replay form for GDN builds on this. |
| Metamorphic / horizontal-equivalence oracles | [`compiler/evaluator.py`](../../../python/tessera/compiler/evaluator.py) | `metamorphic_equivalence`, `horizontal_equivalence` — the replay≡eager proof harness. |

**Genuinely missing:** an opaque `SSMStateHandle` (state today is a raw
`(B,D,N)` array), ring-buffer wrap + cursor, flush trigger, rollback-by-cursor,
and the output-only vs state-and-output **route selector**.

---

## 3. Honest hardware split (Decision #27)

On Apple, single-token decode is **memory-bound**: the win is the state-traffic
halving, not the 64× FLOP reduction (a compute-bound win). Capturing the
bandwidth win requires a dedicated fused MSL decode kernel that keeps `S_0`
resident and appends small vectors in one dispatch — a *composed* MPSGraph
output-only step will be numerically correct but still round-trips tensors and
will not realize the "never write state to HBM" benefit.

Therefore:
- **Hardware-free, bankable now:** the ABI + route selection + replay≡eager and
  greedy-spec≡greedy-AR correctness proofs (Phases 1–4).
- **On-device, validated here:** the fused MSL decode kernel (Phase 5) is real
  and runs on this Mac's Metal (the runtime compiles on-demand via `clang++`,
  no remote hardware needed) — it keeps `S0` resident and reads only the small
  replay inputs.  What remains genuinely *measurement-gated* is the **throughput
  / state-traffic benchmark** (Phase 5-bench) — proving the halving and the
  speedup numbers, which needs a decode benchmark harness, not new silicon.

Do **not** imply the CUDA throughput numbers transfer for free — they are a
measurement still owed.

---

## 4. Phased plan

### Phase 0 — corrective prework (in progress)
Prerequisites that touch the surfaces Track-R builds on.
- **0a.** Fix the stale `selective_ssm` "forward-only in v1" docstring
  ([`__init__.py:2978`](../../../python/tessera/__init__.py)) — VJP+JVP exist.
- **0b.** Tighten `SelectiveSsmOp::verify`
  ([`TesseraOps.cpp:1835`](../../../src/compiler/ir/TesseraOps.cpp)): it was
  too loose (rank-2 `a` trailing dim never checked vs `N`; `state` only
  rank-checked, not `(B,D,N)`) and too strict (direct ranked-shape compares
  reject dynamic-compatible shapes). Use the file's dynamic-compatible
  dim-equality idiom (`isDynamic(a) || isDynamic(b) || a==b`).
- **0c.** This plan doc; correct the review's stale "delta_rule.py is untracked"
  note — it is tracked in this checkout (current uncommitted surface is
  `TesseraOps.{td,cpp}` + the new `tests/tessera-ir/model_class/selective_ssm.mlir`).

**Acceptance:** `tessera-opt` rebuilds clean; `selective_ssm.mlir` lit fixture
passes (incl. new negative cases for rank-2 `a` N-mismatch and bad `state`
shape); docstring no longer claims forward-only.

### Phase 1 — `SSMStateHandle` + ring buffer (hardware-free) ✅ (2026-06-15)
Opaque handle paralleling `MemoryStateHandle`, shipped at
[`python/tessera/cache/ssm_state.py`](../../../python/tessera/cache/ssm_state.py)
(exported as `tessera.cache.SSMStateHandle`):
- state: `checkpoint_state` `S_0` `(B,D,N)` + ring buffers of *small* per-token
  replay inputs `(delta, x, b)` (2D+N scalars/token, not D·N), a `count` cursor,
  `capacity L`, and `spec_window T`.
- methods: `append` / `read_output` (output-only) / `step` (append+read) /
  `flush` (state-and-output → new `S_0`, clear buffer) / `materialize_state` /
  `rollback(n)` (cursor rewind) / `reset` / `clone` / `checkpoint`/`restore`.
- route policy: `should_flush(n)` = `count + 2T + n > L`; `route_for(n)` returns
  `output_only` | `state_and_output` (the `--replayssm-route` decision as a
  compiler-visible policy).
- reconstruction is the exact history-route identity
  `h_m = exp(Dcum_m)·S0 + Σ_{i≤m} exp(Dcum_m − Dcum_i)·u_i` with bounded
  pairwise decay (stable); output-only never materializes `h_m`.

**Proof (landed):** replay output ≡ eager `selective_ssm`, per token, to
~1e-15 across {no-flush, forced-flush, spec-window, full-`(D,N)`-A, output-gate,
batched}. 20 guards in
[`tests/unit/test_ssm_state_handle.py`](../../../tests/unit/test_ssm_state_handle.py)
(replay≡eager, state≡eager, output-only≡state-and-output, flush exactness,
rollback-as-cursor-move, checkpoint/restore, clone isolation, shape guards).
mypy clean; 221 cache-suite tests green.

### Phase 2 — route selection as a compiler decision ✅ (2026-06-15)
`output_only` vs `state_and_output` chosen by occupancy (`flush when h+2T>L`),
made the **single source of truth** in
[`compiler/ssm_replay.py`](../../../python/tessera/compiler/ssm_replay.py)
(`should_flush`/`select_route`) — `SSMStateHandle.should_flush`/`route_for`
delegate to it, so handle and compiler can't diverge (a randomized 50-state
test locks the equivalence). The module also pins the **kernel taxonomy**
(`REPLAYSSM_KERNELS`) mirroring the reference impl's named kernels (Mamba-2
`selective_state_update_replayssm_{output_only,state_and_output,spec}`; GDN
`fused_recurrent_gated_delta_rule_replayssm` / `gdn_replayssm_spec_decode`) with
**honest statuses** — host `reference` exists today, fused Metal decode kernels
are `planned` (Phase 5). Per Decision #27 the planned kernels are **not** added
to the Apple GPU runtime envelope; a guard asserts they're disjoint from
`_APPLE_GPU_RUNTIME_OPS` (only the prefill scan `tessera.selective_ssm` stays
registered). The envelope SSM comment cross-references the contract. Guards:
[`tests/unit/test_ssm_replay_route.py`](../../../tests/unit/test_ssm_replay_route.py)
(8). This is `--replayssm-route` done as a contract, not a kernel fork.

### Phase 3 — speculative integration ✅ (2026-06-15)
[`speculative.py`](../../../python/tessera/speculative.py) gains `advance_ssm`,
the SSM sibling of `advance_kv`: the caller appended `num_drafts` draft tokens
to the ring buffer; `advance_ssm(handle, num_accepted, num_drafts=…)` rewinds
the cursor by the rejected suffix (`handle.rollback`), with guards (a flush that
dropped drafts is a hard error pointing at `spec_window`).
**Proof (landed):** greedy-spec ≡ greedy-AR — a real argmax-feedback selective-
SSM LM decoded with speculation + rollback produces the *identical* token
sequence to pure greedy AR across {no-flush, small-ring, tight-ring}, with the
test asserting both the accept and reject paths actually fired (otherwise the
rollback is untested). The discipline the proof surfaced — flush only at block
boundaries reserving room for the whole `k`-draft burst, feed drafts with
`auto_flush=False` so they stay rollback-able — is the real ReplaySSM spec
contract. Guards:
[`tests/unit/test_ssm_speculative.py`](../../../tests/unit/test_ssm_speculative.py)
(13). mypy clean; 289-test ssm/speculative/cache sweep green.

### Phase 4 — Mamba2 (scalar-A) end-to-end on Apple GPU ✅ (2026-06-15)
The scalar-A reconstruction factorizes into batched matmuls (projection / gram /
state update), which now route through a `matmul3d` hook (default numpy; the
Apple GPU bmm/MPSGraph lane when wired). `SSMStateHandle` gained `matmul3d` +
`backend`; the bmm reformulation is **exact** vs the dense path (~1e-15, numpy
backend) and matches eager at **f32** through the GPU bmm lane. Factory
[`runtime.apple_gpu_ssm_state_handle`](../../../python/tessera/runtime.py)
(+ `apple_gpu_bmm_callable`) wires the Metal bmm; general `(D,N)` A stays on the
dense reference. Correct-but-composed (contractions on Metal; the single-dispatch
fused kernel is Phase 5). Guards:
[`tests/unit/test_ssm_apple_gpu.py`](../../../tests/unit/test_ssm_apple_gpu.py)
(8).

### Phase 5 — fused MSL decode kernel ✅ (2026-06-15)
A real single-dispatch Metal kernel
`tessera_apple_gpu_ssm_replay_decode_f32` (output-only, scalar-A, f32) in
[`apple_gpu_runtime.mm`](../../../src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm)
(+ host reference in
[`apple_gpu_runtime_stub.cpp`](../../../src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime_stub.cpp)):
one thread per `(b,d)` reconstructs `y_t` from the **resident** checkpoint `S0`
and the small replay ring buffer — never materializing the `(D,N)` state (the
inner-product-first trick). Wired via `SSMStateHandle.decode_fn` + factory
[`runtime.apple_gpu_fused_ssm_state_handle`](../../../python/tessera/runtime.py)
(`backend="apple_gpu_fused"`). The runtime compiles on-demand (`clang++`), so the
kernel **builds and runs on this Mac's Metal**, validated vs eager (f32, ~1.8e-6)
and composing with speculative rollback. Guards:
[`tests/unit/test_ssm_apple_gpu_fused.py`](../../../tests/unit/test_ssm_apple_gpu_fused.py)
(8). Not added to `_APPLE_GPU_RUNTIME_OPS` (it's a handle-side decode kernel, not
a Graph-IR op dispatch).

### Phase 5-bench — decode benchmark ✅ (2026-06-15)
[`benchmarks/apple_gpu/benchmark_ssm_replay.py`](../../../benchmarks/apple_gpu/benchmark_ssm_replay.py)
times `summary` vs `replay_reference` vs `replay_fused` decode (stable JSON
schema) and reports **both** the analytical state-traffic model and measured
per-token latency. **Honest findings on this Mac** (B=1, D=N=128, L=16): the
analytical state-traffic reduction is **~23×** (full-state store 1/L of the time
+ resident `S0`); the numpy output-only path even *beats* the summary baseline
on wall-clock (0.033 vs 0.091 ms/tok — it never materializes the dense state);
the per-token fused Metal kernel is **dispatch-overhead-bound at decode-1**
(≈0.93 ms/tok). **This is now fixed by the `replay_block` mode** (block decode in
one dispatch — see the dispatch-overhead fix below): 49.6 µs/tok, **18.8× faster**
than per-token. The benchmark reports all four modes honestly. Guards:
[`tests/unit/test_ssm_replay_benchmark.py`](../../../tests/unit/test_ssm_replay_benchmark.py)
(4).

### Phase 6 — Gated DeltaNet replay ✅ (2026-06-15)
[`tessera.cache.DeltaNetStateHandle`](../../../python/tessera/cache/delta_state.py)
— the SSM analogue for the **gated delta rule**. The delta-rule transition is
matrix-valued (gated generalized-Householder), so there's no closed-form
decay-sum (the Mamba-2 inner-product trick doesn't apply); reconstruction
**replays the short recurrence** from a checkpoint `S0` over the
`(k, v, β, α)` ring buffer — the win is still real (never *writing* the
`(d_k, d_v)` state to HBM every token; flush only). Same ABI: `step` /
`read_output` / `flush` / `materialize_state` / `rollback` / `clone` /
`checkpoint`-`restore`, route policy via the shared `ssm_replay` contract, and
it composes with `speculative.advance_ssm` (generic on any rollback+count
handle). GDN kernel taxonomy in `ssm_replay.py` updated to reference the handle
(metal kernels still `planned`). **Proof (landed):** replay ≡ eager
`stdlib.delta_rule.gated_delta_rule_recurrent` **bit-exact (0.0 err)** across
{no-flush, flush, spec-window, erase=False linear-attn, output-gate}; flush
exactness; rollback exactness; checkpoint/restore; **greedy-spec ≡ greedy-AR for
a DeltaNet LM** across 3 configs (accept + reject paths both fired). Guards:
[`tests/unit/test_delta_state_handle.py`](../../../tests/unit/test_delta_state_handle.py)
(17).

### Phase 6-gpu — GDN fused Metal block kernel ✅ (2026-06-15)
`tessera_apple_gpu_gated_delta_rule_decode_f32` (in `apple_gpu_runtime.mm` +
stub) replays the gated delta-rule recurrence over a block of `T` tokens **from
a checkpoint `S0`** in one dispatch (one thread per `(b,h)`, full state in
registers), returning per-token outputs **and** the final state. Wired via
`DeltaNetStateHandle.block_fn` + `decode_block` + factory
`runtime.apple_gpu_delta_state_handle`. Validated vs eager
`gated_delta_rule_recurrent` at f32 (~7e-6); numpy fallback exact. Envelope
`d_qk≤16, d_v≤64, d_qk·d_v≤256` (register state). Guards in
[`tests/unit/test_ssm_block_decode.py`](../../../tests/unit/test_ssm_block_decode.py).

### Dispatch-overhead fix — block decode in one dispatch ✅ (2026-06-15)
The per-token fused kernel (Phase 5) pays one command-buffer commit+wait *per
token* → dispatch-bound at decode-1 (≈0.93 ms/tok measured). The fix is **block
decode**: process all known-input tokens (prefill / speculative verification /
benchmark) in a SINGLE dispatch that loops internally. Kernel
`tessera_apple_gpu_ssm_block_decode_f32` (scalar-A, one thread per `(b,d)`,
state row in registers) + `SSMStateHandle.decode_block`. **Measured (B=1,
D=N=128, T=64): 64 per-token dispatches 933 µs/tok → one block dispatch
49.6 µs/tok — 18.8× faster** (benchmark `replay_block` mode). True single-token
AR can't batch (token t+1 needs y_t), but verification/prefill/benchmark can —
which is where the dispatch overhead actually hurt. GDN gets the same via its
block kernel above. Guards: `test_block_decode_beats_per_token` +
[`test_ssm_block_decode.py`](../../../tests/unit/test_ssm_block_decode.py) (11).
Taxonomy (`ssm_replay.py`) records both as status `fused` (shipped on-device).

### f16 / bf16 block kernels ✅ (2026-06-15)
**f16:** native MSL `half` I/O variants `tessera_apple_gpu_ssm_block_decode_f16`
and `tessera_apple_gpu_gated_delta_rule_decode_f16` (f32 accumulation — the
`storage=f16, accum=f32` numeric policy); ~8e-3 / ~1e-2 vs eager; halves on-GPU
I/O. **bf16:** storage via fp32-conversion (MSL has no native bf16, matching the
codebase's bf16 matmul strategy) — round to bf16, run the f32 block kernel,
round the output; ~5e-2 / ~1e-1 vs eager (bf16's 8-bit mantissa); soft-deps on
`ml_dtypes`. Both routed via `compute_dtype="fp16"|"bf16"` on the fused
factories (`apple_gpu_{fused_ssm,delta}_state_handle`).

### Multi-head GDN beyond the register envelope ✅ (2026-06-15)
The register GDN kernel caps `d_qk·d_v≤256` (per-thread `float state[256]`).
`tessera_apple_gpu_gated_delta_rule_decode_big_f32` runs **one threadgroup per
`(b,h)` with the `(d_k,d_v)` state in threadgroup memory** and the threads
cooperating (barriers between the v̂ / state-update / output phases), lifting the
cap to `d_qk·d_v≤8192`. `apple_gpu_delta_block_callable` auto-routes by state
size (register → big). Validated vs eager at `d_qk·d_v=512` (~2e-5, f32).

### Design note — handle-side decode vs the runtime envelope (investigated)
The fused decode kernels are reached through the `SSMStateHandle` /
`DeltaNetStateHandle` ABI, **not** registered in `_APPLE_GPU_RUNTIME_OPS`. This
is correct, not a gap: the envelope is *stateless per-op graph dispatch*, while
ReplaySSM decode is a *stateful iterative loop* (checkpoint + ring buffer +
cursor) — exactly the `KVCacheHandle` precedent (the stateless `selective_ssm`
prefill op IS enveloped + Graph-IR-reachable; the decode-loop state is opaque).
**Measured shape sweep (8 shapes):** the *stateless* block kernel (from-zero = a
full `selective_ssm`) beats the bmm prefill path in **6/8** shapes — up to 5.9×
for batched (B=4) and 1.4–1.9× for large T / large state — and loses only for
tiny shapes (1,64,64,64 at 0.5×; 1,128,128,64 a tie). So the enveloped
`selective_ssm` graph op now **defaults to a size heuristic**: route to the
block kernel when `B*D >= 256` or `T >= 128`, else bmm.
`TESSERA_SSM_BLOCK_PREFILL=1|0` forces block|bmm. The tight-tolerance Apple
GPU `selective_ssm` tests use tiny shapes → stay on bmm (no regression).
Guards: `test_block_prefill_size_heuristic_default`,
`test_block_prefill_optin_matches_default`. This closes the "reachable from
Graph IR" question for the stateless case with a measured default, while keeping
the stateful decode handle-side by design.

---

## 5. Why this fits Tessera

The risk in vLLM — "is replayed output *really* identical to eager state, and
does spec-decode change outputs?" — is exactly what the evaluator program
proves: `replay ≡ summary` is a metamorphic/horizontal-equivalence oracle, and
`greedy-spec ≡ greedy-AR` is the proven DFlash invariant. Track-R turns
ReplaySSM from a serving trick into a contract with a derive-validates-declare
proof.
