---
last_updated: 2026-07-27
audit_role: reference
---

# Persistent LSE checkpoint contract

FlashAttention streams KV blocks while carrying per-query running maximum `m`,
normalization sum `l`, and output accumulator. The finalized statistic

```text
LSE[q] = m[q] + log(l[q])
```

lets backward reconstruct `P = exp(S - LSE)` without an additional QK
log-sum-exp pass. Saving it costs one `[B,Hq,Sq]` fp32 workspace slice and one
forward store; recomputing it costs an additional backward reduction over K.

## Shared IR contract

Cross-backend synchronization key:
`LSE-CHECKPOINT-CONTRACT-2026-07-27`.

The former `lse.save/load` declarations were destination-less markers. That
contract has been replaced:

- `tessera_attn.lse.save` takes the finalized scalar or rank-1 LSE, an explicit
  rank-1/2 f32 memref destination, and an SSA row offset. It carries
  `MemoryEffects<[MemWrite]>` and returns no fictional value.
- `tessera_attn.lse.load` takes the same checkpoint source and row offset,
  carries `MemoryEffects<[MemRead]>`, and returns a scalar or rank-1 LSE.
- Both operations require a non-empty checkpoint `identity`, explicit
  `memory_space`, and lifetime `scope`. Cross-entry checkpoints currently
  require `memory_space = "global"` and scope `program_launch` or `session`.
  An optional cache policy is `default`, `streaming`, or `cache`.
- The memref SSA value plus identity links forward and backward. Memory space
  remains visible above physical `memref.load/store`, allowing a target to
  choose cache modifiers and vectorized traffic without rebuilding semantics.

Default shared forward lowering no longer emits a destination-less save.
Checkpoint emission is conditional on a training package binding a real
destination.

## gfx1151 implementation

ROCm compiler-owned attention packages support:

- `lse_checkpoint = "recompute"`: forward writes only O; `_pre` recomputes LSE
  with WMMA and computes `D = sum(O*dO)`.
- `lse_checkpoint = "saved"`: forward stores finalized `m + log(l)` into the
  launch-owned `row_lse` workspace; `_pre` becomes D-only.
- `lse_checkpoint = "auto"`: the measured selector. It chooses saved LSE when
  `max(Sq,Sk) >= 128`, otherwise recompute.

Both modes preserve the same five-entry HSACO, deterministic split/reduced
dK/dV ownership, dQ route, bias/softcap/mask semantics, and gradient storage.
The selected mode is part of native-image identity and launch provenance.

## Exact gfx1151 evidence

Environment: HIP 7.14.60850, AMD clang 23, gfx1151 under WSL. Each row uses 21
resident synchronized host-wall samples (`time.perf_counter_ns`,
`hipDeviceSynchronize`) after five warmups. Module, buffers, and workspace stay
resident. These timings are not selector-eligible HIP-event evidence.

| Storage | Sq/Sk | Recompute ms | Saved ms | Saved delta | Decision |
|---|---:|---:|---:|---:|---|
| fp16 | 17/19 | 0.369831 | 0.366916 | -0.79% | near tie |
| fp16 | 64/64 | 0.365743 | 0.400795 | +9.58% | recompute |
| fp16 | 128/128 | 0.441218 | 0.394483 | -10.59% | saved |
| fp16 | 256/256 | 0.511511 | 0.453940 | -11.26% | saved |
| bf16 | 17/19 | 0.367185 | 0.386067 | +5.14% | recompute |
| bf16 | 64/64 | 0.379003 | 0.371612 | -1.95% | near tie |
| bf16 | 128/128 | 0.438503 | 0.377953 | -13.81% | saved |
| bf16 | 256/256 | 0.480716 | 0.449874 | -6.42% | saved |

All saved and recompute rows produce identical measured gradient errors. Worst
errors across the sweep are below `1.01e-4` for dQ/dK and `2.11e-3` for dV.
Saved images are 10,240 bytes smaller because the `_pre` QK/LSE recurrence is
absent. Workspace size is unchanged: `row_lse` already belongs to the canonical
backward workspace.

The short-length results are noisy and non-monotonic under WSL. The automatic
threshold starts at 128 because saved LSE wins for both dtypes at both 128 and
256. Bare-metal and larger-context evidence may move it later.
Explicit modes remain available for reproducible tuning.

Retained summary:
[`benchmarks/baselines/rocm_gfx1151_lse_checkpoint_decision.json`](../../../benchmarks/baselines/rocm_gfx1151_lse_checkpoint_decision.json).
Harness:
[`benchmarks/rocm/benchmark_rocm_lse_checkpoint.py`](../../../benchmarks/rocm/benchmark_rocm_lse_checkpoint.py).

## Sibling-backend assessment

- ROCm gfx1151: implemented and exact-device validated.
- NVIDIA: follow-up required. The shared memory contract transfers; AMD WMMA,
  HSACO packaging, threshold, and WSL timing do not.
- Apple: follow-up required only if Metal training packages choose saved LSE.
  The existing output-only fused ABI must reject a live checkpoint rather than
  erase an effectful save.

No physical schedule or performance decision transfers across architectures.
