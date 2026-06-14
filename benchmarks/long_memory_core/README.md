# long_memory_core

A Tessera-native benchmark core for **long-horizon memory** workloads, modeled on
the patterns that RULER, LongMemEval, MemoryArena, and LongBench-v2 actually
stress. The shared defect those benchmarks expose is not "run a giant attention
op" — it is **re-derivation of resident state**: retrieve a fact, update it,
aggregate across hops, and abstain when the answer is absent, over a bank that
must persist across turns.

It is a sibling of `lattice_reasoning_core` / `grid_ai_core` and follows the same
house template: a primitive-gap registry plus rows emitted at distinct proof
levels through `benchmarks/common`.

## Scenarios

| Row | Benchmark lens | Proof level | What it proves |
|-----|----------------|-------------|----------------|
| `ruler_multi_needle_read` | RULER multi-needle | reference (executable) | exact key→value recall over a distractor bank |
| `ruler_multihop_read` | RULER multi-hop | reference (executable) | pointer-chase: read A → key(B) → payload(B) |
| `longmemeval_abstain_read` | LongMemEval abstention | reference (executable) | present query hits; absent query abstains (NaN) |
| `longmemeval_version_aware_read` | LongMemEval knowledge-update | reference (executable) | `prefer_recent` breaks an exact-key tie toward the newest write |
| `resident_decode_vs_recompute` | MemoryArena resident state | reference (executable) | metamorphic-gated: resident ≡ recompute reads, with `Σt → T` build-traffic reduction |
| `resident_bank_topk_gpu` | MemoryArena resident state | **gap** | no on-device resident bank / index kernel yet |

## Promotion-ladder discipline

Every row reports `runtime_status`, `execution_kind`, and `correctness` through
the shared schema. A capability with no on-device kernel emits a **missing-backend
row** that names the open gap from `MEMORY_PRIMITIVE_GAPS`; `passed` is `None`
(no claim made). When the kernel lands, the row flips green and the name moves to
`LANDED_MEMORY_PRIMITIVES` — exactly how `abstention_read_threshold` and
`metadata_time_version_filter` were closed.

### Open gaps (`MEMORY_PRIMITIVE_GAPS`) — genuine Metal-kernel work

- `resident_state_handle` — bank stays on-device across decode steps
- `kv_cache_append_read` — fused on-device append-then-read

These need real device-residency wiring (the `apple_gpu_batched` session /
`DeviceTensor` API), not reference Python.

### Kernel landed, frontend pending (`PARTIAL_MEMORY_PRIMITIVES`)

- `segmented_topk_gpu` — hard top-k (k>1) runs on Metal via MPSGraph TopK
  (`tessera_apple_gpu_mpsgraph_topk_f32`, values + indices) on the `topk`
  dispatch lane, **hardware-verified** in `tests/unit/test_apple_gpu_topk.py`.
  Remaining: the `@jit` AST frontend does not yet emit the multi-output
  `tessera.top_k` op into Graph IR, so the single-call jit path falls back to
  eager. A kernel landing, not a kernel gap.

### Landed (`LANDED_MEMORY_PRIMITIVES`)

- `abstention_read_threshold` — `memory_read(..., abstain_below=)` returns a
  NaN-filled, `abstained` result when no entry clears the score floor.
- `metadata_time_version_filter` — `memory_read(..., prefer_recent=)` /
  `recency_key=` ranks by `(score desc, recency desc)`, resolving an exact-key
  tie toward the newest write (insertion order, or a named metadata column).

## On-device proof (`tessera.compiler.memory_tasks`)

The on-device memory primitives are graded against the Evaluator oracle on the
Apple-GPU-native path (rung 8, like `attention_tasks`): `memory/score`
(`query·keysᵀ`), `memory/top1` (argmax select), and `memory/soft_read`
(`softmax(QKᵀ/√d)·V`). This is what closed `memory_index_score_gpu`. Guarded by
`tests/unit/test_memory_tasks.py` (Darwin grades every cell).

## Benchmark task adapters (`adapters.py`)

Adapters map each real benchmark's task *structure* onto the memory contract,
with synthetic oracle-checkable instances (no network; real records flow in via
`from_records` / `from_jsonl` — file-format compat only):

| Adapter | Abilities → primitives |
|---------|------------------------|
| `LongMemEvalAdapter` | info-extraction → top-1 recall; multi-session → metadata recall; temporal / knowledge-update → `recency_key` / `prefer_recent`; abstention → `abstain_below` |
| `MemoryArenaAdapter` | interdependent action loop: a fact written in session 1 must be retrieved to choose the correct action in session 2 |
| `LongBenchV2Adapter` | multi-document MCQ: retrieve the relevant doc, decode the planted choice |

Guarded by `tests/unit/test_long_memory_adapters.py`.

## Run

```bash
# summary only
python benchmarks/long_memory_core/benchmark_long_memory.py --smoke

# include the benchmark task adapters
python benchmarks/long_memory_core/benchmark_long_memory.py --smoke --adapters

# include per-row dicts + telemetry, write JSON
python benchmarks/long_memory_core/benchmark_long_memory.py --rows --telemetry --json out.json
```

Guarded by `tests/unit/test_long_memory_core.py`,
`tests/unit/test_memory_tasks.py`, `tests/unit/test_long_memory_adapters.py`,
and the KV-cache contract in `tests/unit/test_kv_cache_contract.py`.
