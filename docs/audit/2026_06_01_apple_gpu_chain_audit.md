# 2026-06-01 — Apple GPU chain audit + proposed next directions

**Author:** Claude
**Context:** Snapshot after Projects 2 + 3 + 5 landed (MPSGraph
precompile/warmup helper, manifest promotion to `hardware_verified`
for 8 encode ops, conv2d encode-session integration). Honest look at
what we just shipped, what new glass-jaws appeared, and what's worth
doing next.

---

## What just shipped (2026-06-01)

### Project 2 — MPSGraph precompile/warmup helper

* `precompile_chain(trace) -> int` in `apple_gpu_chain.py`: runs the
  whole trace with `max_ops_per_cb=1` so each op's MPSGraph compile
  cost amortizes across many tiny command buffers instead of stacking
  up in one large cb (the shape × op-count cliff).
* `@auto_batch` decorator extended with `.warmup(*args, **kwargs)`:
  one explicit precompile call seeds the MPSGraph cache; subsequent
  steady-state calls run at the default chunking budget and hit the
  warm cache.
* 7 tests in `test_apple_gpu_precompile_warmup.py` — exercise cache
  growth, latency speedup (warmup ms > prod-call ms × 1.5), output
  equivalence, nested-warmup no-op, headline "the previously-cliff
  config now runs at default budget after warmup."

### Project 3 — Manifest promotion to `hardware_verified`

* **8 + 1 ops** promoted from `fused` → `hardware_verified`:
  `softmax`, `softmax_safe`, `gelu`, `rope`, `flash_attn`, `rmsnorm`,
  `layer_norm`, `silu`, `bmm` (Project 3), plus `conv2d` (Project 5,
  see below). Each carries `runtime_symbol` pointing at the real
  per-op encode C ABI + `shape_envelope` documenting the validated
  range + `execute_compare_fixture` resolved at construction time.
* `BackendKernelEntry.__post_init__` enforces both `runtime_symbol`
  AND `execute_compare_fixture` for `hardware_verified` (the
  contract).
* `pipeline_gates._eval_link` now accepts `hardware_verified` +
  `packaged` as linkable statuses; the conformance matrix's
  `_proof_status_from_backend_compile` accepts them as complete.
* 39 lock tests in `test_apple_gpu_hardware_verified_promotion.py` —
  every promoted op's runtime_symbol must resolve to a real
  `extern "C"` definition in the `.mm` source, every fixture must
  exist on disk, the count must stay at exactly 9 (no silent
  expansion), the link gate must say PASS.

### Project 5 — Conv2d encode-session integration

* New runtime symbol `tessera_apple_gpu_conv2d_dev_f32_enc` +
  helper `mpsg_encode_conv2d_dev` in `apple_gpu_runtime.mm`. Builds
  an MPSGraph `convolution2DWithSourceTensor:` node, caches it
  (sharing the same key schema as the legacy run path so kernels
  built on one path are reused on the other), and appends via
  `encodeToCommandBuffer:` instead of running its own queue.
* Python wrappers: `conv2d_enc(session, X, W, bias, **kwargs)` for
  the full surface (bias optional) + `conv2d_enc_no_bias` for the
  chain-registry path (the registry tracks DeviceTensor args by
  positional index — optional bias breaks that contract).
* Registered in `ENCODE_OP_REGISTRY` as `("conv2d", "f32")`.
* `agpu.conv2d(...)` exported from `apple_gpu_ops.py` for use in
  `@auto_batch`-decorated functions.
* 12 tests in `test_apple_gpu_conv2d_encode_session.py` — symbol
  resolution, registry surface, numerical equivalence to the legacy
  host path across 5 shape configs (standard, no-pad, 1×1, depthwise,
  strided), single-cb chaining with silu in the middle, `@auto_batch`
  integration, bias-bearing path, groups validation.

---

## What I noticed that's still worth fixing

### Real glass-jaws (numerically known to bite)

1. **`hardware_verified` for conv2d is technically a stretch on
   dtype coverage.** The manifest entry lists `("fp32",)` only, but
   the encode-session ABI is genuinely f32-only today. Honest. The
   `is_hardware_verified` validator doesn't enforce dtype completeness
   — it only requires `runtime_symbol` + `execute_compare_fixture`.
   That means an op can declare `hardware_verified` while only
   covering 1 of 3 dtypes. Audit gap, not a correctness bug.

2. **f16 / bf16 conv2d encode lanes don't exist.** `conv2d` is the
   only encode-eligible op without the full {f32, f16, bf16} dtype
   matrix. The Python registry test
   `test_conv2d_is_not_encode_eligible_for_dtypes_we_havent_shipped`
   pins this honestly; bridging would need `tessera_apple_gpu_conv2d_dev_f16_enc`
   + `_bf16_enc` symbols + matching wrappers.

3. **gelu's `runtime_symbol` for hardware_verified points at the
   unary dispatcher.** It's structurally honest (gelu rides op_code=5
   through `tessera_apple_gpu_unary_dev_f32_enc`), but a downstream
   tool reading the manifest would NOT find a "gelu_dev_f32_enc"
   symbol if it grepped the runtime by op name. Worth documenting in
   `_NUMERICAL_FIXTURES` or the kernel inventory, or considering a
   thin alias symbol for clarity.

4. **MPSGraph cache has no eviction.** The cache_size grows
   monotonically; long-running processes accumulate compiled graphs
   indefinitely. A 5,000-step training run with 10 distinct shapes
   per step is 50,000 cached graph entries. Today nobody runs the
   process long enough to feel this; tomorrow they will.

5. **Conv2d encode-session doesn't share a cache with f16/bf16 legacy
   paths.** The cache key schema includes `(int)ioType`, so f32 and
   f16 graphs are distinct entries. That's correct for graph-building
   but it means the warmup helper's amortization gain doesn't carry
   across dtype.

### Latent gaps (documentation honestly says "not yet")

6. **The chain registry has no `f16` / `bf16` conv2d row** — fine for
   today but worth tracking in the conformance dashboard so the gap
   doesn't get lost.

7. **`precompile_chain` runs OPs one at a time but warmup_one_cb's
   benefit only kicks in if the production path uses chunking ≥ 2.**
   If a caller pins `max_ops_per_cb=1`, warmup is wasted. The
   docstring is honest about this; a future quality-of-life upgrade
   would be `warmup(*, max_ops_per_cb=N)` so callers can warm with
   the exact chunking they'll use in prod.

8. **`AppleKernelBindingSpec` + `apple_binding_spec` field exist on
   `BackendKernelEntry` but are only used by `packaged` rows.** No
   `hardware_verified` row carries a binding spec today. That's fine
   (the contract is different — hw_verified has a real-symbol+test
   contract; packaged has a binding-spec contract). But the field's
   "only-when-packaged" rule lives in `__post_init__` and isn't
   immediately discoverable. A consumer wondering "why is
   apple_binding_spec None on my hardware_verified entry" has to
   read the validator.

9. **The 30-op chunking budget is hardcoded.** It's a defensive
   default driven by empirical cliff observations, but it's not
   exposed as a knob on `@auto_batch` (only on `plan_chain`). A user
   tuning their model for a specific shape × op-count regime would
   benefit from `@auto_batch(max_ops_per_cb=N)`.

10. **The lit fixture `apple_gpu_lowering.mlir` doesn't cover conv2d
    encode.** The G3 work routed the Tile→Apple pass through the
    runtime envelope's `metal_runtime` op set, but conv2d wasn't on
    the encode lane at the time. Now that it is, the lit fixture
    should reflect that — otherwise a regression of the C++ Tile→
    Apple pass could silently downgrade conv2d back to artifact_only
    without any test catching it.

### Dashboard / drift hygiene

11. **`tests/unit/test_apple_gpu_bf16_encode_session.py::test_bf16_registry_covers_full_op_envelope`**
    was a hardcoded 8-op assertion that broke when conv2d landed.
    Fixed it to assert the asymmetric matrix (8 × {f16, bf16} + 9 ×
    {f32}). Watch for more of these — every time we add an op to the
    registry, some hardcoded count somewhere is wrong.

12. **`docs/audit/op_target_conformance.md` regen drift.** The
    `--check` mode catches drift; the regen on the manifest changes
    was a 1-line `complete: 5 → 5` (no count change since conv2d was
    already complete via different gates). But the workflow needs to
    be: any manifest edit → `--render` → commit both. Worth adding
    a pre-commit hook check.

---

## Proposed next directions (ranked)

### Recommended: A. f16 / bf16 conv2d encode lanes + completeness

Smallest hop, biggest dashboard cleanup. Three new C ABI symbols
(`_dev_f16_enc`, `_dev_bf16_enc`), three new Python wrappers, three
new chain-registry entries. Lift conv2d to the full 3-dtype matrix
and clean up the asymmetric 8/8/9 dashboard. Estimated 1 sprint;
unlocks "conv2d is no longer the outlier."

### Recommended: B. MPSGraph cache eviction (LRU)

The cache is monotonically growing. A bounded LRU with
`TESSERA_MPSGRAPH_CACHE_CAPACITY` env knob (matches the pattern from
the recent graph IR cache LRU work, 2026-05-22). Without this, long
training loops will eventually exhaust the cache's memory budget
and we'll learn about it the hard way. Add a `cache_evictions`
counter to the existing introspection symbol so we can see whether
it's actually thrashing.

### Recommended: C. Lit fixture coverage for conv2d encode lane

Add `tessera.encode_conv2d_f32` to the existing
`apple_gpu_lowering.mlir` fixture so a future Tile→Apple pass
regression can't silently downgrade conv2d to `artifact_only`. Pairs
with G3-style drift gate. Small but defensive.

### Defer for now

* **D. `@auto_batch(max_ops_per_cb=N)` knob.** Real but low-priority
  — only matters for users on the cliff edge, which is currently
  nobody. Add when a user complains.
* **E. NVIDIA/ROCm/Metalium execute-and-compare.** Hardware-blocked.
  See `phase_ghi_hardware_frontier.md`. Stays blocked until real
  GPU access lands; no point planning more pre-work.
* **F. `apple_binding_spec` for hardware_verified entries.** Lower
  priority than A and B; the contract is internally consistent
  today.

---

## Pre-existing failures observed during the post-Project-5 sweep

The full `tests/unit/` sweep after this session's Projects 2 + 3 + 5
landed shows **20 failed, 7,583 passed, 10 skipped** (~8 min 36 s).
All 7 distinct failures (the 20 are parametrize expansions) are
pre-existing on a clean tree — verified by stashing my changes and
re-running. **None are from this session's work.** The list:

1. `test_static_analysis_baseline.py::test_mypy_count_is_at_or_below_baseline`
   — 43 mypy errors vs baseline of 0. Mostly `apple_gpu_batched.py`
   `_xxx_enc(...)` calls where mypy proves `None` is not callable
   (the lazy-bind globals start at `None`). Pre-existing.
2. `test_static_analysis_baseline.py::test_ruff_is_clean_against_pyproject_config`
   — 17 ruff errors (F401 unused imports, F541 f-strings without
   placeholders, UP034 extraneous parens). Pre-existing.
3. `test_runtime_abi_audit.py::test_dashboard_matches_live_data` —
   dashboard says 172 symbols, live count is 213 (clean) / 214 (this
   session adds 1). Pre-existing drift; regen with the dashboard
   tool will close it.
4. `test_test_docs_drift.py::test_readme_fast_count_is_current` —
   README claims 5,750 tests, measured is 6,813. Pre-existing.
5. `test_nn_module.py::TestRotaryEmbedding::test_forward_preserves_shape`
   — `IndexError: index 32 is out of bounds for axis 0 with size 32`
   in `apple_gpu_ops_interception.py:190`. Pre-existing.
6. `test_nn_module.py::TestRotaryEmbedding::test_position_offset` —
   same root cause as #5. Pre-existing.
7. `test_reasoning_model_support.py::test_rope_vjp_and_jvp_match_finite_difference`
   — same `rope` interception bug as #5/#6. Pre-existing.

The honest takeaway: this branch ships clean (5/6/7 are a real bug
in `apple_gpu_ops_interception.py::rope` that should get its own
hot-fix sprint; 1/2/3/4 are drift gates a tidy-up sprint should
close batch-style).

## What's good (worth keeping)

* The `hardware_verified` contract is genuinely strong: by
  construction, every promoted entry has a checked-in runtime
  symbol AND a checked-in numerical-comparison test on disk. That's
  a real proof, not a label.
* The conformance matrix + pipeline_gates teach about the new
  status atomically — both surfaces updated together, no split
  brain.
* The conv2d encode-session reuses the existing MPSGraph cache key
  schema so the run path and the encode path share kernels. Smart.
* The promotion-lock test (`test_promoted_count_is_eight`,
  honestly should be renamed to `_is_ten` now) catches a silent
  registry expansion. Pin tests for hand-edited registries are
  worth their weight.
* `precompile_chain` is just a thin call into the existing
  `plan_chain(max_ops_per_cb=1) + execute_chain` machinery — no
  duplicated execution path, no new failure modes.
