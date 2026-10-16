# SM120 reverse adapter artifact reuse

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization: SM120-REVERSE-ARTIFACT-REUSE-20261016.

## Contract

The family adapter retains a private compiled RuntimeArtifact prototype with its
existing memoized canonical hash. Each call receives a deep detached snapshot and
still enters the ordinary runtime dispatcher, checked ABI, native owner and
compiler-generated Graph -> AD -> Schedule -> Tile -> NVIDIA/LLVM/PTX execution.
No roots, gradients or invocation results are cached. Compiler identity, source,
requested derivatives, LRU eviction and fork admission retain their boundaries.
No shared RuntimeArtifact, operation, dtype, pass or ABI is changed.

## Frozen same-image characterization

Measured source: `c0e991af9e62b899158d9da081a079dd6b89d8e3`.
Parent adapter is copied from merged main `27ccc3b547ffdac51537f56930204ec41d8fcc28`.
The packet retains both source snapshots and compiler/provider hashes unchanged.
Owning device: RTX 5070, SM120. FP16, BF16 and matched-value FP32 controls cover
Sq3/Sk5 and Sq7/Sk129, each with host and resident inputs, bias and saved-LSE seed.

Both adapters receive the **same real compiler product**, share the native owner
and must have identical artifact/program identities. Independent FP64 comparisons
run before timing and after every window; retained outputs must remain unchanged.
Compiler subprocesses and concrete frontend execution are forbidden during timing.
Nine ABBA/BAAB rounds, twenty calls per window retain 360 raw observations per
arm/row. Completed public calls and forward/backward device events are separate.

All twelve rows improve completed-call medians: parent/candidate ratios range
1.215-1.817. FP16 short host: 1.4793 -> 0.8143 ms; FP16 longer resident:
3.0330 -> 2.4955 ms. This characterizes warm host-path overhead on these shapes;
it is not a kernel speedup, default schedule promotion or broad model result.
The initial attempted comparison stopped at its identity gate because cold/warm
compiler receipt fields differ; no timings from that attempt are used here.

## Regression and remaining work

Ten cache/ownership regressions plus adjacent contracts passed (24 total).
The warm serialization guard fails against the parent adapter, demonstrating the
removed work. Separate packet checks validate bound sources, all twelve rows,
input hashes, positive separate samples and recomputed medians/ratios.
Owning-device public reverse and broader validation are recorded separately as
they complete; the benchmark alone does not close integration coverage.

Native row-correction producer/consumer reuse, typed capture/dynamic/composed
reverse, half JVP and broader masks/layouts remain open. Apple, ROCm and x86
require their own runtime/physical implementations and exact-device proof.

Reproduce from the repository root with the recorded owning-device environment:

```sh
python -m benchmarks.nvidia.benchmark_reverse_artifact_reuse \
  --control /path/to/decompressed/parent_adapter.py \
  --output /path/to/adapter-ab.json
```

`manifest.json` binds frozen source, evidence and the recorder. Timing ratios are
characterization, not hardcoded performance acceptance thresholds.
