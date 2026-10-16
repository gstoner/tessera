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
Owning-device validation executes 184 unique cases: 174 passed in the first
run; its ten explicit-proof-switch skips all pass in the targeted rerun with
TESSERA_NVIDIA_DEVICE_PROOF=1. No skipped case is counted as execution evidence.
This covers typed/F32 host/resident reverse, changed inputs, producer ordering,
retained outputs, native guards and serialized replay.

The required CI unit selector passes 26,037 cases (5,903 skips). The production compiler-route lane passes 1,061 cases with zero skips,
and ci_require_executed passes. All twelve audit lanes, focused Ruff and zero-error
mypy gates pass. AST graph update completes without semantic API calls.
One additional skip versus the prior unit receipt is the ROCm lit suite: its
source-age guard rejects the Super-Bear tessera-rocm-opt after checkout refresh.
Fresh ROCm lit execution is missing here; this private NVIDIA adapter change
does not claim ROCm validation or tool freshness. Other skip reasons include
unavailable sibling GPUs, runtimes and optional shared images.
The initial broad unit invocation was deliberately interrupted and is excluded
from completion evidence. The initial audit pointer failure was corrected;
validation.json records source scope and the documentation-only continuation.
The benchmark and these bounded checks do not close the five-slice program.

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
