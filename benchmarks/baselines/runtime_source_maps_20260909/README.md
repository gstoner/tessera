# Runtime source maps and retained exception payloads

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a / BLOCK-ATTNRES-1.
Sync key: RUNTIME-SOURCE-MAPS-2026-09-09.

SM120 and gfx1151 independently pass twenty cases, including native runtime-shaped
2-D slice forward/backward at widths 2, 5 and 6, mismatched cotangent refusal,
and synchronous/asynchronous dynamic exception cause/context payload retention.
Each runtime map artifact is reused across input shapes. This is a native IR
fixture, not automatic capture of arbitrary runtime Python slice expressions.
Source and recorder fingerprints in both packets match the measured tree.

Super-Bear: 342 source/runtime/pipeline/registry tests passed, five skipped;
274 additional package/artifact/registry tests passed, one skipped (the sets
overlap). Princess-Luna: all ten runtime-map/cooperative compiler tests passed.
Package mypy (516 files) and touched-file Ruff passed. Both compilers rebuilt;
Super-Bear uses assertions-enabled LLVM 23. No full unit suite was run.

The same-path compiler-replacement cache regression passed. A supplemental full
`test_scheduled_matmul_consumers.py` run produced 33 passes, 18 skips and four
failures: this assertions compiler lacks the NVIDIA Target IR dialect required
by those SM120 matmul tests. This build-envelope limitation remains unresolved;
it does not invalidate the separately recorded CUDA runtime-map packets.
Audit governance passed (25 tests), and all 30 generated-document checks passed.

The two depth-attention packets use the same final compiler/recorder and distinct
HSACO hashes. Default versus cooperative operation-total medians (nanoseconds):

| Sources / rows / width | Default | Cooperative |
|---|---:|---:|
| 7 / 3 / 8 | 2049547 | 2061226 |
| 17 / 31 / 64 | 2850262 | 2727613 |
| 33 / 127 / 128 | 3744891 | 3676247 |

These are 15 post-warmup synchronized host-wall samples per shape, including
allocation, module/copy/launch overhead. They are not kernel/device-clock timings.
An earlier exploratory comparison had mixed changes in the opposite direction
on two shapes. No stable speedup or selector promotion is established. The
cooperative shared-memory tree is opt-in (`cooperative_width=True` on packaging;
`depth-cooperative=true` in the native pipeline). Baseline stays the default.
The candidate's maximum absolute error across these shapes is 1.08e-5.

Large positive rectangular source views now emit one native slice rather than
per-element expansion; CPU VJP uses its own 16M-element slot cap. Negative,
transposed/general maps remain bounded to 256 elements. Dynamic source exception
payloads have per-site SSA slots; new loop context graphs still require explicit
completion slots. CPU VJP checks forward exception status before backward and
uses zero cotangents for completion metadata. GPU exception AD refuses until a
product-aware checked forward dependency exists.

Full Python tracebacks are not claimed. Native source notes preserve logical
raise locations; actual Python traceback frames belong to the host bridge.

Recorded by `benchmarks/record_source_exception_gpu.py` (`nvidia.json`, `rocm.json`)
and `benchmarks/rocm/benchmark_block_attnres_gfx1151.py` (`depth_default.json`, `depth_cooperative.json`).
