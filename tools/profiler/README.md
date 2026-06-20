# tools/profiler - Tessera Profiler

Features:
- Chrome trace and Perfetto-compatible Trace Event JSON export.
- HTML report generation with hot-op and roofline views.
- `tprof peaks print` for device peak FLOP/s and HBM GB/s from YAML.
- `--peaks/--arch` wiring for reports.
- Runtime API, device activity, and intra-kernel sample event categories for
  backend collectors to populate.
- Compiler-side advanced profiler plans via
  `tessera.compiler.profiling_plan.plan_profile(...)` and
  `tessera-prof --advanced-plan --emit=json`.

## Advanced profiler backend plan

The compiler-facing plan keeps the integration honest: it emits a stable JSON
contract that says which provider owns each requested feature and whether that
provider is `available`, `planned`, or `unsupported`.

| Target | runtime API tracing | device activity tracing | counters | intra-kernel profiling | model analyzer |
| --- | --- | --- | --- | --- | --- |
| NVIDIA / CUDA | CUPTI Callback API | CUPTI Activity API | CUPTI profiler/range metrics | CUPTI PC sampling plus optional compiler-inserted probes | Tessera plan JSON; Triton Model Analyzer handoff for Triton-served exports |
| AMD / ROCm | ROCprofiler-SDK HIP/HSA/API tracing | ROCprofiler-SDK dispatch/device traces | ROCprofiler-SDK dispatch/device counter collection | PC sampling / thread trace where supported | Tessera plan JSON over HIP telemetry |
| Apple GPU / Metal | Tessera Metal/MPS runtime wrappers | Metal System Trace correlation | Metal counter sample buffers | compiler-inserted Target IR probes plus Metal counters | Tessera plan JSON over native Apple telemetry |
| CPU | Tessera C ABI wrappers | host/runtime spans | tprof counters | planned tile probes | Tessera config sweeps |

Design anchors:

- Triton Proton: Python context, user annotations, launch metadata, GPU metrics,
  CUPTI PC sampling, and an intra-kernel instrumentation backend.
- Intel Unitrace: separate host call logging, kernel/device timelines, metric
  query/sampling, include/exclude filters, and paused/resumed collection.
- ROCprofiler-SDK: the ROCm path for HIP/HSA/marker/memory tracing, counters,
  PC sampling, and thread trace.
- Apple Metal: counter sample buffers and timestamp/counter-set correlation.

Apple GPU validation on this host needs native, out-of-sandbox execution before
turning any `planned` row into `available`.

## Examples

```bash
# Build
cmake -S tools/profiler -B build/tprof -DCMAKE_BUILD_TYPE=Release
cmake --build build/tprof -j

# Generate traces and a report using device peaks from YAML.
./build/tprof/tprof \
  --demo-out demo.trace.json \
  --perfetto-out demo.perfetto.json \
  --report-out demo.report.html \
  --peaks tools/profiler/scripts/peaks_sample.yaml \
  --arch sm90

# Print peaks for CI logs.
./build/tprof/tprof peaks print --peaks tools/profiler/scripts/peaks_sample.yaml --arch sm90

# View report locally.
python3 tools/profiler/scripts/tprof_view.py --root . --file demo.report.html
```

YAML shape:

```yaml
devices:
  sm90: { peak_flops: 2.0e14, hbm_gbs: 3000 }
# or:
sm90: { peak_flops: 2.0e14, hbm_gbs: 3000 }
```
