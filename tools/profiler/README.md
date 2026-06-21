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
- Optional `tprof` adapter for the Tessera runtime callback ABI.
- Runner-facing Model Analyzer manifest JSON and compiler-inserted
  intra-kernel probe specs.
- `tessera.profiler_context.v1` context artifacts for correlating traces with
  Apple, NVIDIA, and AMD system-context samples.
- `tprof-context` sample CLI for mock/file/native-best-effort context capture.
- `tessera.profiler_provider_trace.v1` artifacts for normalizing ROCprofiler,
  CUPTI, and Metal callback/activity/counter records into Trace Event JSON.
- `tprof-merge-trace` for combining runtime trace, provider trace, and context
  artifacts into one report-ready Trace Event JSON file.

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
- NVIDIA system context: NVML/DCGM for utilization, memory residency, clocks,
  power, thermals, PCIe/NVLink, and reliability state. This complements CUPTI;
  it does not replace CUPTI activity/profiler records.
- AMD system context: AMD SMI/RDC for utilization, memory residency, clocks,
  power, thermals, PCIe/XGMI, and RAS state. This complements ROCprofiler-SDK;
  it does not replace ROCprofiler dispatch/counter records.
- Apple Metal: counter sample buffers and timestamp/counter-set correlation.
- SiliconScope-style Apple context: sudoless IOReport deltas for GPU residency,
  GPU clock, unified-memory bandwidth, power domains, memory pressure, and
  thermal throttling. Tessera models the pure classification contract today;
  a native collector remains separate from Metal timeline/counter proof.

Apple GPU validation on this host needs native, out-of-sandbox execution before
turning any `planned` row into `available`.

## Context artifacts and provider shells

`tessera.compiler.profiler_context.build_profiler_context_artifact(...)`
normalizes Apple, NVIDIA, and AMD system-context samples into
`tessera.profiler_context.v1` JSON:

```python
from tessera.compiler import (
    AcceleratorProfilerContext,
    build_profiler_context_artifact,
    write_profiler_context_artifact,
)

context = build_profiler_context_artifact(
    target="nvidia",
    samples=[
        AcceleratorProfilerContext(
            vendor="nvidia",
            gpu_utilization=0.70,
            memory_utilization=0.90,
            memory_used_fraction=0.10,
        ).to_dict()
    ],
)
write_profiler_context_artifact(context, "context.json")
```

Reports can ingest this alongside Trace Event JSON:

```bash
python3 tools/profiler/scripts/tprof_report.py \
  --in runtime.trace.json \
  --context-json context.json \
  --out report.html
```

`tools/profiler/scripts/tprof_context.py` is the sample CLI for producing those
artifacts:

```bash
# Stable CI/mock artifact.
python3 tools/profiler/scripts/tprof_context.py \
  --provider mock --target nvidia --out context.json

# Normalize a raw sample/list JSON or an existing context artifact.
python3 tools/profiler/scripts/tprof_context.py \
  --provider mock --target rocm --input sample.json --out context.json

# Best-effort native sampling. Missing libraries/devices still emit a valid
# artifact with source_status="unavailable".
python3 tools/profiler/scripts/tprof_context.py --provider nvidia --out nvml.context.json
python3 tools/profiler/scripts/tprof_context.py --provider rocm --out amdsmi.context.json
python3 tools/profiler/scripts/tprof_context.py --provider apple --out apple.context.json
```

Native context collectors are intentionally dynamic and optional:

- NVIDIA uses NVML through `ctypes`/runtime library discovery, with no hard link
  requirement. It normalizes utilization, memory residency, power/limit,
  temperature, throttle reasons, and ECC totals when those symbols are present.
- AMD uses an optional `amdsmi` Python module when installed, normalizing GPU and
  memory activity, VRAM, power, temperature, and RAS-style counts.
- Apple currently emits host-safe metadata only. IOReport/SMC/HID sampling stays
  behind the macOS-only `TPROF_WITH_APPLE_SYSTEM_CONTEXT` shell guard until it
  has out-of-sandbox native proof.

The C++ shell API in `tprof/provider_shells.h` lists the native context lanes
(`nvidia-system-context`, `rocm-system-context`,
`apple-silicon-system-context`) and the heavier providers
(`cupti-activity-callbacks`, `rocprofiler-sdk-dispatch-counters`,
`metal-command-buffer-counters`). The shells compile without vendor SDKs and
return planned/unavailable status until backend-specific collectors are wired
and validated. CMake options `TPROF_WITH_CUPTI`, `TPROF_WITH_ROCPROFILER`, and
`TPROF_WITH_METAL` are the guarded entry points for those integrations;
`TPROF_WITH_APPLE_SYSTEM_CONTEXT` is the separate Apple system-context helper
gate.

## Heavy provider trace normalization

`tessera.compiler.profiler_provider_trace` is the executable contract for the
heavier SDK integrations. It accepts recorded or callback-fed mappings and
normalizes them to `tessera.profiler_provider_trace.v1` plus Chrome/Perfetto
Trace Event JSON.

The integration order is encoded in the shell capability bits:

1. ROCprofiler-SDK HIP/HSA callback tracing.
2. ROCprofiler-SDK dispatch/activity records.
3. ROCprofiler-SDK counter collection and filtered thread trace.
4. Metal command-buffer timestamp spans and counter sample records.
5. CUPTI runtime/driver callback API events.
6. CUPTI activity records for kernels, memcpy, memset, and other device work.

Use the fixture/normalization CLI before native SDK callbacks are wired:

```bash
python3 tools/profiler/scripts/tprof_provider_trace.py \
  --provider rocprofiler \
  --input rocprofiler_records.json \
  --out provider_trace.json \
  --trace-out provider_trace.trace.json
```

Then merge runtime, provider, and context artifacts into one report input:

```bash
python3 tools/profiler/scripts/tprof_merge_trace.py \
  --runtime-trace runtime.trace.json \
  --provider-trace provider_trace.json \
  --context-json context.json \
  --out merged.trace.json
```

ROCprofiler records map HIP/HSA callback records to `runtime_api`, dispatch and
memory records to `device_activity`, counter records to `counters`, and thread
trace records to `intra_kernel`. Metal command-buffer spans map to
`device_activity`; Metal counter sample buffer records map to `counters` and
carry the command-buffer or Target IR probe correlation ID. CUPTI callback
records map to `runtime_api`; CUPTI activity records map to `device_activity`
and preserve CUPTI correlation IDs so kernels/memcpy/memset can be joined back
to runtime or driver API calls.

The SDK adapter shims are split by provider:

- `tprof/rocprofiler_adapter.h`: HIP/HSA API tracing first, dispatch/activity
  records next, counters and thread trace behind explicit config flags.
- `tprof/metal_adapter.h`: command-buffer span ingestion first, counter sample
  records behind an explicit config flag.
- `tprof/cupti_adapter.h`: runtime/driver callback ingestion first, activity
  records next.

These adapters currently ingest normalized callback data into `tprof` event
categories. Their `*_init` functions return `false` without the relevant compile
guard/SDK, but the record-ingestion functions remain usable for fixtures and
tests.

All provider adapters expose start-paused/pause/resume controls and simple
include/exclude filters. ROCprofiler filters API names and kernel/dispatch
names; CUPTI filters API and activity names; Metal filters command-buffer labels
and counter names. Counter and thread-trace paths are opt-in to avoid accidental
high-volume captures.

`TPROF_WITH_ROCPROFILER=ON` now runs CMake detection for
`rocprofiler-sdk/rocprofiler.h` and a ROCprofiler-SDK library before defining
`TPROF_WITH_ROCPROFILER`; if either is missing, the adapter remains a planned
shell and emits a configure-time warning.

## Runtime callback trace spine

The runtime ABI emits v1 profiling events through a callback instead of linking
the core runtime to `tprof`:

```c
static void on_profile(TsrProfileEventKind kind, const char* name,
                       const char* payload_json, double value, void* user) {
  (void)user;
  if (kind == TSR_PROFILE_RUNTIME_API) {
    /* payload_json has status, duration_us, and call-specific fields. */
  } else if (kind == TSR_PROFILE_DEVICE_ACTIVITY) {
    /* value is elapsed microseconds for tprof-style duration records. */
  }
}

tsrSetProfileEventCallback(on_profile, NULL);
tsrEnableProfiling(1);
```

`payload_json` is owned by the runtime and valid only during the callback.
Runtime API events cover lifecycle, memory, artifact, and launch calls. Portable
device activity events cover CPU/runtime work such as `tsrMemcpy`,
`tsrMemset`, `tsrLaunchHostTileKernel`, and `tsrNativeGemmF32`.

Profiler harnesses that link both `libtessera_runtime` and `tprof_runtime` can
use the optional adapter instead of hand-writing the callback:

```c++
#include "tprof/tessera_runtime_adapter.h"

tprof::enable(tprof::config_t{});
tprof::attach_tessera_runtime_trace();
/* run Tessera runtime work */
tprof::detach_tessera_runtime_trace();
tprof::export_chrome("runtime.trace.json");
```

The dependency direction is one-way: `tools/profiler` includes the runtime ABI
headers, but `src/runtime` does not link against `tprof`.

## Intra-kernel and Model Analyzer contracts

`tessera.compiler.profiling_plan.plan_profile(..., features=["intra_kernel"])`
now emits deterministic compiler-inserted probe specs. The first portable shape
uses `prologue`, `mainloop`, and `epilogue` phases with payload fields such as
`kernel`, `phase`, `tile`, and `program_id`. Backend collectors can later map
those probes to CUPTI PC sampling, ROCprofiler thread trace, Metal counters, or
plain inserted counters without changing the compiler-facing JSON.

`tessera.compiler.profiling_plan.model_analyzer_manifest(plan)` converts the
same provider plan into a runner-facing Model Analyzer manifest with the search
space, objective, required telemetry features, planned probe sites, and output
artifacts. `tessera.compiler.model_analyzer.run_model_analyzer_manifest(...)`
executes that search contract as a local result artifact; default trials are
estimated/planned-estimated unless a backend runner supplies real measurements.
It can also attach a `tessera.profiler_context.v1` summary so trial results can
carry bottleneck context without claiming native collector availability.
The CLI can write both files directly:

```bash
tessera-prof my_model.py \
  --advanced-plan --emit=json --compile-target=sm90 \
  --trace-features=runtime_api,device_activity,intra_kernel,model_analyzer \
  --profiler-context-json context.json \
  --model-analyzer-manifest model_analyzer.json \
  --model-analyzer-result model_analyzer_result.json
```

`tessera.compiler.target_ir.annotate_target_ir_with_probes(...)` attaches the
planned probes to Target IR as backend-specific profiler marker ops, providing
the compiler-side anchor for future CUPTI, ROCprofiler-SDK, Metal counter, or
inserted-counter lowering.

For Apple support, `tessera.compiler.apple_profiler_context` provides a
hardware-free version of the SiliconScope-style context layer: a bandwidth
ceiling table, bottleneck classifier, and JSON contract for future IOReport /
SMC / HID samples. Use it to correlate Tessera runtime traces with system-level
signals such as `bandwidth_bound`, `compute_bound`, `thermal_throttled`, or
`memory_pressured`; do not treat those signals as native Metal command-buffer
activity until the Metal/System Trace collector lands.

The same pattern applies to NVIDIA and AMD through
`tessera.compiler.accelerator_profiler_context`. NVIDIA context should come from
NVML/DCGM, while AMD context should come from AMD SMI/RDC. The normalized
classifier adds accelerator-specific verdicts such as `power_capped`,
`fabric_limited`, and `reliability_risk`, while CUPTI and ROCprofiler-SDK remain
the authoritative providers for runtime/device activity and hardware-counter
proof.

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
