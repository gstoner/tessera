---
status: Informative
classification: Guide
authority: Profiling and autotuning workflows; defers schedule artifact semantics to docs/spec/SHAPE_SYSTEM.md and compiler autotuner implementation
last_updated: 2026-06-20
---

# Tessera Profiling and Autotuning Guide

Performance optimization in Tessera is driven by two loops:

- **Profiler loop:** capture runtime metrics, timelines, counters, and memory
  movement to identify the bottleneck.
- **Autotuning loop:** search legal Schedule IR choices, measure or estimate
  candidates, persist the best schedule artifact, and reuse it by shape and
  target.

The current Python foundation provides `tessera.profiler` and a callable
`tessera.autotune` facade over the compiler autotuner. The lower compiler stack
already contains a Bayesian/grid GEMM autotuner with SQLite cache support and
schedule-artifact hashing.

---

## 1. Runtime Profiler

Use a profiler session to collect operation metrics:

```python
from tessera import profiler

with profiler.session() as p:
    p.record(
        "matmul",
        latency_ms=3.21,
        flops=214.5e9,
        bytes_moved=4.85e9,
        peak_tflops=233.0,
        counters={"sm_occupancy_pct": 92.0},
    )
    p.record("softmax", latency_ms=0.67, flops=2.1e9, bytes_moved=661e6)

print(p.report())
```

Example report:

```text
Op       Latency(ms)  FLOPs(G)  Bandwidth(GB/s)  Efficiency(%)
matmul   3.210        214.500   1510.903         28.7
softmax  0.670        2.100     986.567          0.0
```

For callable regions:

```python
with profiler.session() as p:
    y = p.measure("step", lambda: model(batch), flops=1.2e12, bytes_moved=80e9)
```

The CLI surface mirrors the runtime profiler:

```bash
tessera-prof my_model.py --metrics=flops,bandwidth,occupancy
tessera-prof my_model.py --trace=trace.json
tessera-prof my_model.py --emit=json --compile-target=sm90
tessera-prof my_model.py --advanced-plan --emit=json --compile-target=apple_gpu
tessera-prof my_model.py --advanced-plan --emit=json --compile-target=sm90 \
  --trace-features=runtime_api,device_activity,intra_kernel,model_analyzer \
  --profiler-context-json context.json \
  --model-analyzer-manifest model_analyzer.json \
  --model-analyzer-result model_analyzer_result.json
```

The current `tessera-prof` implementation records a lightweight inspection
event labeled `source_inspection`, can emit report, JSON, or Chrome Trace Event
JSON, and can correlate autotune schedule artifacts with profiler telemetry. As
device execution is wired through the runtime, this command should become the
stable front door for kernel latency, FLOPs, bandwidth, occupancy, memory,
collective, and launch metrics.

The C runtime now exposes a profiling callback spine for native execution
traces. Register `tsrSetProfileEventCallback`, call `tsrEnableProfiling(1)`,
and collect `TSR_PROFILE_RUNTIME_API` plus `TSR_PROFILE_DEVICE_ACTIVITY` events.
Payload JSON includes `status`, `duration_us`, and correlation fields such as
`kernel`, `target`, `bytes`, `memcpy_kind`, `grid`, `tile`, or `device_kind`.
`tessera-prof --advanced-plan` reports which backend provider should eventually
own each feature; v1 runtime callback events are the portable CPU/runtime proof
path before CUPTI, ROCprofiler-SDK, and Metal collectors are promoted from
`planned`.

`--model-analyzer-manifest` writes a runner-facing JSON contract derived from
the same advanced plan. The manifest carries the batch/instance/dynamic
batching search space, primary objective, required runtime/device telemetry,
planned output artifacts, and any compiler-inserted intra-kernel probe sites.
This is Tessera's local Model Analyzer handoff: CPU/runtime sweeps can consume
it immediately, while NVIDIA, ROCm, and Apple GPU runners stay gated on their
native collector integrations.

`--model-analyzer-result` runs the manifest search space through Tessera's local
Model Analyzer runner and writes a result JSON with every trial, the selected
best configuration, and runner status. Unless a future backend runner supplies
real measurements, the default result is explicitly marked as estimated or
planned-estimated rather than hardware-measured.
Result artifacts can attach profiler context summaries, provider status
snapshots, merged trace paths, and provider requirements. If a manifest requires
an unavailable native provider feature, the result reports
`provider_requirements.met = false` and adds `provider_requirements_unmet` to
the bottleneck labels instead of treating mock/file data as native proof.

`tessera.compiler.profiler_context` defines the portable
`tessera.profiler_context.v1` artifact used to correlate traces with lower-rate
system context. Apple, NVIDIA, and AMD context samples normalize to:

- `provider` such as `apple-silicon-system-context`, `nvidia-system-context`,
  or `rocm-system-context`.
- `source_status` such as `planned`, `compiled_shell`, or `measured`.
- A `bottleneck_summary` over labels such as `bandwidth_bound`,
  `compute_bound`, `power_capped`, `thermal_throttled`, `fabric_limited`, or
  `reliability_risk`.

The same artifact can be consumed by the HTML report generator:

```bash
python3 tools/profiler/scripts/tprof_report.py \
  --in runtime.trace.json \
  --context-json context.json \
  --out report.html
```

Use `tools/profiler/scripts/tprof_context.py` to create the artifact:

```bash
# Deterministic CI path.
python3 tools/profiler/scripts/tprof_context.py \
  --provider mock --target nvidia --out context.json

# File ingestion path for native helper output or recorded samples.
python3 tools/profiler/scripts/tprof_context.py \
  --provider mock --target apple_gpu --input sample.json --out context.json

# Best-effort native paths; missing libraries/devices still emit a valid
# context artifact with source_status="unavailable".
python3 tools/profiler/scripts/tprof_context.py --provider nvidia --out nvml.context.json
python3 tools/profiler/scripts/tprof_context.py --provider rocm --out amdsmi.context.json
python3 tools/profiler/scripts/tprof_context.py --provider apple --out apple.context.json
```

Unavailable optional probes keep diagnostic metadata such as `error`,
`error_type`, and adapter retry details in the sample `metadata` block so SDK
absence, permission failures, and method signature drift can be separated in
reports.

and by `tessera-prof --profiler-context-json context.json` when writing a Model
Analyzer result. This attaches context to the result artifact; it does not
promote native collector status by itself.

When `intra_kernel` is requested, the compiler plan emits portable probe specs
for `prologue`, `mainloop`, and `epilogue` phases. Those specs use stable
payload fields (`kernel`, `phase`, `tile`, `program_id`) so later backend
implementations can lower them to CUPTI PC sampling correlation, ROCprofiler
thread trace, Metal counter correlation, or inserted counters without changing
the JSON shape.

`tessera.compiler.target_ir.annotate_target_ir_with_probes(...)` attaches those
probe specs to backend-facing Target IR as per-target profiler marker ops
(`tessera_nvidia.profiler_probe`, `tessera_rocm.profiler_probe`,
`tessera_apple.gpu.profiler_probe`, or `tessera.cpu.profiler_probe`). These ops
verify like any other Target IR op and provide the compiler-side anchor that a
later backend can lower to native counters or inserted instrumentation.

Apple support also has a system-context lane inspired by SiliconScope. Tessera
models the pure part in `tessera.compiler.apple_profiler_context`: unified
memory bandwidth ceilings, GPU-usage/bandwidth bottleneck classification, and a
JSON contract for future IOReport/SMC/HID samples. This context can explain why
an Apple run is `bandwidth_bound`, `compute_bound`, `thermal_throttled`, or
`memory_pressured`, but it is not a replacement for Metal System Trace or Metal
counter sample buffers.

The same separation applies to NVIDIA and AMD. Use
`tessera.compiler.accelerator_profiler_context` for hardware-free classification
of NVML/DCGM-style or AMD SMI/RDC-style samples: utilization, memory residency,
power, thermals, clocks, PCIe/NVLink/XGMI pressure, and reliability events. The
context lane can label a run `bandwidth_bound`, `compute_bound`, `power_capped`,
`fabric_limited`, or `reliability_risk`; CUPTI and ROCprofiler-SDK still own the
runtime/activity/counter records that prove what kernels actually did.

`tools/profiler/include/tprof/provider_shells.h` is the native integration
handoff. It lists lightweight system-context shells for NVML/DCGM, AMD SMI/RDC,
and Apple IOReport/SMC/HID-style sampling, plus heavier provider shells for
CUPTI callbacks/activity/counters, ROCprofiler-SDK dispatch/counters, and Metal
command-buffer/counter-sample correlation. The shell API is intentionally
compilable without vendor SDKs; backend proof must wire the SDK, collect data,
and update provider status before docs or plans say `available`.

The first native context collectors are best-effort and dynamic:

- NVIDIA loads NVML at runtime with no hard link and samples utilization, memory
  residency, power/limit, temperature, throttle reasons, and ECC totals when
  available.
- AMD loads the optional `amdsmi` Python module and normalizes GPU/memory
  activity, VRAM, power, temperature, and RAS-style counts when available.
- Apple emits host-safe metadata only for now; IOReport/SMC/HID stays behind
  the macOS-only `TPROF_WITH_APPLE_SYSTEM_CONTEXT` shell guard and still needs
  fresh-process, out-of-sandbox proof before promotion.

`tessera.compiler.profiler_provider_trace` is the next layer down: it describes
the heavy provider records that prove API calls, device activity, counters, and
intra-kernel samples. The artifact schema is
`tessera.profiler_provider_trace.v1`, with each record also rendered as
Chrome/Perfetto-compatible Trace Event JSON.

```bash
python3 tools/profiler/scripts/tprof_provider_trace.py \
  --provider rocprofiler \
  --input rocprofiler_records.json \
  --input rocprofiler_more_records.json \
  --out provider_trace.json \
  --trace-out provider_trace.trace.json
```

Provider readiness is a separate artifact/CLI so native availability can be
reported without starting a collector:

```bash
python3 tools/profiler/scripts/tprof_provider_status.py --provider apple
python3 tools/profiler/scripts/tprof_provider_status.py --provider rocm
python3 tools/profiler/scripts/tprof_provider_status.py --provider nvidia
```

Merge provider records with Tessera runtime trace and system context before
rendering the final report:

```bash
python3 tools/profiler/scripts/tprof_merge_trace.py \
  --runtime-trace runtime.trace.json \
  --provider-trace provider_trace.json \
  --context-json context.json \
  --provider-status rocm.status.json \
  --out merged.trace.json
```

Merged trace validation is strict about Trace Event timestamps. A non-numeric
`ts` is reported as malformed input before sorting, rather than silently being
treated as timestamp zero.

Provider trace artifacts may carry `provider_statuses` sidecars. The merge
tool lifts those sidecars into `provider_status` marker events so report and
Model Analyzer consumers see the same availability diagnostics whether status
was supplied separately or bundled with a provider replay. Reports can also
write `tessera.profiler_report_summary.v1` via
`tools/profiler/scripts/tprof_report.py --summary-json`, including hot ops,
derived arithmetic intensity, provider/category grouping, dropped records,
correlation counts, provider diagnostics, and context bottleneck summaries.
Provider traces additionally contribute Chrome/Perfetto metadata markers for
provider, backend, target, source status, record source, record count, and
dropped-record totals.

The staged mapping is:

- ROCprofiler-SDK first maps HIP/HSA callback records to `runtime_api`.
- ROCprofiler-SDK dispatch, kernel, memcpy, and memory records map to
  `device_activity`.
- ROCprofiler-SDK counters map to `counters`; filtered thread-trace records map
  to `intra_kernel` and must carry dispatch/kernel correlation fields.
- Metal command-buffer timestamp records map to `device_activity`; Metal counter
  sample buffer records map to `counters` and should carry command-buffer or
  Target IR probe correlation IDs. Apple provider status remains
  `compiled_shell` until `tools/profiler/scripts/tprof_apple_metal_smoke.py`
  proves `MTLCreateSystemDefaultDevice` in a fresh native process plus
  command-buffer timestamp or counter-set evidence.
  `--prove-counters` adds native `MTLDevice.counterSets` capability discovery
  diagnostics without collecting counters or claiming availability.
  `--prove-command-buffer` can call a compiled tprof Metal adapter library
  exported through `TPROF_METAL_ADAPTER_LIB` or `--adapter-library`.
- CUPTI runtime/driver callback records map to `runtime_api`.
- CUPTI kernel, memcpy, memset, and device activity records map to
  `device_activity`, preserving CUPTI correlation IDs so activity can be joined
  to the originating API callback.

`tprof_rocm_native_smoke.py` and `tprof_nvidia_cupti_smoke.py` emit hardware
proof snapshots without requiring CI hardware. On hosts without AMD/NVIDIA
devices they return `native_failed` with library/device diagnostics when run
with `--allow-unavailable`; they only promote to `native_available` once the
required callback/activity proof fields are true.

The first C++ SDK adapter shims are intentionally thin:

- `tprof/rocprofiler_adapter.h` exposes HIP/HSA API, dispatch/activity, counter,
  and thread-trace ingestion and replay functions.
- `tprof/metal_adapter.h` exposes command-buffer and counter-sample ingestion
  and replay functions.
- `tprof/cupti_adapter.h` exposes runtime/driver callback and activity
  ingestion and replay functions.

Their init functions are SDK-gated and can return `false`, but the ingestion
functions feed normalized fixture or callback data into the existing `tprof`
runtime categories. Each adapter also exposes `*_adapter_status()` with
compiled/initialized/paused/source-status fields so hardware hosts can
distinguish a planned shell from an SDK-compiled shell without promoting
availability.

ROCprofiler thread trace remains explicitly bounded: configure
`thread_trace_max_bytes`, keep thread trace opt-in, and treat
`thread_trace_volume_limited` as a dropped-record warning. This follows
ROCprofiler-SDK guidance that thread trace can generate high-volume data and
should be filtered to the kernels of interest.

Profiler events carry:

- Latency in milliseconds.
- FLOPs in G.
- Derived bandwidth in GB/s.
- Optional peak-normalized efficiency.
- Hardware counters such as SM occupancy, warp divergence, L2 hit rate, and
  shared-memory bank conflicts.

---

## 2. Timeline Traces

Profiler sessions export Chrome Trace Event JSON:

```python
with profiler.session() as p:
    p.record("matmul", latency_ms=3.21)
    p.record("softmax", latency_ms=0.67)

p.timeline("trace.json")
```

Open the resulting file in Chrome trace viewers or compatible observability
systems. Future runtime integrations should attach stream, device, rank, kernel
launch ID, graph hash, and schedule hash to each trace event.

---

## 3. Cost Models

Schedule IR should make the cost model explicit:

```mlir
%1 = "tessera.schedule.tile"(%0)
       {tile_sizes = [128, 128, 32], cost_model = "roofline"}
```

The public roofline helper estimates compute-vs-memory bounds:

```python
from tessera import autotune

model = autotune.RooflineCostModel(peak_tflops=312.0, bandwidth_gbps=2000.0)
estimate = model.estimate(flops=2 * 1024**3, bytes_moved=6 * 1024**2)
print(estimate.bound, estimate.latency_ms)
```

Tessera cost models should be keyed by:

- Op kind.
- Logical shape and derived shape bindings.
- Dtype and numeric policy.
- Layout and shard map.
- Movement plan.
- Target architecture and relevant hardware features.

---

## 4. Autotuning Workflow

The public entry point tunes GEMM-like workloads:

```python
from tessera import autotune, ops

cfg = autotune(ops.matmul, shapes=(1024, 1024, 1024), max_trials=20)
print(cfg)
```

Representative result:

```text
TuningResult(tflops=..., latency_ms=..., config=TuningConfig(...))
```

Current methods:

- `method="roofline"`: synthetic analytical evaluator.
- `method="grid"` / `method="bayesian"`: select the search policy path through
  the compiler autotuner when dependencies are available.
- `method="on_device"`: accepted as the forward-compatible API for measured
  candidate execution; until runtime kernels are wired, it uses the same
  deterministic evaluator.

Autotuning should run after shape and schedule feasibility have pruned illegal
tile/layout choices. It should never benchmark candidates that violate shape
divisibility, target fragment legality, shared-memory budgets, or movement
constraints.

---

## 5. Persistent Caches

Tessera stores tuning results in SQLite:

```text
$TESSERA_CACHE_DIR/autotune/tuning_cache.db
```

If `TESSERA_CACHE_DIR` is unset, the default is:

```text
$HOME/.tessera/autotune/tuning_cache.db
```

Public helpers:

```python
best = autotune.load(ops.matmul, (1024, 1024, 1024))
key = autotune.cache_key(ops.matmul, (1024, 1024, 1024), dtype="bf16", arch="sm90")
artifact = autotune.schedule_artifact(best, op=ops.matmul, shapes=(1024, 1024, 1024))
```

Cache keys must include:

- Operation.
- Shape.
- Dtype and numeric policy.
- Architecture.
- Layout and shard map when available.
- Movement plan for kernels whose performance depends on staging/prefetch.

---

## 6. On-Device Measurements

The runtime autotuner should support online measurement:

```python
cfg = autotune(
    ops.matmul,
    shapes=(8192, 8192, 8192),
    method="on_device",
    max_trials=50,
)
```

The foundation implementation accepts `method="on_device"`. On targets with a
real execution path (x86 CPU, **Apple CPU/GPU** — the Apple GPU lane has a
recorded perf ratchet, `perf_gate --ratchet`) measured timing is available; on
artifact-only targets (NVIDIA/ROCm, pending Phase G/H hardware) the result is
marked `status="unmeasured"` with a reason explaining that no device timer
exists yet. Either way it emits the same schedule artifact and telemetry schema
as synthetic tuning, so downstream tooling is written once and upgrades to real
timing as each backend's runtime lands.

Required behavior for the production implementation:

- Compile candidates with fixed graph/schedule hashes.
- Warm up kernels before measuring.
- Use CUDA/HIP events or equivalent device timers.
- Record p50/p95 latency and variance.
- Capture failure diagnostics for bad candidates.
- Persist both failed and successful trials for future pruning.
- Export measurements for learned surrogate cost models.

---

## 7. Advanced Profiling

Profiler events may include counters:

```python
p.record(
    "flash_attention",
    latency_ms=1.8,
    flops=900e9,
    bytes_moved=420e6,
    counters={
        "sm_occupancy_pct": 91.0,
        "warp_divergence_pct": 2.0,
        "l2_hit_pct": 78.0,
        "shared_bank_conflict_factor": 1.03,
    },
)
```

Recommended counter groups:

- Compute: tensor core active, achieved TFLOPs, eligible warps/cycle.
- Memory: HBM throughput, L2 hit rate, shared-memory replay factor.
- Control: branch efficiency, warp divergence, barrier stalls.
- Distributed: all-reduce time, overlap percentage, bytes per collective.
- Runtime: launch latency, host wait time, queue depth.

---

## 8. Compiler Integration Contract

| Component | Responsibility |
|-----------|----------------|
| Graph IR | Preserve op identity, shape, dtype, numeric policy, and graph hash |
| Schedule IR | Expose tunable knobs, movement plans, layout choices, and cost model attributes |
| Tile IR | Verify target legality and expose resource estimates |
| Runtime Profiler | Measure latency, counters, bytes, and trace events |
| Autotuner | Search legal candidates, persist measurements, emit schedule artifacts |

Current compiler hooks include:

- `schedule.knob` metadata for GEMM tile sizes, warps, and stages.
- `schedule.artifact` metadata with movement, numeric policy, and roofline cost model.
- Tile IR resource estimates for MMA, async copy, queue/barrier, and softmax-style reductions.
- Target IR launch metadata and target feature metadata for CPU/x86, Apple, NVIDIA, and ROCm artifacts.
- Compile-bundle profiling correlation fields for graph, schedule, tile, and target hashes.
| Learned Surrogate | Train on autotuner measurements and predict latency/energy/memory |

Schedule artifacts should contain:

- Shape and architecture.
- Numeric policy.
- Movement plan.
- Tile knobs.
- Measured or estimated metrics.
- Stable hash.

The command foundation is in `python/tessera/cli/prof.py` and is installed as:

- `tessera-prof`

---

## 9. Best Practices

- Profile before tuning; do not tune blind.
- Always record shape, dtype, target, graph hash, and schedule hash.
- Keep autotune search spaces legal and small enough for CI smoke tests.
- Use persistent caches in development, but invalidate them when target,
  numerics policy, movement plan, or lowering changes.
- Use deterministic seeds and fixed clocks where possible for microbenchmarks.
- Export traces for any performance regression report.
