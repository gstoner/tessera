---
status: Draft
classification: Engineering Specification
authority: CITL trace/profiler prototype for ROCm systems
last_updated: 2026-05-01
---

# CITL ROCm Trace and Profiler Specification

This document scopes the low-level KFD, eBPF, HIP runtime, ROCr, rocprofiler,
and AMD SMI changes needed for a first Compiler-in-the-Loop trace/profiler
prototype. The purpose is measurement and attribution, not mitigation. Compiler
policy decisions can consume this data later.

## 1. Problem

CITL needs to associate compiler decisions with what the GPU and platform did at
runtime: kernel timing, counters, power, thermal state, clocks, throttle bits,
clock-gating state, queue events, and reliability events. The hard part is not
only reading those values. It is aligning them to a specific compiled operation,
schedule, kernel dispatch, stream, queue, process, GPU, and timestamp domain.

The first profiler must solve five concrete issues:

1. **Per-dispatch identity**: each launched kernel needs a stable CITL
   fingerprint that survives lowering into HIP/ROCr dispatches.
2. **Timestamp alignment**: compiler/runtime events, rocprofiler GPU records,
   AMD SMI platform samples, and kernel-driver events must be joinable.
3. **Sampling aliasing**: AMD SMI power and thermal samples can be slower than
   short fused kernels, so samples need nearby dispatch context and energy
   accumulation where available.
4. **KFD visibility limits**: normal HSA/HIP user queues do not require a KFD
   syscall per kernel packet, so KFD/eBPF should not be treated as the primary
   per-kernel tracer.
5. **Clock-gating ambiguity**: clock-gating and deep-sleep state are platform
   power-management state, not a kernel-local measurement. The profiler should
   report it as context and correlation evidence, not as direct causation.

## 2. First Prototype Boundary

The first concept should be implemented as a user-mode profiler with optional
kernel trace context.

Required:

- rocprofiler-sdk activity tracing for HIP/HSA calls, kernel dispatch records,
  memory copies, correlation IDs, counters, and optional PC sampling.
- HIP/ROCr runtime metadata hooks that attach CITL external correlation IDs and
  compiler fingerprints to dispatches.
- AMD SMI sampling for power, energy, temperatures, clocks, voltages, throttle
  status, ECC/RAS, and clock/deep-sleep fields where exposed.
- Perfetto/Chrome trace export using Tessera's existing profiler direction.

Optional:

- eBPF consumers for existing KFD/amdgpu tracepoints where available.
- New KFD/amdgpu tracepoints for queue lifecycle, eviction, restore, reset, VM
  fault, and power-management transitions if the deployed kernel lacks them.
- HIP stream memory markers for controlled experiments, not as the default
  tracing path.

Out of scope for the first prototype:

- Kernel-driver patches that attempt to inspect every user-mode dispatch packet.
- Automatic DVFS, throttling, clock locking, power-cap changes, or mitigation.
- Claims of direct per-kernel rail droop or clock-gating causality.

## 3. Measurement Architecture

```
Tessera compiler
  emits CITL fingerprint, graph hash, schedule hash, kernel metadata
        |
        v
HIP/ROCr launch wrapper
  pushes rocprofiler external correlation id and ROCTx range
        |
        v
rocprofiler-sdk
  records HIP/HSA callbacks, kernel dispatches, counters, PC samples
        |
        +-------------------+
                            v
AMD SMI sampler ----> trace joiner ----> Perfetto/Chrome trace + JSON report
                            ^
                            |
KFD/eBPF context events ----+
```

The profiler stores one trace record per dispatch plus sampled platform records
around the dispatch window. A dispatch can have many platform samples, and a
platform sample can overlap many dispatches.

## 4. HIP and ROCr Runtime Changes

### 4.1 Required HIP Wrapper

Tessera should introduce a thin HIP launch wrapper for ROCm targets:

```cpp
struct CitlDispatchMeta {
  uint64_t citl_id;
  const char* graph_hash;
  const char* schedule_hash;
  const char* kernel_fingerprint;
  const char* op_name;
  const char* kernel_name;
  uint32_t logical_device;
  uint32_t stream_id;
  uint32_t launch_ordinal;
};
```

The wrapper must:

- create a monotonic `citl_id` per process;
- push the `citl_id` as a rocprofiler external correlation ID before the HIP
  launch and pop it after the API call returns;
- emit a ROCTx range named with `op_name`, `kernel_fingerprint`, and
  `schedule_hash`;
- record host-side launch timestamps using a monotonic clock;
- record stream, device, grid, block, shared-memory, and kernel symbol metadata;
- pass through all HIP errors without wrapping them into profiler-only errors.

### 4.2 ROCr Dispatch Timestamp Support

Where available, the profiler should use ROCr dispatch profiling timestamps to
reduce ambiguity between host API launch time and actual GPU execution time.
ROCr's `hsa_amd_profiling_get_dispatch_time` can provide dispatch timing for
signals created with profiling enabled. This is useful for microbenchmarks and
calibration runs. It should not be required for production tracing because it
may require different signal handling and could add overhead.

### 4.3 HIP Stream Memory Markers

HIP stream memory operations such as `hipStreamWriteValue32/64` and
`hipStreamWaitValue32/64` can place ordered markers in a stream. For CITL they
are useful for controlled timestamp-calibration experiments:

- write marker before a kernel;
- launch kernel;
- write marker after the kernel;
- use host polling or mapped signal memory to align host and GPU ordering.

They should not be used as the default power sampling mechanism. They are beta
APIs, add synchronization/ordering complexity, and do not directly capture SMU
power state.

## 5. rocprofiler-sdk Requirements

The first profiler should build on rocprofiler-sdk rather than bypass it.

Required services:

- callback tracing for HIP runtime, HSA core, HSA AMD extension, and marker
  APIs;
- buffered tracing for kernel dispatch, memory copy, memory allocation, and
  scratch memory records;
- counter collection for selected dispatches or sampled windows;
- PC sampling for hotspot attribution on supported GPUs;
- external correlation IDs to join compiler metadata to asynchronous GPU
  records.

Required dispatch fields:

- agent ID;
- queue ID;
- dispatch ID;
- kernel ID;
- kernel name;
- correlation ID;
- start timestamp;
- end timestamp;
- workgroup size;
- grid size;
- private segment size;
- group segment size.

Useful initial counters:

- wave count and wave cycles;
- graphics/compute busy or GUI active counters;
- L2, memory, and cache counters available for the target ASIC;
- LDS activity where available;
- VALU/MFMA issue or instruction mix proxies where available;
- occupancy-derived metrics.

PC sampling should be optional and sampled. It is high-value for mapping risk to
instruction regions but too expensive for always-on profiling.

## 6. AMD SMI Sampler Requirements

AMD SMI is the primary platform telemetry source. The sampler should run in the
same trace session as rocprofiler and emit timestamped samples.

Minimum fields:

- current socket power;
- average socket power;
- energy accumulator;
- edge, hotspot/junction, memory, HBM, VR, GFX VR, SOC VR temperatures where
  available;
- GFX, SOC, memory, VCLK, and DCLK clocks;
- voltage fields such as GFX, SOC, memory, and board voltage where available;
- throttle status and independent throttle status;
- fan speed where relevant;
- ECC counts and RAS/CPER records;
- firmware timestamp if exposed by GPU metrics.

Clock-gating and clock-state fields:

- `clk_deep_sleep` from clock info where exposed;
- GFX clock lock status;
- throttle bits;
- SMU feature or clock/power-gating state from supported driver/debug
  interfaces when available;
- `amdgpu_pm_info` only for lab/debug collection, because debugfs is not a
  stable production telemetry API.

Sampling rules:

- support a high-rate mode for calibration and a low-rate mode for normal
  profiling;
- record sampler period, jitter, and dropped sample count;
- allow AMD SMI metric caching to be disabled in calibration runs when supported;
- never treat a sample as instantaneous per-kernel truth unless the timestamp
  domain and firmware semantics make that valid;
- prefer energy deltas over instantaneous power for dispatch windows longer than
  the sampler period.

## 7. KFD and eBPF Scope

KFD/eBPF is valuable for context, not as the first per-dispatch source of truth.
KFD manages process devices, queues, memory, doorbells, eviction, restore,
preemption, VM faults, and reset paths. After a user-mode queue is established,
normal dispatch packets can be written by user-space and consumed by the GPU
without a KFD syscall per packet. Therefore, eBPF attached to KFD functions will
miss ordinary per-kernel packet writes unless the kernel path is explicitly
involved.

### 7.1 eBPF Events to Consume

The profiler should consume existing tracepoints or kprobes for:

- KFD process create/destroy and PASID association;
- queue create/destroy/update;
- queue map/unmap to hardware scheduler;
- queue eviction, preemption, restore, and suspension;
- doorbell mmap and doorbell assignment;
- GPU reset begin/end;
- VM fault, retry fault, and memory migration events;
- RAS or fatal error reports surfaced through amdgpu/KFD;
- amdgpu power-management transitions if exposed as tracepoints.

These events become Perfetto context tracks. They explain discontinuities in
kernel timing, power samples, or counter collection.

### 7.2 New Tracepoints Worth Adding

If kernel changes are allowed, add stable tracepoints instead of relying on
fragile kprobes:

```text
amdkfd:kfd_queue_create
amdkfd:kfd_queue_destroy
amdkfd:kfd_queue_update
amdkfd:kfd_queue_map
amdkfd:kfd_queue_unmap
amdkfd:kfd_queue_evict
amdkfd:kfd_queue_restore
amdkfd:kfd_doorbell_assign
amdkfd:kfd_vm_fault
amdgpu:amdgpu_gpu_reset_begin
amdgpu:amdgpu_gpu_reset_end
amdgpu:amdgpu_pm_state_sample
```

Recommended payload:

- timestamp from kernel trace clock;
- PID, TGID, process name;
- PASID;
- DRM render node minor;
- GPU BDF or stable GPU UUID;
- queue ID and queue type;
- doorbell offset or ID;
- VMID where safe to expose;
- XCC/XCD/AID partition identifiers where available;
- reason code for eviction, restore, reset, or fault;
- status/result code.

Do not put user pointers, raw packet contents, kernel virtual addresses, or
payload data in tracepoints.

### 7.3 eBPF Join Keys

The trace joiner should map KFD/eBPF records to profiler dispatch records using:

- PID/TGID;
- PASID;
- GPU UUID or BDF;
- queue ID;
- timestamp window;
- optional doorbell ID.

It should not require exact queue ID equality if rocprofiler and KFD use
different queue namespaces on some ROCm/kernel combinations. The joiner must
store confidence levels for each association.

## 8. Additional Extensions Needed

### 8.1 CITL Metadata ABI

Add a small process-local ABI for registering compiler metadata:

```c
typedef struct {
  uint64_t citl_id;
  const char* graph_hash;
  const char* schedule_hash;
  const char* kernel_fingerprint;
  const char* op_name;
  const char* source_location;
} citl_profiler_metadata_t;

void citl_profiler_register_dispatch(const citl_profiler_metadata_t*);
```

The implementation can initially be a header-only shim over ROCTx and
rocprofiler external correlation IDs.

### 8.2 Trace Schema

Add trace event categories:

- `citl.compiler`: graph, schedule, tile, and target metadata;
- `citl.hip`: host API launch ranges;
- `citl.dispatch`: GPU kernel dispatch windows;
- `citl.counter`: rocprofiler counter samples;
- `citl.platform`: AMD SMI power, thermal, clock, voltage, and throttle samples;
- `citl.kfd`: KFD/eBPF queue, eviction, reset, and fault context;
- `citl.ras`: ECC and CPER/RAS events.

Every event must include:

- `trace_session_id`;
- `pid`;
- `gpu_uuid` or BDF;
- `timestamp_ns`;
- `clock_domain`;
- `source`;
- `confidence`.

Dispatch events must additionally include `citl_id`, `correlation_id`,
`dispatch_id`, `queue_id`, `kernel_fingerprint`, `graph_hash`, and
`schedule_hash`.

### 8.3 Clock and Timestamp Calibration

The profiler needs a clock calibration record at session start and periodically
afterward:

- host monotonic time;
- rocprofiler timestamp domain;
- AMD SMI firmware timestamp where exposed;
- kernel trace clock where eBPF is enabled.

The trace joiner should expose skew estimates and join confidence in the final
report.

### 8.4 Permissions Model

Modes:

- `user`: rocprofiler + ROCTx + AMD SMI reads only;
- `lab`: user mode plus high-rate sampling and debugfs reads;
- `kernel-context`: lab mode plus eBPF tracepoint/kprobe collection;
- `driver-dev`: kernel-context plus custom KFD/amdgpu tracepoints.

The default mode should be `user`.

## 9. Prototype Work Items

1. Add the CITL metadata shim and HIP launch wrapper.
2. Emit ROCTx ranges and rocprofiler external correlation IDs.
3. Build an AMD SMI sampler that records power, energy, clocks, temperatures,
   throttle status, voltage, ECC, and firmware timestamp.
4. Build a trace joiner that merges compiler metadata, rocprofiler records, AMD
   SMI samples, and optional eBPF records.
5. Export Perfetto/Chrome trace events using Tessera profiler conventions.
6. Add optional eBPF collection for KFD/amdgpu lifecycle events.
7. Add lab-only stream marker calibration using HIP stream write/wait values.
8. Define the first report: per-kernel duration, energy-window estimate,
   overlapping power/thermal/clock samples, throttle flags, and queue/reset/fault
   context.

## 10. Acceptance Criteria

- A fused Tessera/HIP kernel appears as one dispatch event with a stable CITL
  fingerprint and a rocprofiler correlation ID.
- AMD SMI samples appear on platform tracks and can be joined to dispatch
  windows with an explicit confidence score.
- Queue eviction, restore, VM fault, or reset events appear as KFD context when
  eBPF mode is enabled.
- The profiler can run without eBPF or driver changes.
- The report distinguishes measured values, sampled context, derived estimates,
  and unverified correlations.
- The trace opens in Perfetto or Chrome trace viewers.

## 11. References

- ROCprofiler-SDK overview and tracing capabilities:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-6.4.2/index.html>
- rocprofv3 tracing, kernel dispatch fields, and attach options:
  <https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html>
- ROCprofiler-SDK asynchronous tracing records:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.2.0/_doxygen/rocprofiler-sdk/html/group___b_u_f_f_e_r___t_r_a_c_i_n_g___s_e_r_v_i_c_e.html>
- ROCprofiler-SDK external correlation ID behavior:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-6.3.3/_doxygen/html/group___c_a_l_l_b_a_c_k___t_r_a_c_i_n_g___s_e_r_v_i_c_e.html>
- ROCprofiler-SDK PC sampling:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-6.4.2/how-to/using-pc-sampling.html>
- ROCr runtime API:
  <https://rocm.docs.amd.com/projects/ROCR-Runtime/en/develop/api-reference/api.html>
- HIP stream memory operations:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_m.html>
- AMD SMI CLI metrics:
  <https://rocmdocs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html>
- AMD SMI Python API metrics, clocks, temperature, voltage, and throttle fields:
  <https://rocm.docs.amd.com/projects/amdsmi/en/docs-7.0.1/reference/amdsmi-py-api.html>
- AMDGPU debugfs power-management context:
  <https://www.kernel.org/doc/html/latest/gpu/amdgpu/debugfs.html>
