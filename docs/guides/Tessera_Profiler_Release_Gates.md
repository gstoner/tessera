---
status: Informative
classification: Guide
authority: Profiler release-gate policy; defers native-availability semantics to the profiler provider-status/trace spine
last_updated: 2026-08-06
---

# Tessera Profiler Release Gates

This guide defines the release gates for the profiler project. Native profiler availability is never inferred from mock data, replay fixtures, compile-only adapter shells, or provider status text alone.

## Default CI

The default CI lane must run without accelerator hardware or vendor SDKs:

- `tprof-context` mock and file collectors.
- `tprof-provider-status` for `apple`, `rocm`, `nvidia`, and `cpu`.
- `tprof-provider-trace` replay fixtures for ROCprofiler, CUPTI, and Metal.
- `tprof-merge-trace` runtime + provider + context fixtures.
- `tprof-report` HTML and summary JSON generation.
- Model Analyzer mock/estimated runner with profiler context, provider status,
  and merged-trace attachments.
- Compiler Target IR probe metadata tests.

## Optional SDK Build Gates

These lanes are allowed to soft-fail when the SDK is absent, but must not make
native availability claims unless the native proof lane also passes:

- macOS: `TPROF_WITH_METAL=ON`.
- ROCm: `TPROF_WITH_ROCPROFILER=ON`.
- NVIDIA: `TPROF_WITH_CUPTI=ON`.

Compiled adapter shells report `compiled_shell`. They are not
`native_available`.

## Hardware Proof Gates

Native availability requires a fresh hardware proof from the target machine:

- Apple: fresh-process, out-of-sandbox `tprof-apple-metal-smoke` proof with
  Metal device visibility and command-buffer or counter discovery evidence.
- ROCm: AMD GPU plus a fresh `rocprofv3`/ROCprofiler-SDK context, HIP/HSA
  callback records, dispatch/activity correlation, and any requested counter
  and PC-sampling records.
- NVIDIA: NVIDIA GPU plus CUPTI subscriber callback records, activity-buffer
  records, and kernel/memcpy/memset correlation.
- x86/AVX-512: exact CPU identity plus stable affinity, independent host/raw/TSC
  clock agreement, and a `perf_event_open` proof that reports permission,
  enabled/running time, multiplexing, and requested-versus-collected events.

Each native proof job should publish a provider availability snapshot artifact.
The snapshot must include provider, target, status, diagnostics, SDK/driver
versions when known, permission status when applicable, and dropped-record
counts when collection buffers are involved.

The optional `.github/workflows/profiler-native-proofs.yml` workflow is the
current proof scaffold. It is manual or label-gated and uploads one provider
status artifact per backend:

- `profiler-provider-status-apple` from `tprof_apple_metal_smoke.py`.
- `profiler-provider-status-rocm` from `tprof_rocm_native_smoke.py`.
- `profiler-provider-status-nvidia` from `tprof_nvidia_cupti_smoke.py`.

The ROCm and NVIDIA smoke scripts are safe on hosts without AMD/NVIDIA
hardware: they report `native_failed` diagnostics rather than promoting
availability. Apple remains `compiled_shell` until a fresh process proves Metal
visibility plus command-buffer timestamp or counter-set evidence.

### ROCm timing evidence gates

ROCm benchmark packets follow `TPROF-ROCM-TIME-1` in
[`CITL_ROCM_TRACE_PROFILER_SPEC.md`](../spec/CITL_ROCM_TRACE_PROFILER_SPEC.md).
Each sample carries independent `host_wall_ns`, `hip_event_ns`,
`device_wall_clock_ns`, and `profiler_activity_ns` records. An unavailable clock
is a valid schema state, but it cannot be replaced with a value from another
clock domain.

- Synchronized WSL host wall plus a qualified gfx1151 device-wall-clock
  envelope may support same-host regression and retain/reject decisions.
- Promotion requires bare-metal gfx1151 calibration against a valid HIP event
  or native ROCprofiler dispatch/activity interval.
- Counter-, PC-sampling-, and stall-dependent decisions require exact-device
  capability discovery and native records; an architecture name or metric
  catalog is not proof.
- Instrumented and uninstrumented artifacts must both be retained with ISA and
  resource deltas. Promotion also requires the application-kernel duration
  ratio to stay within the packet's declared overhead limit.

`tprof_rocm_native_capture.py` is the native boundary. It invokes `rocprofv3`
around the real application, preserves the exact command and output digests,
normalizes only records that were actually emitted, and never substitutes a
catalog lookup for dispatch, counter, or PC-sampling evidence. On WSL without
`/dev/kfd` it stops before invoking the known-aborting profiler and emits the
explicit `ROCPROFILER_DEVICE_INTERFACE_UNAVAILABLE` blocker.

An optional `rtg_hsa_dispatch` proof is a separate experimental provider. It
must run in a fresh process, report queue-interception overhead and teardown
status, and remain non-promotable by itself. It does not satisfy the ROCprofiler
native-availability gate and cannot claim counters or PC sampling.

### x86/AVX-512 timing and PMU gates

The generic `cpu` provider's `native_available` state proves portable runtime
callback spans and host timing only. It does not prove `TPROF-X86-TIME-1`, PMU
counters, instruction sampling, AMD IBS, Intel PEBS, or AVX-512 package
performance.

The native implementation is exposed by `tprof x86 timing-status`; the
optional workflow serializes it with
`tools/profiler/scripts/tprof_x86_native_smoke.py`. That snapshot becomes
`native_available` only when AVX-512, raw monotonic time, invariant fenced
RDTSCP, stable affinity, clock agreement, and a readable perf task-clock sample
all pass. A hosted runner without those capabilities emits `native_failed` and
the underlying permission/capability diagnostics without failing the optional
workflow.

- Host `steady_clock`, `CLOCK_MONOTONIC_RAW`, fenced `RDTSCP`, and perf task
  clock remain separate records with explicit calibration and validity.
- A migrated TSC sample is invalid. TSC evidence also requires invariant-TSC
  capability and a recorded calibrated frequency.
- Perf counters record permission state, event source/encoding, enabled time,
  running time, multiplexing ratio, and scaling. Unsupported or denied events
  are unavailable, not zero.
- Model-specific counters and sampling require exact vendor/family/model,
  microcode, kernel, event-map digest, and symbol/image correlation.
- `tprof_x86_event_map.py` digests the exact sysfs PMU encodings and `perf`
  catalog. `tprof_x86_sample.py` must embed that artifact and pin the sampled
  command to one logical CPU before a packet can be promotion-eligible.
- WSL timing can rank regressions on the same host. Counter- or sampling-based
  promotion requires a non-virtualized exact machine unless a future proof
  establishes equivalent PMU semantics through the virtualization layer.
- Perf samples must match the recorded image build ID and DSO-symbolized
  linkage name plus a unique nonzero static ELF symbol range. Raw runtime IPs
  are never compared directly with unrelocated ELF virtual addresses.
- AMD IBS is enabled only when the CPU is AMD family 26 with AVX-512 and the
  running kernel's `perf list` advertises the requested IBS event.
- A Zen 5 performance packet binds aligned and ragged results, source commit,
  dirty-worktree state, timing proof, and symbol-sampling proof. Any missing
  proof reduces the packet to retain/reject evidence.

## Claim Lint

Documentation, reports, and generated status tables must keep native provider
rows at `planned`, `compiled_shell`, `native_failed`, or `unavailable` until the
matching hardware proof snapshot exists. Mock, file, replay, and compile-only fixtures can demonstrate schema compatibility, but cannot promote availability.

## Troubleshooting

SDK discovery failures should surface as diagnostics, not process failures:

- Missing Metal framework or sandbox-hidden devices should be reported as Apple
  provider diagnostics and retried with the fresh-process proof path.
- Missing ROCprofiler-SDK headers/libraries should leave ROCm at `planned` or
  `compiled_shell`.
- Missing CUPTI/NVML libraries should leave NVIDIA at `planned` or
  `compiled_shell`.
- Permission, counter, or buffer exhaustion failures should include
  `error_type`, `error`, `permission_status`, or `dropped_records` fields where
  available.
