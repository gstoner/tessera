---
status: Informative
classification: Guide
authority: Profiler release-gate policy; defers native-availability semantics to the profiler provider-status/trace spine
last_updated: 2026-06-21
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
- ROCm: AMD GPU plus ROCprofiler-SDK context/tool registration, HIP/HSA callback
  records, and dispatch/activity correlation.
- NVIDIA: NVIDIA GPU plus CUPTI subscriber callback records, activity-buffer
  records, and kernel/memcpy/memset correlation.

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
