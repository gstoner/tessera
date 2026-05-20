# Apple release gate — policy

> **Locked in 2026-05-20 (Apple follow-up #4).** This document is
> the canonical policy that every release claiming Apple support
> must satisfy.  The runnable enforcement lives in
> `scripts/release_gate.py`.

## Policy

> **Any release of Tessera that publicly claims Apple GPU (or
> Apple CPU) support MUST pass
> `python scripts/release_gate.py --target=apple_gpu` cleanly,
> with verdict "all 8 gates passed", on at least one Darwin host
> within 24 hours of the release tag.**

No exceptions for:

* documentation-only releases (Apple claims in docs are still
  claims; the gate verifies the underlying artifacts are still in
  place).
* hotfix releases (the gate runs in <30 seconds; there is no
  meaningful cost saving from skipping it).
* prerelease tags / RCs / nightlies (especially these — the gate
  is exactly the kind of check that should run on every nightly
  before promotion).

## What the gate verifies (8 gates today)

| # | Gate | What fails if a release ships broken |
|---|---|---|
| 1 | `support_table_drift` | Generated support_table.md is stale vs `OP_SPECS` + `capabilities` + `backend_manifest`. |
| 2 | `e2e_coverage_drift` | Generated e2e_op_coverage.md is stale vs the audit walker. |
| 3 | `tests_manifest_drift` | Generated tests_status.md is stale vs `tests_manifest.py`. |
| 4 | `apple_target_map_drift` | Generated apple_target_map.md is stale vs the capability / manifest / driver data. |
| 5 | `canonicals_native_dispatch` | `matmul_softmax_matmul` canonical doesn't actually dispatch its fused MSL kernel on Darwin (regression in driver, runtime shim, or canonical wiring). |
| 6 | `apple_cpu_execution_kind_bench` | matmul-vs-numpy ratio gate breaks (Tessera left the Accelerate fast lane). |
| 7 | `spectral_correctness_proof` | Stockham FFT correctness sentinel disagrees with naive DFT beyond 1e-3 tolerance. |
| 8 | `apple_gpu_hardware_marked_tests` | Any `@pytest.mark.hardware_apple_gpu` test fails on the Darwin runner. |

The gate runs in **under 30 seconds** on Apple Silicon when the
build artifacts are warm.  Cold runs (including the C++ spectral
correctness binary build) add ~10 seconds.

## What the gate does NOT verify

The gate is an *external-release* blocker, not an internal
correctness oracle.  These are intentionally out of scope:

* **End-to-end model training / inference correctness** — covered
  by the Phase D + reasoning-model tests in `tests/unit/`, not by
  the gate.  Promote those tests to the
  `@pytest.mark.hardware_apple_gpu` lane when they earn it.
* **Performance regression** — the gate uses ratio-based smoke
  bounds (3× tolerance, 1e-3 correctness), not perf-gate bounds.
  Perf gates live in `tests/performance/` and run on a separate
  cadence.
* **Documentation prose accuracy** — `claim_lint` catches overclaim
  language; the release gate trusts that the docs match the
  manifests because the manifest drift gates fail otherwise.

## How releases run the gate

```bash
# From the repo root on a Darwin machine.
python scripts/release_gate.py --target=apple_gpu

# Expected exit code: 0
# Expected last line: "[release_gate:apple_gpu] all 8 gates passed"
```

If the gate exits non-zero:

1. **Do not tag the release.** Roll the version back to the last
   passing commit.
2. Read the gate's stderr — each failure includes the failing
   gate's name + captured output tail.
3. Fix the root cause (almost always a manifest / dashboard /
   canonical drift) and re-run the gate before tagging.

## Future: NVIDIA and ROCm release gates

When Phase G (NVIDIA H100) and Phase H (ROCm MI300X) move from
artifact-only to hardware-proof, the parallel gates
(`--target=nvidia_sm90`, `--target=rocm`) become external-release
blockers for those target claims.  Today they ship with structural
gates only — enough to flag a stale `nvidia_target_map.md` /
`rocm_target_map.md`, but not hardware-proof.  See
`scripts/release_gate.py::_GATE_MATRIX` for the structural set; the
hardware lanes are append-only additions to the matrix when the
underlying capability rows light up.

## Audit trail

| Change | Date | Effect |
|---|---|---|
| Policy created | 2026-05-20 | Apple follow-up #4 — `--target=apple_gpu` becomes the canonical Apple release blocker.  Gate matrix: 8 gates (structure + Apple-specific). |
| `matmul_softmax_matmul` promoted to native dispatch | 2026-05-20 | Apple follow-up #5 / phase E.  Removes a `REFERENCE_FORCED` from the happy path; canonical now exercises the real fused MSL kernel. |
| `matmul_gelu` canonical added | 2026-05-20 | Apple follow-up #1.  Second generic-tensor canonical to dispatch through the unified proof envelope. |
| `visual_complex_fused` canonical added | 2026-05-20 | Apple follow-up #2.  Visual Complex (M7) surface — 4 fused kernels in one report.  Required the `complex.stereographic` dtype-preservation fix. |
