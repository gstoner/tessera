--- BEGIN-MERGE: Operator_Benchmarks_Spec ---
# Tessera Operator Benchmarks — Spec (Part 2, Continuation)

## Configuration & Reproducibility
- YAML sweep configs with deterministic seeds; CSV output with schema
- Per-op argument schemas validated at parse-time
- Device metadata (when available): arch, driver, clock modes

## CI & Reporting
- Quick sweep on PR; weekly scheduled full sweep producing HTML+CSV artifacts
- Pluggable chart renderer; comparison across runners (future: multi-run aggregations)

## Backend Hooks
- `reference`: CPU reference kernels, executable everywhere.
- `artifact`: registered MLIR sample artifacts, reported as
  `runtime_status="artifact_only"` and `telemetry.status="unmeasured"`.
- `tessera-runtime`: explicit `backend_unavailable` status until generated
  operator runtime launches are wired to the Tessera C ABI.
- Optional vendor paths: CUTLASS/MIOpen/CK/Triton as future cross-checks.

## Current Gaps
- Runtime launch hooks are not implemented for generated Tessera kernels.
- Artifact samples currently stop at Graph IR contracts; Schedule/Tile/Target
  benchmark artifact validation should be added as those paths become stable for
  each operator.
- Operator tests lock registration, quick sweep coverage, artifact coverage, and
  telemetry schema. They do not replace numerical conformance suites.
--- END-MERGE: Operator_Benchmarks_Spec ---
