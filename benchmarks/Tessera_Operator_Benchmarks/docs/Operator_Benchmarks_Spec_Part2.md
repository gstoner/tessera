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
- `artifact`: generated Tessera JIT bundles with Graph, Schedule, Tile, and
  Target artifacts present for every registered operator. Static MLIR samples
  remain readable examples, but validation uses generated bundles.
- `tessera-runtime --runtime bridge`: portable Python bridge that generates
  seeded NumPy inputs, compiles the operator with `@tessera.jit`, launches the
  generated `RuntimeArtifact`, and emits the shared JSON row schema.
- `tessera-runtime --runtime native`: future generated C ABI dispatch path;
  reports `backend_unavailable` until native hooks are built and detected.
- Optional vendor paths: CUTLASS/MIOpen/CK/Triton as future cross-checks.

## Current Gaps
- Native C ABI runtime launch hooks are still pending; the Python runtime bridge
  is the portable generated-artifact execution path for all registered ops.
- Artifact validation covers generated Graph/Schedule/Tile/Target bundles.
  Hardware validation of non-CPU target kernels remains out of scope for this
  hardware-free profile.
- Operator tests now lock bridge mappings, generated artifact coverage,
  telemetry schema, and independent NumPy conformance for the bridge. They do
  not replace future hardware runtime conformance suites.
--- END-MERGE: Operator_Benchmarks_Spec ---
