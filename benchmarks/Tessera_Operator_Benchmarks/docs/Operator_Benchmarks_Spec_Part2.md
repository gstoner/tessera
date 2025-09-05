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
- Provide `#ifdef OPBENCH_WITH_TESSERA` sections for real kernels.
- Optional vendor paths: cutlass/MIOpen/CK/Triton as cross-checks.
--- END-MERGE: Operator_Benchmarks_Spec ---
