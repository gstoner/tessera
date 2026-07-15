---
status: Normative
classification: Architecture
authority: Defines Tessera compiler test proof layers, environment states, and migration rules
last_updated: 2026-07-15
---

# Compiler test architecture

Tessera tests are classified on two independent axes: the proof they provide
and the environment required to provide it. A directory name is not proof of
either. In particular, placement under `tests/unit/` must never make a native
GPU benchmark part of the hermetic CPU lane.

## Proof layers

| Layer | Required assertion | Forbidden shortcut |
|---|---|---|
| Semantic unit | Pure frontend, IR, selector, registry, diagnostic, or numerical-oracle behavior | Probing optional tools or hardware |
| Compiler artifact | Exact pass pipeline succeeds or fails with a named diagnostic; emitted IR/object properties are checked | Treating parser acceptance or source scanning as execution proof |
| Process integration | Packaging, import, crash isolation, runtime ABI, or multi-component behavior across a process boundary | Depending on the caller's incidental `PYTHONPATH` |
| Device correctness | Exact target dispatch plus execute/compare against the same oracle used by peer backends | Accepting reference fallback as native success |
| Measured performance | Repeated timing, committed baseline, resource counters, and an explicit regression rule | Wall-clock assertions in the required parallel CPU lane |
| Audit | Registry totality, lifecycle state, generated-document drift, and cross-surface consistency | Duplicating source-of-truth data in the test |

## Environment states

- Hermetic CPU tests are the required PR lane. They exclude `slow`,
  `performance`, and all `hardware_*` markers.
- Compiler-artifact tests carry `compiler_tool` and obtain tools through the
  session-scoped `compiler_toolchain` fixture. Missing tools have one canonical
  skip state.
- Child-process tests carry `integration` and use `python_subprocess_env`.
- Native tests carry exactly one target marker: `hardware_apple_gpu`,
  `hardware_nvidia`, or `hardware_rocm`.
- Any assertion based on elapsed time carries `performance` and runs serially
  in a benchmark/device lane, never under PR-lane xdist load.

## Test construction rules

1. Register every new dtype, operation, diagnostic, target, pass, and plan state
   before adding coverage; the registry test is part of the feature, not cleanup.
2. Prefer semantic assertions and named diagnostics over snapshots of large IR
   bodies. Use FileCheck for canonical C++ pass contracts.
3. Separate host-free selector/lowering tests from exact-device execute/compare
   tests even when they exercise the same feature.
4. A benchmark has a correctness oracle and kernel-only plus end-to-end timing.
   Resource evidence (VGPR/LDS/occupancy/spills where applicable) is stored with
   the baseline.
5. Tests must be order-independent. Shared runtime state is reset by fixtures;
   child processes inherit an explicit import environment.

## Migration sequence

The current flattened suite is migrated without a flag day:

1. Centralize tool discovery, child-process environment, and marker policy.
2. Remove measured and native tests from the hermetic PR selection.
3. Migrate compiler-path families from private path probes to shared fixtures.
4. Move stable families into `tests/compiler/`, `tests/integration/`,
   `tests/device/<target>/`, and `tests/performance/`; leave compatibility
   collection paths until their CI owners are active.
5. Replace filename allowlists in `tests/unit/conftest.py` with explicit markers.
6. Ratchet raw compiler subprocesses onto the process-group-safe runner and
   delete obsolete or duplicate fixtures only after equivalent proof is mapped.

The machine-enforced rules live in
`tests/unit/test_test_suite_architecture.py`; the command map lives in
`docs/spec/VALIDATION_SPINE.md`.
