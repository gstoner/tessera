"""G4 — single-source runtime execution matrix drift guard.

`tessera.compiler.execution_matrix` is the one place `(target, compiler_path) ->
ExecutionRow` lives. `runtime.launch()` consults it (instead of hard-coding
non-CPU branches); `capabilities.py` shares its `TARGET_CAPABILITIES`
vocabulary; the dashboard at `docs/audit/generated/runtime_execution_matrix.md`
renders it. This test fails if any of those four agree on a different answer —
the "single source of truth" invariant the compiler-layer remediation plan calls
out (G4, `docs/audit/compiler_layer_gap_remediation.md`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler import execution_matrix as EM
from tessera.compiler.capabilities import TARGET_CAPABILITIES

DASHBOARD = (Path(__file__).resolve().parents[2] / "docs" / "audit"
             / "generated" / "runtime_execution_matrix.md")


# --- Matrix-level invariants -------------------------------------------------

def test_validate_against_capabilities_passes():
    """Every executable row's target + every unimplemented target must be in the
    capability registry; every executor_id must be in KNOWN_EXECUTORS."""
    errors = EM.validate_against_capabilities()
    assert not errors, "matrix↔capabilities cross-check failed:\n  " + "\n  ".join(errors)


def test_every_known_executor_is_used_by_some_row():
    """A KNOWN_EXECUTORS entry with no row using it is dead documentation."""
    used = {r.executor_id for r in EM.all_rows() if r.executor_id}
    dead = set(EM.KNOWN_EXECUTORS) - used
    assert not dead, f"KNOWN_EXECUTORS not used by any row: {sorted(dead)}"


def test_executable_rows_have_an_executor_id():
    """`executable=True` rows must name a real executor; `executable=False` rows
    must not (caught by the launch dispatcher otherwise)."""
    for row in EM.all_rows():
        if row.executable:
            assert row.executor_id is not None, f"executable row missing executor_id: {row}"
        else:
            assert row.executor_id is None, f"non-executable row has executor_id: {row}"


def test_unimplemented_targets_disjoint_from_executable_rows():
    """A target can't be both 'has an executor' and 'on the unimplemented list'."""
    executable_targets = {r.target for r in EM.all_rows()}
    unimplemented = set(EM.unimplemented_targets())
    overlap = executable_targets & unimplemented
    assert not overlap, f"targets in both executable rows and unimplemented list: {sorted(overlap)}"


# --- Lookup semantics --------------------------------------------------------

@pytest.mark.parametrize("target,compiler_path,expect_executor", [
    ("apple_cpu", "apple_cpu_accelerate", "apple_cpu_accelerate"),
    ("apple_gpu", "apple_gpu_mps",        "apple_gpu_mps"),
    ("cpu",       "native_cpu",           "native_cpu"),
    ("cpu",       "jit_cpu_numpy",        "jit_cpu_numpy"),
])
def test_lookup_returns_expected_executor(target, compiler_path, expect_executor):
    row = EM.lookup(target, compiler_path)
    assert row is not None and row.executable is True
    assert row.executor_id == expect_executor


def test_lookup_returns_none_for_unimplemented_target():
    # NVIDIA + the artifact_only path is intentionally not in the matrix.
    assert EM.lookup("nvidia_sm90", "artifact_only") is None


def test_executor_for_metadata_resolves_via_target_and_compiler_path():
    row = EM.executor_for_metadata({"target": "apple_gpu", "compiler_path": "apple_gpu_mps"})
    assert row is not None and row.executor_id == "apple_gpu_mps"


def test_executor_for_metadata_falls_through_without_compiler_path():
    # Legacy artifacts without compiler_path: matrix returns None so launch()
    # falls through to its historical executable+execution_kind logic.
    assert EM.executor_for_metadata({"target": "cpu"}) is None


# --- Dashboard drift guard ---------------------------------------------------

def test_dashboard_exists():
    assert DASHBOARD.exists(), (
        f"missing {DASHBOARD}. Regenerate with: "
        "python3 -c 'from tessera.compiler.execution_matrix import write_dashboard; write_dashboard()'")


def test_dashboard_matches_live_data():
    """The on-disk dashboard must equal what render_dashboard() produces from
    the current matrix. Any drift means someone changed the matrix without
    regenerating, or hand-edited the generated file."""
    live = EM.render_dashboard()
    disk = DASHBOARD.read_text()
    if live != disk:
        # Surface a useful diff hint in the failure message.
        live_lines = live.splitlines()
        disk_lines = disk.splitlines()
        first_diff = next((i for i in range(min(len(live_lines), len(disk_lines)))
                           if live_lines[i] != disk_lines[i]), min(len(live_lines), len(disk_lines)))
        pytest.fail(
            f"Runtime execution matrix dashboard drift at line {first_diff + 1}: "
            f"on-disk has {disk_lines[first_diff] if first_diff < len(disk_lines) else '<EOF>'!r}, "
            f"live has {live_lines[first_diff] if first_diff < len(live_lines) else '<EOF>'!r}. "
            "Regenerate with `python3 -c 'from tessera.compiler.execution_matrix "
            "import write_dashboard; write_dashboard()'`.")


# --- runtime.launch() consults the matrix (single-source invariant) ----------

def test_runtime_launch_imports_the_matrix():
    """`runtime.launch()` must consult `execution_matrix` (not just import it
    decoratively), so adding a new backend executor flows through this file.
    Asserted by source inspection — any rewrite that drops the matrix consult
    must update this assertion (forcing a deliberate change)."""
    import inspect
    from tessera import runtime as R
    src = inspect.getsource(R)
    assert "from .compiler.execution_matrix" in src, (
        "runtime.py must import execution_matrix (the single source of truth "
        "for runtime.launch's executor dispatch); see G4 in "
        "docs/audit/compiler_layer_gap_remediation.md")


def test_runtime_launch_unimplemented_branch_cites_the_matrix():
    """The non-CPU unimplemented fallthrough should point readers to the matrix
    dashboard in its reason, so the runtime + docs are linked."""
    import inspect
    from tessera import runtime as R
    src = inspect.getsource(R.launch)
    assert "runtime_execution_matrix.md" in src
