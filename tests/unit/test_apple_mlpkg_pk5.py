"""PK5 — compiler integration for packaged ML kernels.

Fifth step of the packaged-kernel sprint. Extends
``BackendKernelEntry`` with:

* ``_PACKAGED_STATUS = "packaged"`` — new status value (distinct from
  ``fused`` / ``reference`` / ``artifact_only``).
* ``packaged_pipeline_path`` — required path field naming the
  ``.mtlpackage`` directory the runtime loads at dispatch time.
* ``__post_init__`` validator: status="packaged" without a path
  raises ``ValueError`` so a registry entry can't promise a packaged
  kernel without a deliverable.
* ``as_dict`` surfaces the new field when non-None.

These tests pin the data-model contract. Wiring `.mtlpackage` paths
into actual per-op rows is a follow-up (the dashboard would need an
``apple_gpu_packaged`` indicator) — PK5 just lands the scaffolding so
the rest of the sprint has a stable type to extend.
"""

from __future__ import annotations

import pytest

from tessera.compiler.backend_manifest import (
    _PACKAGED_STATUS,
    _VALID_STATUSES,
    BackendKernelEntry,
)


# ---- Status enum --------------------------------------------------------

def test_packaged_status_is_in_valid_set():
    """The new status value must be accepted by the enum check —
    otherwise constructing a packaged entry would raise."""
    assert _PACKAGED_STATUS == "packaged"
    assert _PACKAGED_STATUS in _VALID_STATUSES


def test_existing_statuses_unchanged():
    """Adding a new status must not break the existing ones. Regression
    guard against a future enum-refactor that drops a status."""
    for s in ("fused", "reference", "artifact_only",
              "compileable", "planned", "hardware_verified"):
        assert s in _VALID_STATUSES, f"existing status {s!r} disappeared"


# ---- Packaged entry construction ---------------------------------------

def test_packaged_entry_requires_pipeline_path():
    """``status='packaged'`` without ``packaged_pipeline_path`` must
    raise a precise ValueError naming the target."""
    with pytest.raises(ValueError, match="packaged_pipeline_path"):
        BackendKernelEntry(
            target="apple_gpu",
            status="packaged",
            dtypes=("fp32",),
        )


def test_packaged_entry_with_path_constructs_clean():
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        packaged_pipeline_path="tests/fixtures/apple_gpu/matrix-multiplication.mtlpackage",
        notes="Apple sample matmul package — PK5 reference",
    )
    assert entry.status == "packaged"
    assert entry.packaged_pipeline_path == (
        "tests/fixtures/apple_gpu/matrix-multiplication.mtlpackage")


def test_packaged_path_is_optional_for_non_packaged_status():
    """Other status values must NOT require the field — only
    ``packaged`` does. Regression guard against making the field
    accidentally mandatory for every row."""
    BackendKernelEntry(target="apple_gpu", status="fused", dtypes=("fp32",))
    BackendKernelEntry(target="apple_cpu", status="reference",
                       dtypes=("fp32",))
    BackendKernelEntry(target="nvidia_sm90", status="artifact_only",
                       dtypes=("fp32",))
    # All construct successfully without packaged_pipeline_path.


def test_empty_path_is_treated_as_missing():
    """Empty string is not a valid path — the validator must catch it
    along with None."""
    with pytest.raises(ValueError, match="packaged_pipeline_path"):
        BackendKernelEntry(
            target="apple_gpu", status="packaged", dtypes=("fp32",),
            packaged_pipeline_path="",
        )


# ---- as_dict round-trip -------------------------------------------------

def test_as_dict_omits_path_for_non_packaged():
    """The field is suppressed when None — keeps the JSON compact
    (the pattern from Sprint G-3 fields)."""
    entry = BackendKernelEntry(
        target="apple_gpu", status="fused", dtypes=("fp32",))
    d = entry.as_dict()
    assert "packaged_pipeline_path" not in d


def test_as_dict_includes_path_for_packaged():
    entry = BackendKernelEntry(
        target="apple_gpu", status="packaged", dtypes=("fp32",),
        packaged_pipeline_path="some/path.mtlpackage",
    )
    d = entry.as_dict()
    assert d["packaged_pipeline_path"] == "some/path.mtlpackage"
    assert d["status"] == "packaged"


# ---- Interop with other dataclass fields --------------------------------

def test_packaged_status_does_not_block_other_optional_fields():
    """A packaged entry can also carry ``execute_compare_fixture`` and
    ``benchmark_json`` (no conflict between the audit-Action-4 proof
    and the packaged-status flag)."""
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        packaged_pipeline_path="some/path.mtlpackage",
        runtime_symbol="tessera_apple_gpu_mlpkg_dispatch",
        execute_compare_fixture="tests/unit/test_apple_mlpkg_pk4.py",
        benchmark_json="benchmarks/apple_gpu/mlpkg_smoke.json",
    )
    d = entry.as_dict()
    assert d["packaged_pipeline_path"] == "some/path.mtlpackage"
    assert d["runtime_symbol"] == "tessera_apple_gpu_mlpkg_dispatch"
    assert d["execute_compare_fixture"] == (
        "tests/unit/test_apple_mlpkg_pk4.py")
    assert d["benchmark_json"] == "benchmarks/apple_gpu/mlpkg_smoke.json"


def test_packaged_entry_target_is_freeform():
    """The field doesn't constrain target — packaged kernels are
    currently Apple-specific by convention but the dataclass is
    target-agnostic so future ROCm / NVIDIA packaged formats wouldn't
    need a schema change."""
    BackendKernelEntry(
        target="apple_gpu", status="packaged", dtypes=("fp32",),
        packaged_pipeline_path="path/to/apple.mtlpackage",
    )
    # Hypothetical future:
    BackendKernelEntry(
        target="apple_cpu", status="packaged", dtypes=("fp32",),
        packaged_pipeline_path="path/to/cpu.mtlpackage",
    )
