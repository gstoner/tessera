"""Drift gates for NVIDIA + ROCm target maps
(Apple follow-up #3, 2026-05-20).

Each generated ``<target>_target_map.md`` must match the live
``render_markdown(target)`` output.  Today both dashboards are
populated from artifact-only capability rows — when Phase G/H
hardware bring-up moves a row to ``compileable`` / ``executable`` /
``fused``, the dashboard auto-promotes; this drift gate catches the
case where the manifest changes but the on-disk doc didn't get
regenerated.
"""
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "docs" / "audit" / "generated"


# Targets we ship a dashboard for today.  Only the canonical
# representatives (one per family) to keep CI cost flat; the CLI
# accepts more (every sm_* / gfx_* arch) but those are opt-in.
_DEFAULT_TARGETS = ("nvidia_sm90", "rocm")


@pytest.mark.parametrize("target", _DEFAULT_TARGETS)
def test_gpu_target_map_doc_is_current(target: str) -> None:
    from tessera.compiler.gpu_target_map import render_markdown

    doc = AUDIT_DIR / f"{target}_target_map.md"
    assert doc.exists(), (
        f"{doc.relative_to(ROOT)} missing — run "
        f"`python -m tessera.cli.gpu_target_map --target={target} "
        f"--render` to create it."
    )
    on_disk = doc.read_text(encoding="utf-8")
    rendered = render_markdown(target)
    assert on_disk == rendered, (
        f"{doc.relative_to(ROOT)} is stale — re-run the renderer.  "
        "When manifest entries change (e.g., a row flips from "
        "artifact_only → compileable), the dashboard must follow."
    )


@pytest.mark.parametrize("target", _DEFAULT_TARGETS)
def test_gpu_target_map_has_non_empty_row_count(target: str) -> None:
    """Sanity: each dashboard has at least 20 rows (the planned
    fused-kernel inventory).  A renderer regression that returns
    no rows shouldn't pass CI silently."""
    from tessera.compiler.gpu_target_map import (
        all_nvidia_rows, all_rocm_rows,
    )
    rows = all_nvidia_rows(target) if target.startswith("nvidia") else all_rocm_rows(target)
    assert len(rows) >= 20, (
        f"{target} dashboard has only {len(rows)} rows — expected "
        f"≥20 from ``backend_manifest._{target.split('_')[0].upper()}_ARTIFACT``"
    )


def test_rocm_map_uses_exact_architecture_grain_and_provenance() -> None:
    from tessera.compiler.gpu_target_map import all_rocm_exact_rows

    rows = all_rocm_exact_rows()
    assert rows
    assert all(r.target != "rocm" for r in rows)
    assert all(r.target.startswith("rocm_gfx") for r in rows)
    for row in rows:
        if row.runtime_execute == "complete" or row.numerical_check == "complete":
            assert row.target == "rocm_gfx1151"
            assert row.evidence_arch == "gfx1151"
        if row.status in {"device_verified_jit", "device_verified_abi"}:
            assert row.compiler_path
            assert row.runtime_execute == "complete"
            assert row.numerical_check == "complete"


def test_rocm_priority_targets_do_not_inherit_gfx1151_proof() -> None:
    from tessera.compiler.gpu_target_map import all_rocm_exact_rows

    rows = all_rocm_exact_rows()
    products = {(r.target, r.products) for r in rows}
    assert ("rocm_gfx950", "MI350X / MI355X / MI350P") in products
    assert ("rocm_gfx1250", "MI455X") in products
    assert ("rocm_gfx1201", "Radeon AI PRO R9700") in products
    for row in rows:
        if row.target != "rocm_gfx1151":
            assert row.status not in {"device_verified_jit", "device_verified_abi"}
            assert row.runtime_execute == "missing"
            assert row.numerical_check == "missing"
            assert not row.evidence_arch
            assert row.expected_mfu is None


def test_rocm_csv_separates_isa_toolchain_and_evidence_arch() -> None:
    import csv
    import io
    from tessera.compiler.gpu_target_map import render_csv

    rows = list(csv.DictReader(io.StringIO(render_csv("rocm"))))
    assert rows
    assert {"target", "isa", "toolchain_min", "evidence_arch",
            "runtime_execute", "numerical_check"} <= set(rows[0])
    assert "arch_min" not in rows[0]


def test_apple_and_gpu_dashboards_share_status_vocabulary() -> None:
    """The Apple and GPU dashboards must use the same status strings
    so reviewers see a uniform vocabulary across targets."""
    from tessera.compiler import apple_target_map, gpu_target_map

    apple_statuses = {r.gpu_status for r in apple_target_map.all_rows()}
    # NVIDIA + ROCm statuses are pulled from the same set.
    gpu_statuses = (
        {r.status for r in gpu_target_map.all_nvidia_rows("nvidia_sm90")}
        | {r.status for r in gpu_target_map.all_rocm_rows("rocm")}
    )
    # ``device_verified_abi`` is the top rung of the readiness ladder
    # (Project 3, 2026-06-01) — fused + a checked-in numerical-proof
    # fixture. It surfaces on the Apple dashboard for the promoted
    # encode-session ops (softmax/rmsnorm/.../conv2d).
    # ``device_verified_jit`` (2026-06-25) is the rung just below it: executes on
    # hardware via runtime.launch() as a compiler-generated hsaco + a
    # numerical fixture, but with no shipped C-ABI runtime_symbol
    # (the ROCm compiler-generated attention/epilogue family).
    allowed = {"fused", "compileable", "executable", "artifact_only",
               "reference", "planned", "absent", "unknown", "ready",
               "device_verified_abi", "device_verified_jit"}
    assert apple_statuses <= allowed, (
        f"Apple dashboard uses unknown status(es): "
        f"{apple_statuses - allowed!r}"
    )
    assert gpu_statuses <= allowed, (
        f"GPU dashboard uses unknown status(es): "
        f"{gpu_statuses - allowed!r}"
    )
