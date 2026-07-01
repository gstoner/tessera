"""Drift gate + structural guards for the Apple target map dashboard
(Apple plan phase A, 2026-05-20).

The dashboard at ``docs/audit/generated/apple_target_map.md`` is
regenerated from ``capabilities.py`` + ``backend_manifest.py`` + the
driver-side dispatch routes.  These tests:

1. Re-render and assert the on-disk doc matches (CI drift gate).
2. Lock the per-family row counts the dashboard exposes.
3. Verify every shipped Apple GPU MSL kernel in
   ``_APPLE_GPU_KERNELS`` has a symbol entry in the dispatch map
   (so a new kernel forces a dashboard update).
4. Verify the ``proof_test`` columns point at files that actually
   exist on disk.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATED_DOC = ROOT / "docs" / "audit" / "generated" / "apple_target_map.md"


def test_apple_target_map_doc_is_current() -> None:
    """The committed doc must match what the renderer produces."""

    from tessera.compiler.apple_target_map import render_markdown

    assert GENERATED_DOC.exists(), (
        "Apple target map missing; run "
        "`python -m tessera.cli.apple_target_map --render`."
    )
    on_disk = GENERATED_DOC.read_text(encoding="utf-8")
    rendered = render_markdown()
    assert on_disk == rendered, (
        "apple_target_map.md is out of date — re-run "
        "`python -m tessera.cli.apple_target_map --render` after "
        "any change to capabilities.py / backend_manifest.py / driver.py."
    )


def test_per_family_row_counts_match_backend_manifest() -> None:
    """Per-family counts must match the upstream tables exactly so a
    new fused kernel automatically widens the dashboard."""

    from tessera.compiler.apple_target_map import all_rows
    from tessera.compiler import backend_manifest as bm

    rows = all_rows()
    by_family: dict[str, list] = {}
    for r in rows:
        by_family.setdefault(r.family, []).append(r)

    # GA row count == _CLIFFORD_APPLE_GPU_FUSED size
    assert len(by_family["ga"]) == len(bm._CLIFFORD_APPLE_GPU_FUSED)
    # EBM row count == _EBM_APPLE_GPU_FUSED size
    assert len(by_family["ebm"]) == len(bm._EBM_APPLE_GPU_FUSED)
    # M7 row count == _COMPLEX_APPLE_GPU_FUSED size
    assert len(by_family["m7"]) == len(bm._COMPLEX_APPLE_GPU_FUSED)
    # Generic tensor: union of _APPLE_CPU_KERNELS + _APPLE_GPU_KERNELS
    tensor_keys = set(bm._APPLE_CPU_KERNELS) | set(bm._APPLE_GPU_KERNELS)
    assert len(by_family["tensor"]) == len(tensor_keys)


def test_every_apple_gpu_msl_kernel_has_dispatch_symbol() -> None:
    """If a new driver-backed MSL kernel lands in ``_APPLE_GPU_KERNELS`` but
    the dashboard's symbol map doesn't grow to match, the dashboard would
    silently render an empty symbol.  Manifest/composed tensor rows can be
    symbol-less because they are routed through a higher-level lane.
    """

    from tessera.compiler.apple_target_map import (
        _APPLE_GPU_KERNELS_SYMBOL_MAP,
        _DRIVER_DISPATCH_OPS,
    )
    from tessera.compiler import backend_manifest as bm

    driver_backed = set(bm._APPLE_GPU_KERNELS) & set(_DRIVER_DISPATCH_OPS)
    missing = sorted(driver_backed - set(_APPLE_GPU_KERNELS_SYMBOL_MAP))
    assert not missing, (
        f"new Apple GPU kernels {missing!r} lack a dispatch symbol "
        "entry in apple_target_map._APPLE_GPU_KERNELS_SYMBOL_MAP — "
        "add the runtime symbol you'd see in `driver.py::_backend_artifact_for`."
    )


def test_apple_cpu_execution_kind_axis_is_explicit_per_op() -> None:
    """Apple CPU execution_kind must not silently claim
    ``accelerate_native`` for every op.  Only ops in the explicit
    table get the native label; everything else is
    ``numpy_reference``.
    """

    from tessera.compiler.apple_target_map import (
        _APPLE_CPU_EXECUTION_KIND, all_rows,
    )

    # Only the two matmul-family aliases have the native label today.
    assert _APPLE_CPU_EXECUTION_KIND == {
        "matmul": "accelerate_native",
        "gemm":   "accelerate_native",
    }, (
        "execution_kind table changed unexpectedly — adding entries "
        "must come with an Apple CPU microbench proving the native "
        "fast-path actually runs (Apple plan, phase B)."
    )

    # No tensor-family row should silently report accelerate_native
    # without being in the table.
    for row in all_rows():
        if row.cpu_execution_kind == "accelerate_native":
            assert row.op_name in _APPLE_CPU_EXECUTION_KIND, (
                f"{row.op_name} reports accelerate_native but isn't "
                "in the explicit execution_kind table"
            )


def test_proof_tests_point_at_real_files() -> None:
    """Every ``proof_test`` cell must resolve to a file (or a glob
    that matches at least one file) so the dashboard never claims a
    test that doesn't exist."""

    from tessera.compiler.apple_target_map import all_rows

    missing: list[str] = []
    for row in all_rows():
        path = row.proof_test
        if not path or path == "-":
            continue
        # Glob handling — treat ``*`` and ``?`` as wildcards.
        if any(ch in path for ch in "*?"):
            matches = list((ROOT).glob(path))
            if not matches:
                missing.append(f"{row.op_name}: {path}")
        else:
            candidate = ROOT / path
            if not candidate.exists():
                missing.append(f"{row.op_name}: {path}")
    assert not missing, (
        "apple_target_map proof_test cells point at missing files:\n"
        + "\n".join(missing)
    )


def test_dispatch_column_routes_match_real_dispatch() -> None:
    """The ``gpu_dispatch`` column must say ``driver`` for ops the
    generic driver routes (matmul / softmax / gelu / rope /
    flash_attn / rmsnorm) and ``manifest`` for GA / EBM / M7 plus tensor
    composite-helper lanes."""

    from tessera.compiler.apple_target_map import _DRIVER_DISPATCH_OPS, all_rows

    for row in all_rows():
        if row.gpu_status == "absent":
            assert row.gpu_dispatch == "absent"
            continue
        if row.family in ("ga", "ebm", "m7"):
            assert row.gpu_dispatch == "manifest", row
        elif row.family == "tensor":
            expected = "driver" if row.op_name in _DRIVER_DISPATCH_OPS else "manifest"
            assert row.gpu_dispatch == expected, row
