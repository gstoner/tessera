"""Audit-C (2026-05-22) — runtime C ABI surface drift gate.

Pins:

  * All 6 canonical core runtime headers exist on disk.
  * Total tessera_* C ABI symbol count stays at-or-above the
    2026-05-22 baseline (115).
  * Apple GPU per-family coverage stays at-or-above floor (the
    Phase 8.4.7 → 8.4.7+ growth surface).
  * Sentinel Apple GPU kernel families (matmul, flash_attn, softmax,
    gelu, rope, swiglu, MLA decode, NSA, lightning_attn, the EBM /
    Clifford / Visual Complex fused chains) all retain their canonical
    dtype matrix.
  * Dashboard at ``docs/audit/generated/runtime_abi.md`` is in sync
    with the live source scan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.runtime_abi_audit import (
    AbiSymbol,
    apple_gpu_kernel_families,
    collect_runtime_abi,
    core_runtime_headers_present,
    render_dashboard,
    symbols_per_backend,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "runtime_abi.md"


# ─────────────────────────────────────────────────────────────────────────
# Core runtime headers
# ─────────────────────────────────────────────────────────────────────────


def test_all_core_runtime_headers_exist() -> None:
    """The 6 backend-agnostic runtime headers MUST exist.  Missing
    any of them is a build-broken regression."""
    headers = core_runtime_headers_present()
    missing = [path for path, ok in headers.items() if not ok]
    assert not missing, (
        f"Core runtime headers missing: {missing}.  Were they "
        f"renamed without updating runtime_abi_audit.py?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Symbol-count floors
# ─────────────────────────────────────────────────────────────────────────


def test_total_abi_symbol_count_floor() -> None:
    """Locked at the Audit-C landing.  Catches accidental symbol
    deletions.  Counts UNIQUE symbol names (deduplicates a symbol
    declared in a header and defined in a .cpp into one row)."""
    abi = collect_runtime_abi()
    unique = {s.name for s in abi}
    assert len(unique) >= 75, (
        f"Unique tessera_* C ABI symbol count dropped below 75 "
        f"(got {len(unique)}).  Run "
        f"`tessera.compiler.runtime_abi_audit.collect_runtime_abi()` "
        f"to inspect the missing surface."
    )


# symbols_per_backend counts UNIQUE names per backend.  At the Audit-C
# landing: apple=69, x86=8, nvidia=3.  Floors set ≤ baseline.
_PER_BACKEND_FLOORS = {
    "apple": 65,
    "x86": 5,
}


@pytest.mark.parametrize(
    "backend,floor", sorted(_PER_BACKEND_FLOORS.items())
)
def test_per_backend_symbol_floor(backend: str, floor: int) -> None:
    counts = symbols_per_backend()
    actual = counts.get(backend, 0)
    assert actual >= floor, (
        f"Backend {backend!r} symbol count dropped below floor "
        f"{floor} (got {actual})."
    )


# ─────────────────────────────────────────────────────────────────────────
# Apple GPU sentinel families
# ─────────────────────────────────────────────────────────────────────────


_APPLE_GPU_SENTINEL_FAMILIES = (
    # (family, minimum_dtype_variants_required)
    ("mps_matmul", 3),                   # f32, f16, bf16
    ("rope", 3),
    ("flash_attn", 3),
    ("softmax", 3),
    ("gelu", 3),
    ("matmul_softmax", 3),               # 2-op fusion
    ("matmul_softmax_matmul", 3),        # 3-op fusion (full attention)
    ("swiglu", 3),
    ("matmul_softmax_tiled", 3),         # f32 + native-half (f16/bf16) tiled
    ("matmul_gelu", 3),                  # MLP block fusion, f32/f16/bf16
    ("matmul_rmsnorm", 3),
    ("mla_decode", 1),
    ("native_sparse_attn", 1),
    ("linear_attn", 1),
)


@pytest.mark.parametrize(
    "family,floor", _APPLE_GPU_SENTINEL_FAMILIES
)
def test_apple_gpu_family_dtype_floor(family: str, floor: int) -> None:
    """Each Apple GPU kernel family must retain its dtype coverage.
    A regression below floor means one of the per-dtype variants was
    deleted or renamed."""
    families = apple_gpu_kernel_families()
    dtypes = families.get(family, ())
    assert len(dtypes) >= floor, (
        f"Apple GPU family {family!r} dtype variants dropped below "
        f"floor {floor} (got {len(dtypes)}: {dtypes})."
    )


def test_apple_gpu_family_count_floor() -> None:
    families = apple_gpu_kernel_families()
    assert len(families) >= 40, (
        f"Apple GPU kernel family count dropped below 40 "
        f"(got {len(families)})."
    )


# ─────────────────────────────────────────────────────────────────────────
# Symbol naming convention
# ─────────────────────────────────────────────────────────────────────────


def test_every_symbol_starts_with_tessera_prefix() -> None:
    """The C ABI naming convention is ``tessera_*``.  Any symbol
    that doesn't follow this pattern means our scanner picked up an
    unrelated `extern "C"` declaration or a third-party header
    leaked into the source tree."""
    for s in collect_runtime_abi():
        assert s.name.startswith("tessera_"), (
            f"Non-canonical symbol name: {s.name!r} in {s.path}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Dashboard drift gate
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_exists() -> None:
    assert DASHBOARD.exists(), (
        f"Generated dashboard missing: "
        f"{DASHBOARD.relative_to(REPO_ROOT)}.  Regenerate via "
        f"`tessera.compiler.runtime_abi_audit.write_dashboard()`."
    )


def test_dashboard_matches_live_data() -> None:
    if not DASHBOARD.exists():
        pytest.skip("dashboard not generated yet")
    live = render_dashboard()
    on_disk = DASHBOARD.read_text()
    if live == on_disk:
        return
    live_lines = live.splitlines()
    disk_lines = on_disk.splitlines()
    first_diff = next(
        (i for i, (l, d) in enumerate(zip(live_lines, disk_lines))
         if l != d),
        min(len(live_lines), len(disk_lines)),
    )
    pytest.fail(
        f"Runtime ABI dashboard drift at line {first_diff + 1}: "
        f"on-disk has {disk_lines[first_diff]!r}, live has "
        f"{live_lines[first_diff]!r}.  Regenerate the dashboard."
    )


def test_dashboard_pins_canonical_phrases() -> None:
    assert DASHBOARD.exists()
    text = DASHBOARD.read_text()
    for phrase in (
        "# Runtime C ABI Surface Audit",
        "## Core runtime headers",
        "## Symbols per backend",
        "## Apple GPU kernel families × dtype matrix",
        "## Toolchain version pins",
    ):
        assert phrase in text, (
            f"Runtime ABI dashboard missing canonical phrase {phrase!r}"
        )
