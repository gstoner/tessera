"""Ask 3 + Ask 4-B — halo + mesh integration + halo-aware metadata.

Structural + behavioral guards for the new ``HaloMeshIntegrationPass``
that inserts ``halo.exchange`` before sharded ``stencil.apply`` ops and
flags BC-vs-mesh-policy conflicts as named diagnostics.  Plus a registry
guard for the ``_HALO_AWARE_OPS`` table that makes
``attn_local_window_2d`` discoverable from the same place stencils live.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEIGHBORS_ROOT = REPO_ROOT / "src" / "compiler" / "tessera_neighbors"
PASS_CPP = (
    NEIGHBORS_ROOT
    / "lib" / "Dialect" / "Neighbors" / "Transforms"
    / "HaloMeshIntegrationPass.cpp"
)
LIT_FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "neighbors_halo_mesh_integration.mlir"
)

_pc_mod = importlib.import_module("tessera.compiler.primitive_coverage")


# --------------------------------------------------------------------------- #
# Structural — C++ pass source
# --------------------------------------------------------------------------- #


def test_pass_source_exists() -> None:
    assert PASS_CPP.exists()


def test_pass_declares_registration_fn() -> None:
    text = PASS_CPP.read_text()
    assert "void registerHaloMeshIntegrationPass()" in text
    assert "tessera-halo-mesh-integration" in text


def test_pass_inserts_halo_exchange() -> None:
    text = PASS_CPP.read_text()
    # The exchange op must be emitted as a real tessera.neighbors op,
    # not as a placeholder attribute.
    assert '"tessera.neighbors.halo.exchange"' in text
    assert 'addOperands' in text
    assert 'halo.width' in text
    assert 'mesh.axis' in text
    assert 'inserted_by' in text


def test_pass_emits_bc_mesh_diagnostic() -> None:
    text = PASS_CPP.read_text()
    assert "mesh.bc_conflict" in text
    assert "periodic" in text
    # The diagnostic string is concatenated across two C++ literals;
    # check the two halves the runtime emits.
    assert "incompatible with" in text
    assert "mesh axis policy" in text


def test_pass_writes_idempotent_sentinel() -> None:
    text = PASS_CPP.read_text()
    assert "halo.mesh_integrated" in text


def test_pass_registered_everywhere() -> None:
    header = (
        NEIGHBORS_ROOT
        / "include" / "tessera" / "Dialect" / "Neighbors" / "Transforms"
        / "Passes.h"
    )
    assert "registerHaloMeshIntegrationPass" in header.read_text()
    assert "HaloMeshIntegrationPass.cpp" in (
        NEIGHBORS_ROOT / "CMakeLists.txt").read_text()
    assert "registerHaloMeshIntegrationPass" in (
        REPO_ROOT / "tools" / "tessera-opt" / "tessera-opt.cpp"
    ).read_text()


def test_lit_fixture_covers_periodic_conflict_and_reflect_clean() -> None:
    text = LIT_FIXTURE.read_text()
    assert "-tessera-halo-mesh-integration" in text
    # Periodic stencil over an open mesh axis must produce mesh.bc_conflict.
    assert "stencil BC 'periodic' incompatible" in text
    # Reflect BC is safe under either policy — checked via CHECK-NOT.
    assert "CHECK-NOT" in text


# --------------------------------------------------------------------------- #
# Structural — halo_aware registry
# --------------------------------------------------------------------------- #


def test_halo_aware_registry_lists_attn_local_window_2d() -> None:
    """Ask 4-B — _HALO_AWARE_OPS is the canonical enumeration the C++
    pass (and external tooling) consults to know which ops require halo
    exchange when sharded."""
    assert "attn_local_window_2d" in _pc_mod._HALO_AWARE_OPS
    entry = _pc_mod._HALO_AWARE_OPS["attn_local_window_2d"]
    assert entry["halo_width_from_kwarg"] == "window"
    assert entry["halo_width_attr"] == "attn.window"
    # The window covers the spatial axes of a (B, H, Hq, Wq, D) tensor.
    assert entry["spatial_axes"] == "2,3"


def test_attn_local_window_2d_carries_halo_aware_metadata() -> None:
    # ``coverage_for`` returns the augmented ``PrimitiveCoverage`` entry
    # (which carries metadata).  Raw ``OP_SPECS[name]`` returns the
    # narrower ``OpSpec`` and is intentionally metadata-free.
    cov = _pc_mod.coverage_for("attn_local_window_2d")
    assert "halo_aware" in cov.metadata
    assert cov.metadata["halo_aware"]["halo_width_from_kwarg"] == "window"
    assert cov.metadata["halo_aware"]["halo_width_attr"] == "attn.window"


# --------------------------------------------------------------------------- #
# Behavioral — subprocess against tessera-opt
# --------------------------------------------------------------------------- #


def _find_tessera_opt() -> str | None:
    for candidate in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
        str(REPO_ROOT / "build" / "bin" / "tessera-opt"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def test_pass_wraps_sharded_attn_local_window_2d() -> None:
    """Closing-the-loop (2026-05-21) — the pass now recognises
    tessera.attn_local_window_2d as a halo-aware consumer and inserts
    halo.exchange before it just like for stencil.apply."""
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    awl_fixture = (
        REPO_ROOT / "tests" / "tessera-ir" / "phase7"
        / "halo_mesh_integration_attn_local_window_2d.mlir"
    )
    assert awl_fixture.exists()
    r = subprocess.run(
        [binary, "--allow-unregistered-dialect",
         "-tessera-halo-mesh-integration", str(awl_fixture)],
        capture_output=True, text=True, timeout=30,
    )
    if (
        r.returncode != 0
        and "Did you mean" in r.stderr
        and "tessera-halo-mesh-integration" in r.stderr
    ):
        pytest.skip("tessera-opt predates the pass — rebuild required")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # halo.exchange inserted with the attn-specific provenance.
    assert 'source_op = "tessera.attn_local_window_2d"' in out
    # halo.window attribute carries the original window through.
    assert "halo.window = [1, 1]" in out
    assert "halo.window = [2, 3]" in out
    # Unsharded variant did NOT get a halo.exchange wrapper.
    unsharded_section = out.split("unsharded_attn_local_window_2d")[1]
    # Up to the next func, count halo.exchange occurrences.
    next_func = unsharded_section.find("func.func")
    relevant = (unsharded_section[:next_func]
                if next_func != -1 else unsharded_section)
    assert "tessera.neighbors.halo.exchange" not in relevant


def test_source_walks_attn_local_window_2d_in_addition_to_stencil() -> None:
    """Structural guard: the C++ pass source must match both op names."""
    text = PASS_CPP.read_text()
    assert '"tessera.neighbors.stencil.apply"' in text
    assert '"tessera.attn_local_window_2d"' in text
    # The new handler is documented.
    assert "processAttnLocalWindow2D" in text


def test_pass_runs_against_lit_fixture() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built — skipping behavioral contract")
    result = subprocess.run(
        [
            binary,
            "--allow-unregistered-dialect",
            "-tessera-stencil-lower",
            "-tessera-boundary-condition-lower",
            "-tessera-halo-mesh-integration",
            str(LIT_FIXTURE),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if (
        result.returncode != 0
        and "Unknown command line argument" in result.stderr
        and "tessera-halo-mesh-integration" in result.stderr
    ):
        pytest.skip("tessera-opt binary predates the pass — rebuild required")
    assert result.returncode == 0, (
        f"halo-mesh integration failed: {result.stderr}"
    )
    out = result.stdout
    # Both fixture functions get a halo.exchange.
    assert out.count("tessera.neighbors.halo.exchange") >= 2
    # The pass leaves provenance.
    assert 'inserted_by = "halo-mesh-integration"' in out
    # Both ops get the integrated sentinel.
    assert out.count("halo.mesh_integrated = true") >= 2
    # Test 1 (periodic BC) must emit the conflict diagnostic on the
    # apply op; test 2 (reflect BC) must not.
    assert 'mesh.bc_conflict' in out
    # The conflict text is the named diagnostic per Architecture Decision #21.
    assert "stencil BC 'periodic' incompatible" in out
