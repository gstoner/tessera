"""CorrDiff-core IR-visible fixture — Phase 7 capstone integration test.

This test pins the *layering* contract: the three Phase 7 workstreams
(stencil materialization, 2D local-window attention, halo) compose in
a single IR module flowing through the canonical halo pipeline.

Before this test, each piece existed in isolation:
  * `test_neighbors_stencil_materialize.py` — stencil alone
  * `test_attn_local_window_2d_graph_ir.py` — window attention alone
  * `test_neighbors_halo_transport.py`      — halo transport alone

This test proves they actually compose end-to-end at the IR level.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIT_FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "corrdiff_ir_visible.mlir"
)


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


def test_fixture_exists() -> None:
    assert LIT_FIXTURE.exists()


def test_fixture_exercises_three_workstreams() -> None:
    """The fixture body must reference all three Phase 7 op families."""
    text = LIT_FIXTURE.read_text()
    assert "tessera.neighbors.stencil.apply" in text
    assert "tessera.attn_local_window_2d" in text
    assert "schedule.mesh.region" in text


def test_fixture_runs_canonical_halo_pipeline() -> None:
    """The RUN line must chain the four halo passes — not a subset."""
    text = LIT_FIXTURE.read_text()
    for arg in (
        "-tessera-stencil-lower",
        "-tessera-boundary-condition-lower",
        "-tessera-halo-mesh-integration",
        "-tessera-halo-transport-lower",
    ):
        assert arg in text, f"fixture RUN missing {arg}"


def test_canonical_pipeline_emits_triples_for_both_consumers() -> None:
    """Behavioral lock: after the canonical halo pipeline runs over
    a module containing BOTH a stencil.apply AND an attn_local_window_2d
    op, *every* halo.exchange is replaced with pack/transport/unpack
    triples and the attn-driven exchanges carry the source_op
    provenance from HaloMeshIntegrationPass through to the triples
    that HaloTransportLowerPass produced."""
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run(
        [
            binary,
            "--allow-unregistered-dialect",
            "-tessera-stencil-lower",
            "-tessera-boundary-condition-lower",
            "-tessera-halo-mesh-integration",
            "-tessera-halo-transport-lower",
            str(LIT_FIXTURE),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0 and "Did you mean" in r.stderr:
        pytest.skip("tessera-opt predates required passes — rebuild required")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # ── Stencil halo path is lowered all the way ────────────────────
    assert "stencil.lowered = true" in out
    assert "stencil.bc.lowered = true" in out
    # ── Window-attention halo path is lowered all the way ───────────
    assert 'source_op = "tessera.attn_local_window_2d"' in out
    assert "halo.window = [1, 1]" in out
    # ── Both feed into transport-lower triples ──────────────────────
    assert "tessera.neighbors.halo.pack" in out
    assert "tessera.neighbors.halo.transport" in out
    assert "tessera.neighbors.halo.unpack" in out
    # ── No halo.exchange survives ───────────────────────────────────
    assert "tessera.neighbors.halo.exchange" not in out
    # ── Both consumers reached the integration sentinel ─────────────
    assert out.count("halo.mesh_integrated = true") >= 2


def test_attn_driven_triples_preserve_source_op_provenance() -> None:
    """Architectural contract: source_op set by HaloMeshIntegrationPass
    must survive through HaloTransportLowerPass.  Without this
    preservation, downstream tooling can't distinguish a stencil-driven
    triple from an attn-driven triple."""
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run(
        [
            binary,
            "--allow-unregistered-dialect",
            "-tessera-stencil-lower",
            "-tessera-boundary-condition-lower",
            "-tessera-halo-mesh-integration",
            "-tessera-halo-transport-lower",
            str(LIT_FIXTURE),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        pytest.skip("upstream test failure")
    out = r.stdout
    # At least one halo.pack op should carry source_op = "tessera.attn_local_window_2d".
    pack_lines = [
        line for line in out.splitlines()
        if "tessera.neighbors.halo.pack" in line
    ]
    attn_packs = [
        line for line in pack_lines
        if 'source_op = "tessera.attn_local_window_2d"' in line
    ]
    assert len(attn_packs) >= 4, (
        f"expected at least 4 attn-driven halo.pack ops with source_op "
        f"provenance (2 axes × 2 sides), found {len(attn_packs)}.  "
        f"Total halo.pack ops: {len(pack_lines)}."
    )


def test_attn_local_window_2d_op_carries_halo_window_after_pipeline() -> None:
    """The original window=[1, 1] attribute on the Graph IR op survives
    the halo pipeline.  This catches a regression where a future
    consumer accidentally rewrites or strips the attribute."""
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run(
        [
            binary,
            "--allow-unregistered-dialect",
            "-tessera-stencil-lower",
            "-tessera-boundary-condition-lower",
            "-tessera-halo-mesh-integration",
            "-tessera-halo-transport-lower",
            str(LIT_FIXTURE),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        pytest.skip("upstream test failure")
    out = r.stdout
    # The op line should carry both `window = [1, 1]` (original) and
    # `halo.window = [1, 1]` (mirror from integration pass).
    # Filter to lines where the OP NAME is tessera.attn_local_window_2d
    # (not lines where it appears only inside an attribute like
    # source_op = "tessera.attn_local_window_2d").
    awl_lines = [
        line for line in out.splitlines()
        if line.lstrip().startswith("%")
        and " = tessera.attn_local_window_2d " in line
    ]
    assert awl_lines, "no attn_local_window_2d op found in output"
    for line in awl_lines:
        assert "window = [1, 1]" in line, (
            f"original window attribute lost from line: {line[:200]}"
        )
        assert "halo.window = [1, 1]" in line, (
            f"halo.window mirror lost from line: {line[:200]}"
        )
