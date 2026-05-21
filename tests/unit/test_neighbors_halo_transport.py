"""Sub-4 — Halo transport lowering + mock-collective runtime.

Three layers of guards:

1. Structural — the C++ pass + Tessera ODS shells + dialect registration
   are wired together.  These run without a built tessera-opt.

2. Behavioral — when the binary is built, the full halo pipeline lowers
   a sharded stencil into pack/transport/unpack triples with no
   surviving halo.exchange ops.

3. Numerical — the Python mock-collective transport
   (tessera.testing.halo_transport) actually moves bytes between virtual
   ranks: after one ring exchange, rank 0's "hi" ghost equals rank 1's
   "lo" interior column for a 2-rank 1D mesh.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEIGHBORS_ROOT = REPO_ROOT / "src" / "compiler" / "tessera_neighbors"
PASS_CPP = (
    NEIGHBORS_ROOT
    / "lib" / "Dialect" / "Neighbors" / "Transforms"
    / "HaloTransportLowerPass.cpp"
)
DIALECT_CPP = (
    NEIGHBORS_ROOT / "lib" / "Dialect" / "Neighbors" / "IR"
    / "TesseraNeighbors.cpp"
)
TESSERA_OPS_TD = REPO_ROOT / "src" / "compiler" / "ir" / "TesseraOps.td"
LIT_FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "neighbors_halo_transport_lower.mlir"
)


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #


def test_pass_source_exists() -> None:
    assert PASS_CPP.exists()


def test_pass_emits_three_triple_ops() -> None:
    text = PASS_CPP.read_text()
    assert "tessera.neighbors.halo.pack" in text
    assert "tessera.neighbors.halo.transport" in text
    assert "tessera.neighbors.halo.unpack" in text
    # Peer rule for the ring topology.
    assert '"neg1"' in text
    assert '"pos1"' in text


def test_pass_skips_width_zero_axes() -> None:
    text = PASS_CPP.read_text()
    assert "if (w == 0) continue" in text


def test_dialect_registers_three_new_ops() -> None:
    text = DIALECT_CPP.read_text()
    assert "struct HaloPackOp" in text
    assert "struct HaloTransportOp" in text
    assert "struct HaloUnpackOp" in text
    # addOperations<> must list them.
    add_ops_start = text.find("addOperations<")
    assert add_ops_start != -1
    add_ops = text[add_ops_start: add_ops_start + 500]
    assert "HaloPackOp" in add_ops
    assert "HaloTransportOp" in add_ops
    assert "HaloUnpackOp" in add_ops


def test_tessera_dialect_shells_registered() -> None:
    """The parent Tessera dialect (`tessera`) also declares string-name
    shells for the new ops so generic-form parsing works.  Without
    these the parser rejects ``"tessera.neighbors.halo.pack"(...)``
    even when the neighbors dialect knows the op."""
    text = TESSERA_OPS_TD.read_text()
    assert 'Tessera_NeighborsHaloPackOp' in text
    assert 'Tessera_NeighborsHaloTransportOp' in text
    assert 'Tessera_NeighborsHaloUnpackOp' in text


def test_lit_fixture_chains_all_four_passes() -> None:
    text = LIT_FIXTURE.read_text()
    for arg in (
        "-tessera-stencil-lower",
        "-tessera-boundary-condition-lower",
        "-tessera-halo-mesh-integration",
        "-tessera-halo-transport-lower",
    ):
        assert arg in text


# --------------------------------------------------------------------------- #
# Behavioral
# --------------------------------------------------------------------------- #


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


def test_pass_runs_against_lit_fixture() -> None:
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
    if (
        r.returncode != 0
        and "Unknown command line argument" in r.stderr
        and "tessera-halo-transport-lower" in r.stderr
    ):
        pytest.skip("tessera-opt predates the pass — rebuild required")
    assert r.returncode == 0, f"halo transport pass failed: {r.stderr}"
    out = r.stdout
    # No surviving halo.exchange — fully lowered.
    assert 'tessera.neighbors.halo.exchange' not in out
    # The triple ops all appear.
    assert 'tessera.neighbors.halo.pack' in out
    assert 'tessera.neighbors.halo.transport' in out
    assert 'tessera.neighbors.halo.unpack' in out
    # Provenance tag set on every triple op.
    assert out.count('inserted_by = "halo-transport-lower"') >= 6


# --------------------------------------------------------------------------- #
# Numerical — mock-collective transport
# --------------------------------------------------------------------------- #


class TestMockCollective:
    """Exercise tessera.testing.halo_transport — the runtime contract."""

    def test_pack_lo_returns_first_slab(self):
        from tessera.testing.halo_transport import halo_pack
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        slab = halo_pack(a, axis=1, side="lo", width=1)
        np.testing.assert_array_equal(slab, a[:, 0:1])

    def test_pack_hi_returns_last_slab(self):
        from tessera.testing.halo_transport import halo_pack
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        slab = halo_pack(a, axis=1, side="hi", width=2)
        np.testing.assert_array_equal(slab, a[:, 3:5])

    def test_unpack_replaces_ghost_region(self):
        from tessera.testing.halo_transport import halo_pack, halo_unpack
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        slab = np.full((4, 1), -99.0, dtype=np.float32)
        out = halo_unpack(a, slab, axis=1, side="lo", width=1)
        # The slab replaced column 0; the rest of `a` is unchanged.
        np.testing.assert_array_equal(out[:, 0], -99.0)
        np.testing.assert_array_equal(out[:, 1:], a[:, 1:])
        # And `a` itself was not mutated.
        np.testing.assert_array_equal(a[:, 0], np.array([0, 5, 10, 15]))

    def test_two_rank_ring_exchange_periodic(self):
        """1D mesh of 2 ranks, each holds a 4×4 tile.  After one ring
        exchange, rank 0's "hi" boundary equals rank 1's "lo" interior
        and vice versa — the wrap-around makes both ranks agree on the
        joined field."""
        from tessera.testing.halo_transport import halo_exchange_ring
        # rank 0: zeros (so the unpacked "hi" boundary shows rank-1 values)
        # rank 1: full of 7.0
        rank0 = np.zeros((4, 4), dtype=np.float32)
        rank1 = np.full((4, 4), 7.0, dtype=np.float32)
        # Halo on axis 1 (column-direction), width 1.
        out = halo_exchange_ring([rank0, rank1], axes_widths=[(1, 1)])
        out0, out1 = out
        # rank 0's "hi" ghost should now be rank 1's "lo" interior = 7.0
        np.testing.assert_array_equal(out0[:, 3], 7.0)
        # rank 0's "lo" ghost should now be rank 1's "hi" interior = 7.0
        np.testing.assert_array_equal(out0[:, 0], 7.0)
        # rank 1's "lo" and "hi" ghosts pull from rank 0's interior = 0
        np.testing.assert_array_equal(out1[:, 0], 0.0)
        np.testing.assert_array_equal(out1[:, 3], 0.0)
        # Interior cells (columns 1, 2) were not touched.
        np.testing.assert_array_equal(out0[:, 1:3], rank0[:, 1:3])
        np.testing.assert_array_equal(out1[:, 1:3], rank1[:, 1:3])

    def test_three_rank_ring_exchange(self):
        from tessera.testing.halo_transport import halo_exchange_ring
        # rank r is filled with value r.
        ranks = [np.full((2, 4), float(r), dtype=np.float32) for r in range(3)]
        out = halo_exchange_ring(ranks, axes_widths=[(1, 1)])
        # rank 0 receives from rank 2 ("hi" side, via wrap) → "lo" ghost = 2
        # and from rank 1 ("lo" side, via wrap) → "hi" ghost = 1
        np.testing.assert_array_equal(out[0][:, 0], 2.0)
        np.testing.assert_array_equal(out[0][:, 3], 1.0)
        # rank 1 receives from rank 0 (lo) and rank 2 (hi)
        np.testing.assert_array_equal(out[1][:, 0], 0.0)
        np.testing.assert_array_equal(out[1][:, 3], 2.0)
        # rank 2 receives from rank 1 (lo) and rank 0 (hi, via wrap)
        np.testing.assert_array_equal(out[2][:, 0], 1.0)
        np.testing.assert_array_equal(out[2][:, 3], 0.0)
