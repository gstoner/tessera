"""Halo execution lane — first end-to-end execute-and-compare oracle test.

This is the template for Layer-6 (oracle) testing the audit doc
identified as the single most-valuable test class.  It ties three
artifacts together that previously only existed independently:

  1. The C++ pass output       — pack/transport/unpack ops emitted by
                                  HaloTransportLowerPass.
  2. The Python mock collective — halo_pack / halo_transport_ring /
                                  halo_unpack in
                                  tessera.testing.halo_transport.
  3. A numpy oracle            — hand-built ghost-cell content from
                                  the input arrays.

The single test below proves that the C++ ABI (per-(axis, side, width)
parameters on each emitted op) **matches** what the Python mock
collective consumes, and **both match** a hand-rolled oracle.  If any
of the three drifts, the test fails with a named message pointing at
which contract broke.

CPU/mock-backed today.  When NVIDIA/ROCm hardware enables (Phase G/H),
the same harness extends to:
  * Generate IR for a halo.exchange.
  * Lower through the real transport kernel.
  * Execute on hardware.
  * Compare to this same oracle.

Architecture: ``docs/audit/compiler_correctness_testing_audit.md`` §
"Coverage matrix — semantic / oracle".
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_halo = importlib.import_module("tessera.testing.halo_transport")


# ─────────────────────────────────────────────────────────────────────────────
# IR parsing — extract (axis, side, width) triples from emitted IR.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransportTriple:
    """One (pack, transport, unpack) triple as extracted from emitted IR."""
    axis: int
    side: str   # "lo" or "hi"
    width: int
    peer_rule: str  # "neg1" or "pos1"

    @property
    def py_key(self) -> tuple[int, str]:
        """Key used by tessera.testing.halo_transport's mock collective."""
        return (self.axis, self.side)


_PACK_RE = re.compile(
    r'"tessera\.neighbors\.halo\.pack".*?'
    r'axis\s*=\s*(\d+).*?'
    r'side\s*=\s*"(\w+)".*?'
    r'width\s*=\s*(\d+)',
    re.DOTALL,
)

_TRANSPORT_RE = re.compile(
    r'"tessera\.neighbors\.halo\.transport".*?'
    r'axis\s*=\s*(\d+).*?'
    r'peer_rule\s*=\s*"(\w+)".*?'
    r'side\s*=\s*"(\w+)".*?'
    r'width\s*=\s*(\d+)',
    re.DOTALL,
)


def _extract_triples_from_ir(ir_text: str) -> list[TransportTriple]:
    """Parse emitted IR into a list of (axis, side, width, peer_rule)
    tuples — one per transport op (each pack/transport/unpack triple
    has matching axis+side+width by construction).

    The parser is line-based; the regex matches against each
    halo.transport op line individually."""
    triples = []
    for line in ir_text.splitlines():
        if "tessera.neighbors.halo.transport" not in line:
            continue
        # Each attribute appears once per line; match individually.
        ax = re.search(r'axis\s*=\s*(\d+)', line)
        sd = re.search(r'side\s*=\s*"(\w+)"', line)
        wd = re.search(r'width\s*=\s*(\d+)', line)
        pr = re.search(r'peer_rule\s*=\s*"(\w+)"', line)
        if not (ax and sd and wd and pr):
            continue
        triples.append(TransportTriple(
            axis=int(ax.group(1)),
            side=sd.group(1),
            width=int(wd.group(1)),
            peer_rule=pr.group(1),
        ))
    return triples


# ─────────────────────────────────────────────────────────────────────────────
# Oracle — hand-built ghost-cell content.
# ─────────────────────────────────────────────────────────────────────────────


def _oracle_ring_exchange(
    fields: list[np.ndarray],
    axes_widths: list[tuple[int, int]],
) -> list[np.ndarray]:
    """A pure-numpy reference of the canonical periodic ring exchange.

    Independent re-derivation of what the mock collective and the
    transport pass should produce: for each axis a with width w, each
    rank r's "lo" ghost column equals rank (r-1) % N's "hi" interior
    column on axis a, and vice versa.  Interior cells are untouched.
    """
    N = len(fields)
    out = [f.copy() for f in fields]
    for axis, width in axes_widths:
        if width <= 0:
            continue
        # Snapshot pre-exchange to avoid intra-step overwrites.
        snapshot = [f.copy() for f in fields]
        for r in range(N):
            # "lo" ghost on r ← from "hi" interior of (r-1) % N.
            src_rank = (r - 1) % N
            lo_dst = [slice(None)] * out[r].ndim
            lo_dst[axis] = slice(0, width)
            hi_src = [slice(None)] * snapshot[src_rank].ndim
            hi_src[axis] = slice(
                snapshot[src_rank].shape[axis] - width,
                snapshot[src_rank].shape[axis],
            )
            out[r][tuple(lo_dst)] = snapshot[src_rank][tuple(hi_src)]
            # "hi" ghost on r ← from "lo" interior of (r+1) % N.
            src_rank = (r + 1) % N
            hi_dst = [slice(None)] * out[r].ndim
            hi_dst[axis] = slice(
                out[r].shape[axis] - width, out[r].shape[axis],
            )
            lo_src = [slice(None)] * snapshot[src_rank].ndim
            lo_src[axis] = slice(0, width)
            out[r][tuple(hi_dst)] = snapshot[src_rank][tuple(lo_src)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


_INPUT_MODULE = """\
func.func @t(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %x = "tessera.neighbors.halo.exchange"(%arg0) {halo.width = [1, 1], mesh.axis = "dp"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %x : tensor<?x?xf32>
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 — the execute-and-compare oracle test.
# ─────────────────────────────────────────────────────────────────────────────


class TestHaloExecutionLane:
    """Single canonical workflow:

      (1) Construct a 2-rank workload and an oracle.
      (2) Lower a halo.exchange IR through tessera-opt to triples.
      (3) Extract per-triple (axis, side, width) parameters.
      (4) Replay those parameters through the Python mock collective.
      (5) Assert mock output bitwise-matches the oracle.

    Steps 3+4+5 together prove the IR ABI ⇔ runtime behaviour ⇔ oracle
    contract is self-consistent.
    """

    def _ir_triples(self) -> list[TransportTriple]:
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = subprocess.run(
            [binary,
             "-tessera-halo-transport-lower"],
            input=_INPUT_MODULE,
            capture_output=True, text=True, timeout=30,
        )
        if (
            r.returncode != 0
            and "Did you mean" in r.stderr
            and "tessera-halo-transport-lower" in r.stderr
        ):
            pytest.skip("tessera-opt predates the pass — rebuild required")
        assert r.returncode == 0, r.stderr
        triples = _extract_triples_from_ir(r.stdout)
        assert triples, (
            f"no transport triples extracted from emitted IR; first 500 "
            f"chars:\n{r.stdout[:500]}"
        )
        return triples

    def test_ir_emits_expected_per_axis_per_side_parameters(self) -> None:
        """Step 1 of the lane — the IR carries (axis, side, width)
        parameters that match the input halo.width contract.

        Input: halo.exchange with width=[1, 1] on a 2D field.
        Expected: 4 triples — (axis 0, lo, 1), (axis 0, hi, 1),
                              (axis 1, lo, 1), (axis 1, hi, 1).
        Each side carries the matching peer_rule.
        """
        triples = self._ir_triples()
        assert len(triples) == 4, (
            f"expected 4 triples (2 axes × 2 sides), got {len(triples)}: "
            f"{triples}"
        )
        expected = {
            (0, "lo", 1, "neg1"),
            (0, "hi", 1, "pos1"),
            (1, "lo", 1, "neg1"),
            (1, "hi", 1, "pos1"),
        }
        actual = {(t.axis, t.side, t.width, t.peer_rule) for t in triples}
        assert actual == expected, (
            f"IR triples drifted from expected ABI:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    def test_execute_and_compare_two_ranks_periodic_ring(self) -> None:
        """The flagship test — IR-derived parameters drive the Python
        mock collective; output must match a numpy oracle bitwise.

        2-rank workload: rank 0 = zeros(4, 4); rank 1 = full(4, 4, 7.0).
        After periodic ring exchange on axis 1 width 1, rank 0's "hi"
        ghost column equals 7.0, rank 1's "lo" ghost column equals 0.0.
        Interior columns are untouched.
        """
        triples = self._ir_triples()

        # Step 2 — extract (axis, width) pairs from the triples to drive
        # the mock-collective with the exact contract the IR emits.
        axes_widths = sorted({
            (t.axis, t.width) for t in triples
        })
        # halo.exchange input had halo.width=[1, 1] on a rank-2 field,
        # so the emitted triples cover axes (0, 1) both width 1.  Drop
        # axis 0 for this test (we're exchanging only along axis 1).
        # Keep both for parity with the IR, but the mock-collective
        # handles each axis independently anyway.

        # Step 3 — input arrays per rank.
        rank0 = np.zeros((4, 4), dtype=np.float32)
        rank1 = np.full((4, 4), 7.0, dtype=np.float32)

        # Step 4 — drive the mock collective using only the (axis, width)
        # pairs the IR carried.  The mock collective owns the ring
        # semantics (matching peer_rule="neg1|pos1").
        mock_out = _halo.halo_exchange_ring(
            [rank0, rank1],
            axes_widths=list(axes_widths),
        )

        # Step 5 — independent oracle (re-derived from periodic-ring
        # semantics in this file).
        oracle_out = _oracle_ring_exchange(
            [rank0, rank1], list(axes_widths),
        )

        # Bitwise match — float32 memcpy semantics on both sides.
        for r in range(2):
            np.testing.assert_array_equal(
                mock_out[r], oracle_out[r],
                err_msg=(
                    f"rank {r}: mock collective output diverged from "
                    f"oracle.  Either the C++ ABI changed (different "
                    f"axis/side/width emitted), the Python mock "
                    f"collective behaviour changed, or the oracle is "
                    f"wrong — exactly one of the three needs fixing."
                ),
            )

        # Belt-and-braces: the structural ghost-cell content matches
        # the documented contract.
        # axis 1: rank 0 "hi" boundary = rank 1 "lo" interior = 7.0
        np.testing.assert_array_equal(mock_out[0][:, 3], 7.0)
        # axis 1: rank 0 "lo" boundary = rank 1 "hi" interior = 7.0 (wrap)
        np.testing.assert_array_equal(mock_out[0][:, 0], 7.0)
        # axis 1: rank 1 "lo" + "hi" boundaries pull from rank 0 = 0.0
        np.testing.assert_array_equal(mock_out[1][:, 0], 0.0)
        np.testing.assert_array_equal(mock_out[1][:, 3], 0.0)

    def test_execute_and_compare_three_ranks_random_inputs(self) -> None:
        """A more demanding workload: 3 ranks with deterministic random
        arrays, single-axis exchange.

        Restricted to one axis because 2-axis halo exchange has corner-
        cell semantic ambiguity (both axes' wraps target the same
        corner; whichever unpack runs last wins).  Today's mock
        collective uses dict-insertion-order to disambiguate, which is
        an implementation detail not a contract.  Single-axis keeps
        the oracle comparison crisp.

        Future: when the C++ pass emits diagonal-exchange triples for
        corner handling, this test extends to multi-axis trivially.
        """
        triples = self._ir_triples()
        # Single axis chosen from the emitted IR (still proves the
        # parameters came from the IR, not hard-coded).
        all_axes = sorted({t.axis for t in triples})
        chosen_axis = all_axes[-1]  # column-direction
        chosen_widths = [(t.axis, t.width) for t in triples
                         if t.axis == chosen_axis]
        axes_widths = sorted(set(chosen_widths))

        rng = np.random.default_rng(seed=42)
        ranks = [rng.standard_normal((6, 8)).astype(np.float32)
                 for _ in range(3)]

        mock_out = _halo.halo_exchange_ring(ranks, axes_widths=axes_widths)
        oracle_out = _oracle_ring_exchange(ranks, list(axes_widths))

        for r in range(3):
            np.testing.assert_array_equal(
                mock_out[r], oracle_out[r],
                err_msg=f"rank {r} of 3-rank random workload diverged",
            )

    def test_widths_in_ir_match_halo_width_input_attribute(self) -> None:
        """Critical for ABI lock: every triple emitted carries the same
        width as the input halo.exchange's halo.width attribute.
        Catches the bug class "C++ pass starts emitting width=2 for an
        input width=1 op" silently."""
        triples = self._ir_triples()
        # Input halo.width = [1, 1] — every triple's width must be 1.
        for t in triples:
            assert t.width == 1, (
                f"triple {t} has width {t.width}, expected 1 (matching "
                f"input halo.width=[1, 1])"
            )

    def test_each_axis_has_both_sides(self) -> None:
        """Every axis with width>0 must emit BOTH a "lo" and a "hi"
        triple — single-sided exchange is a regression that would leave
        one boundary unfilled."""
        triples = self._ir_triples()
        by_axis: dict[int, set[str]] = {}
        for t in triples:
            by_axis.setdefault(t.axis, set()).add(t.side)
        for axis, sides in by_axis.items():
            assert sides == {"lo", "hi"}, (
                f"axis {axis}: emitted sides {sides} != {{lo, hi}} — "
                f"single-sided exchange detected"
            )
