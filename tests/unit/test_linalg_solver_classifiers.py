"""The linalg solver passes classify every catalog ``linalg_solver`` op.

TILE-LATENT-DEFECTS-2026-09-27. ``MixedPrecision`` and ``IterativeRefinement``
select ops by exact identity (they used to match ``contains("solve")``). The
C++ library cannot read the Python op catalog, so this test is the link: every
``OP_SPECS`` entry with ``lowering="linalg_solver"`` must be named, as an exact
string comparison, by both classifiers and exercised by the lit fixture. The
first version of the exact allowlists dropped the public ``tessera.solve``,
which is how this test came to exist.
"""
from __future__ import annotations

import re
from pathlib import Path

from tessera.compiler.op_catalog import OP_SPECS

ROOT = Path(__file__).resolve().parents[2]
PASSES = ROOT / "src/solvers/linalg/lib/Passes"
FIXTURE = ROOT / "tests/tessera-ir/phase5/linalg_solver_op_identity.mlir"


def _catalog_solver_ops() -> set[str]:
    return {s.graph_name for s in OP_SPECS.values() if s.lowering == "linalg_solver"}


def _exact_names(path: Path) -> set[str]:
    return set(re.findall(r'name == "([^"]+)"', path.read_text()))


def test_catalog_has_linalg_solver_ops() -> None:
    assert "tessera.solve" in _catalog_solver_ops()


def test_every_catalog_solver_op_is_classified_by_both_passes() -> None:
    ops = _catalog_solver_ops()
    for source in ("MixedPrecision.cpp", "IterativeRefinement.cpp"):
        missing = ops - _exact_names(PASSES / source)
        assert not missing, f"{source} does not classify catalog ops {sorted(missing)}"


def _registered_graph_ops() -> set[str]:
    td = (ROOT / "src/compiler/ir/TesseraOps.td").read_text()
    return {"tessera." + m for m in re.findall(r'Op<\s*Tessera_Dialect,\s*"([^"]+)"', td)}


#: Catalog solver ops with no ODS declaration. The tessera dialect rejects an
#: unknown op on parse, so no lit fixture can carry these; the classifier check
#: above still covers them. When one is registered this set must shrink.
_UNREGISTERED = {"tessera.solve"}


def test_unregistered_exemption_is_exact() -> None:
    ops = _catalog_solver_ops()
    assert _UNREGISTERED <= ops
    assert _UNREGISTERED.isdisjoint(_registered_graph_ops()), (
        "an exempt op is now registered: drop it from _UNREGISTERED and add "
        "it to the lit fixture")


def test_every_registered_catalog_solver_op_is_exercised_by_the_fixture() -> None:
    text = FIXTURE.read_text()
    for op in _catalog_solver_ops() - _UNREGISTERED:
        assert op in _registered_graph_ops(), op
        assert re.search(rf"^\s*//\s*MP: {re.escape(op)}\s*$", text, re.M), op
        assert re.search(rf"^\s*//\s*IR: {re.escape(op)}\s*$", text, re.M), op


def test_no_substring_matching_returns() -> None:
    for source in ("MixedPrecision.cpp", "IterativeRefinement.cpp"):
        code = re.sub(r"//[^\n]*", "", (PASSES / source).read_text())
        assert ".contains(" not in code, source
