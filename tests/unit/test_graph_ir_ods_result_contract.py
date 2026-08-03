"""The Python shape registry must agree with the Graph IR ODS on result types.

CLAUDE.md names the real architectural problem as a seam: "the Python
synthesizer and the C++ MLIR pipeline are two disconnected compilers." The
cache ops are a small, checkable instance of it.

`Tessera_KVCacheType` has been declared in `TesseraOps.td` for a long time, and
the ODS states the signatures precisely::

    tessera.kv_cache.append : (!tessera.kv_cache, tensor, tensor)
                                  -> !tessera.kv_cache

The Python emitter had no way to name a non-tensor type, so it emitted
`tensor<*x?>` at both ends. Nothing failed — an untyped tensor is what an
unknown tensor looks like too — and the five cache ops were filed as
"deliberately undeclared: opaque cache handle ... not describable by a tensor
shape rule". The type was never undescribable. It was declared in one compiler
and unreachable from the other.

This gate reads the ODS and holds the Python registry to it, so the two cannot
drift again without something going red.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tessera.compiler.graph_ir import _SHAPE_RULES, TENSOR_OPAQUE
from tessera.compiler.op_catalog import _SPECS, shape_rule_for

_TD = pathlib.Path(__file__).resolve().parents[2] / "src/compiler/ir/TesseraOps.td"

_OP_DEF = re.compile(r'def\s+(\w+)\s*:\s*Op<\s*Tessera_Dialect\s*,\s*"([^"]+)"', re.S)
_RESULTS = re.compile(r"let\s+results\s*=\s*\(outs([^;]*)\);", re.S)


def _ods_results() -> dict[str, str]:
    """Graph IR op mnemonic -> the text of its ODS `results` clause."""
    if not _TD.exists():  # pragma: no cover - source tree always has it
        pytest.skip(f"{_TD} not present")
    text = _TD.read_text()
    out: dict[str, str] = {}
    matches = list(_OP_DEF.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        results = _RESULTS.search(text[start:end])
        if results:
            out[match.group(2)] = results.group(1)
    return out


def _rule_result(graph_name: str):
    """What the Python rule claims this op returns, given opaque operands."""
    rule = shape_rule_for(graph_name)
    fn = _SHAPE_RULES.get(rule)
    if fn is None:
        return None
    return fn([TENSOR_OPAQUE, TENSOR_OPAQUE, TENSOR_OPAQUE], {})


def test_ods_handle_results_are_declared_state_handle_in_python():
    """Every op the ODS says returns `!tessera.kv_cache` must say so in Python.

    This is the check that would have caught the original defect. It fails if
    someone adds a handle-returning op to the ODS and leaves the Python side
    inferring a tensor — the exact shape of the bug this replaced.
    """
    catalog = {spec.graph_name for spec in _SPECS}
    wrong = []
    for mnemonic, results in _ods_results().items():
        if "KVCacheType" not in results:
            continue
        graph_name = f"tessera.{mnemonic}"
        if graph_name not in catalog:
            # e.g. `kv_cache.create`, which the Python surface does not expose.
            # Not a failure: the ODS may declare more than the frontend offers.
            continue
        result = _rule_result(graph_name)
        if result is None or "!tessera.kv_cache" not in str(result):
            wrong.append(
                f"{graph_name}: ODS returns Tessera_KVCacheType but the Python "
                f"rule {shape_rule_for(graph_name)!r} returns {result}"
            )

    assert not wrong, (
        "Python shape rules disagree with the Graph IR ODS on the result type:\n  "
        + "\n  ".join(wrong)
    )


def test_state_handle_ops_are_exactly_the_ods_handle_returning_ops():
    """No op may claim `state_handle` that the ODS does not back.

    The converse direction. Without it the rule could be applied to anything —
    a declaration is only worth what its evidence is, and here the evidence is
    the dialect.
    """
    ods_handles = {
        f"tessera.{m}" for m, r in _ods_results().items() if "KVCacheType" in r
    }
    claimed = {
        spec.graph_name for spec in _SPECS
        if shape_rule_for(spec.graph_name) == "state_handle"
    }
    unbacked = sorted(claimed - ods_handles)
    assert not unbacked, (
        f"ops declared `state_handle` with no handle-returning ODS op: {unbacked}"
    )


def test_kv_cache_read_returns_tensors_not_a_handle():
    """`read` is the member of the cache family that does NOT return a handle.

    All five were filed under one shared sentence — "opaque cache handle rather
    than a tensor type" — which was true of four of them. That is how the wrong
    classification stayed invisible: the reason read plausibly, so nothing in
    review pointed at the fifth op.
    """
    result = _rule_result("tessera.kv_cache.read")
    assert isinstance(result, tuple), f"expected (K, V), got {result}"
    assert len(result) == 2, result
    assert all("!tessera." not in str(t) for t in result), (
        f"kv_cache.read returns tensors, not handles: {result}"
    )


def test_kv_cache_read_has_no_graph_ir_ods_op():
    """Records a REAL gap rather than leaving it to be rediscovered.

    `tessera.kv_cache.read` is in the Python catalog and is lowered by the x86,
    ROCm and Apple Target IR dialects, but the Graph IR dialect never declared
    it — so unlike its four siblings there is no ODS contract to check the
    Python rule against. If someone adds the op to `TesseraOps.td`, this test
    fails and points them at extending the contract check above to cover it.
    """
    assert "kv_cache.read" not in _ods_results(), (
        "tessera.kv_cache.read is now declared in TesseraOps.td — extend "
        "test_ods_handle_results_are_declared_state_handle_in_python to cover "
        "its result contract, and delete this test."
    )
