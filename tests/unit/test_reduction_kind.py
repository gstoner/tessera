"""`tessera.reduce` executes the combiner its Graph op states (NVIDIA pre-PR review, 2026-09-26).

The frontends canonicalize ``ops.reduce(op=...)`` into the Graph op's ``kind``
and drop ``op``; the CPU runtime executor and the CPU compiler path read
``kwargs.get("op", "sum")`` and so executed every reduce as a sum --
``reduce(op="max", axis=1)`` on ``[[1, 5], [2, 3]]`` returned ``[6, 5]``.
Host-free: every path here is CPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import tessera
from tessera.compiler.reduction_kind import REDUCTION_KINDS, reduction_kind

X = np.array([[1.0, 5.0], [2.0, 3.0]], np.float32)
EXPECTED = {"sum": [6.0, 5.0], "max": [5.0, 3.0], "min": [1.0, 2.0], "mean": [3.0, 2.5]}


def test_the_kind_set_is_exactly_the_ods_reduction_kind_attr() -> None:
    td = (Path(__file__).resolve().parents[2] / "src/compiler/ir/TesseraOps.td").read_text()
    match = re.search(r"def Tessera_ReductionKindAttr\s*:\s*Tessera_EnumStrAttr<\[([^\]]*)\]", td)
    assert match is not None
    assert tuple(re.findall(r'"([^"]+)"', match[1])) == REDUCTION_KINDS


def _unwrap(value):
    return np.asarray(getattr(value, "_data", value))


# The kind must be a literal at the call site: the Graph IR frontend reads the
# source, and a closure variable is (correctly) an SSA value, not an attribute.
@tessera.jit
def _jit_sum(x):
    return tessera.ops.reduce(x, op="sum", axis=1)


@tessera.jit
def _jit_max(x):
    return tessera.ops.reduce(x, op="max", axis=1)


@tessera.jit
def _jit_min(x):
    return tessera.ops.reduce(x, op="min", axis=1)


@tessera.jit
def _jit_mean(x):
    return tessera.ops.reduce(x, op="mean", axis=1)


_JITTED = {"sum": _jit_sum, "max": _jit_max, "min": _jit_min, "mean": _jit_mean}


@pytest.mark.parametrize("kind", REDUCTION_KINDS)
def test_jit_reduce_executes_the_stated_kind(kind) -> None:
    f = _JITTED[kind]
    np.testing.assert_allclose(_unwrap(f(X)), EXPECTED[kind])
    assert f'kind = "{kind}"' in f.graph_ir.to_mlir()


@pytest.mark.parametrize("kind", REDUCTION_KINDS)
def test_eager_reference_matches_the_jit_for_every_kind(kind) -> None:
    np.testing.assert_allclose(_unwrap(tessera.ops.reduce(X, op=kind, axis=1)), EXPECTED[kind])


def test_jit_refuses_a_kind_outside_the_ods_set() -> None:
    """Refused while lowering (Tile IR states the combiner), before any call."""
    with pytest.raises(Exception, match="E_REDUCTION_KIND_UNSUPPORTED"):
        @tessera.jit
        def f(x):
            return tessera.ops.reduce(x, op="prod", axis=1)

        f(X)


@pytest.mark.parametrize("executor", ["runtime", "matmul_pipeline"])
def test_cpu_executors_refuse_a_kindless_reduce(executor) -> None:
    """A Graph reduce with neither `kind` nor `op` fails closed (#21a)."""
    if executor == "runtime":
        from tessera.runtime import _execute_runtime_cpu_op

        def run(kwargs):
            return _execute_runtime_cpu_op("tessera.reduce", [X], kwargs, np)
    else:
        from tessera.compiler.matmul_pipeline import _execute_op

        def run(kwargs):
            return _execute_op("tessera.reduce", [X], kwargs)
    with pytest.raises(ValueError, match="E_REDUCTION_KIND_MISSING"):
        run({"axis": 1})
    np.testing.assert_allclose(run({"axis": 1, "kind": "max"}), [5.0, 3.0])


def test_kind_and_legacy_op_must_agree() -> None:
    assert reduction_kind({"op": "min"}, where="t") == "min"
    assert reduction_kind({"kind": '"max"'}, where="t") == "max"
    with pytest.raises(ValueError, match="E_REDUCTION_KIND_UNSUPPORTED"):
        reduction_kind({"kind": "max", "op": "sum"}, where="t")
