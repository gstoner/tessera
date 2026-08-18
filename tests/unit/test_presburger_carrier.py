from __future__ import annotations

import pytest
import tessera as ts

from tessera.compiler.graph_ir import GraphIRFunction
from tessera.compiler.presburger import (
    NonlinearWitnessConstraint,
    NonlinearWitnessSystem,
    PolynomialTerm,
    PresburgerConstraint,
    PresburgerSystem,
    attach_presburger_system,
    attach_nonlinear_witness_system,
)


def test_typed_presburger_carrier_is_content_addressed_and_parseable_mlir():
    system = PresburgerSystem(
        symbols=("M", "N"),
        constraints=(
            PresburgerConstraint("ge", (1, 0), -1),
            PresburgerConstraint("eq", (-1, 1), 0),
            PresburgerConstraint("mod", (1, 0), modulus=4),
        ),
    )
    fn = GraphIRFunction("bounded")
    attach_presburger_system(fn, system)

    mlir = fn.to_mlir()
    assert "tessera.presburger_constraints" in mlir
    assert "coefficients = array<i64: 1, 0>" in mlir
    assert system.digest in mlir
    assert system.check_witness({"M": 8, "N": 8}) is True
    assert system.check_witness({"M": 8}) is None
    assert system.check_witness({"M": 8, "N": 7}) is False
    assert system.check_witness({"M": 6, "N": 6}) is False


def test_presburger_carrier_rejects_partial_rows_and_nonlinear_escape_hatches():
    with pytest.raises(ValueError, match="cover every symbol"):
        PresburgerSystem(
            symbols=("M", "N"),
            constraints=(PresburgerConstraint("eq", (1,), 0),),
        )
    with pytest.raises(ValueError, match="relation"):
        PresburgerConstraint("product", (1, 1), 0)  # type: ignore[arg-type]


def test_jit_emits_typed_affine_constraints_before_compilation():
    @ts.jit
    def bounded(x: ts.Tensor["M", "N"]):  # noqa: F821
        ts.require(ts.constraint.Range("M", 1, 64))
        ts.require(ts.constraint.Equal("M", "N"))
        ts.require(ts.constraint.Divisible("N", 8))
        return ts.ops.add(x, x)

    text = bounded.graph_ir.to_mlir()
    assert "tessera.presburger_constraints" in text
    assert 'symbols = ["M", "N"]' in text
    assert text.count("coefficients = array<i64:") == 4
    assert 'relation = "mod"' in text
    assert "modulus = 8 : i64" in text


def test_nonlinear_shape_guards_are_exact_witnesses_not_presburger_facts():
    system = NonlinearWitnessSystem(
        symbols=("B", "M", "N"),
        constraints=(
            NonlinearWitnessConstraint(
                "eq",
                (
                    PolynomialTerm(1, (1, 1, 0)),
                    PolynomialTerm(-1, (0, 0, 1)),
                ),
            ),
        ),
    )
    fn = GraphIRFunction("product_guard")
    attach_nonlinear_witness_system(fn, system)

    mlir = fn.to_mlir()
    assert "tessera.nonlinear_shape_guards" in mlir
    assert "powers = array<i64: 1, 1, 0>" in mlir
    assert system.digest in mlir
    assert system.check_witness({"B": 2, "M": 8, "N": 16}) is True
    assert system.check_witness({"B": 2, "M": 8}) is None
    assert system.check_witness({"B": 2, "M": 7, "N": 16}) is False
    assert "tessera.presburger_constraints" not in mlir
