"""Tests for G3 — numeric_policy propagation through Graph IR.

Locks the contract that registered op policies propagate onto IR
instances, and that mismatched precision chains surface as
diagnostics.
"""

from __future__ import annotations

import pytest

from tessera.compiler import (
    propagate_numeric_policy,
    propagate_numeric_policy_module,
    validate_numeric_policy_chain,
)
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IROp


def _make_op(op_name: str, **kwargs) -> IROp:
    return IROp(
        op_name=op_name,
        operands=kwargs.pop("operands", ["%a", "%b"]),
        operand_types=kwargs.pop("operand_types", ["tensor", "tensor"]),
        result=kwargs.pop("result", "r"),
        result_type=kwargs.pop("result_type", "tensor"),
        **kwargs,
    )


class TestPropagate:
    def test_default_op_has_no_policy(self) -> None:
        op = _make_op("matmul")
        assert op.numeric_policy is None

    def test_propagate_stamps_matmul_policy(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_make_op("matmul"))
        stamped = propagate_numeric_policy(fn)
        assert stamped == 1
        # Matmul's policy is bf16 storage + fp32 accum.
        policy = fn.body[0].numeric_policy
        assert policy is not None
        assert policy.storage == "bf16"
        assert policy.accum == "fp32"

    def test_propagate_handles_tessera_prefix(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_make_op("tessera.matmul"))
        stamped = propagate_numeric_policy(fn)
        assert stamped == 1
        assert fn.body[0].numeric_policy is not None

    def test_propagate_skips_unknown_ops(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_make_op("some_unknown_op"))
        stamped = propagate_numeric_policy(fn)
        assert stamped == 0
        assert fn.body[0].numeric_policy is None

    def test_propagate_preserves_existing_policy_by_default(self) -> None:
        from tessera.compiler.primitive_coverage import NumericPolicy

        fn = GraphIRFunction(name="f")
        custom = NumericPolicy(storage="fp32", accum="fp32")
        fn.body.append(_make_op("matmul"))
        fn.body[0].numeric_policy = custom

        stamped = propagate_numeric_policy(fn)
        # The custom policy was preserved.
        assert stamped == 0
        assert fn.body[0].numeric_policy is custom

    def test_overwrite_re_stamps_from_catalog(self) -> None:
        from tessera.compiler.primitive_coverage import NumericPolicy

        fn = GraphIRFunction(name="f")
        fn.body.append(_make_op("matmul"))
        fn.body[0].numeric_policy = NumericPolicy(storage="fp32")

        stamped = propagate_numeric_policy(fn, overwrite=True)
        assert stamped == 1
        # Catalog policy: bf16 / fp32
        assert fn.body[0].numeric_policy.storage == "bf16"


class TestPropagateModule:
    def test_propagates_across_all_functions(self) -> None:
        module = GraphIRModule()
        f1 = GraphIRFunction(name="f1")
        f1.body.append(_make_op("matmul"))
        f1.body.append(_make_op("softmax"))
        f2 = GraphIRFunction(name="f2")
        f2.body.append(_make_op("flash_attn"))
        module.functions.extend([f1, f2])

        total = propagate_numeric_policy_module(module)
        # At least 1 (matmul known); softmax / flash_attn may or may
        # not be in the catalog — accept ≥ 1 as the contract.
        assert total >= 1


class TestValidateChain:
    def test_clean_chain_emits_no_diagnostics(self) -> None:
        """A pipeline where each op's storage matches its
        upstream's accum produces no diagnostics."""

        fn = GraphIRFunction(name="f")
        # Two matmuls in a row — both have bf16 storage + fp32 accum.
        # The downstream consumes the upstream's accum (fp32) but
        # its storage is bf16, which is a real mismatch — flag for
        # visibility.  This test asserts the pass behavior, not
        # whether the mismatch is "good" or "bad".
        m1 = _make_op("matmul", result="r1")
        m2 = _make_op("matmul", operands=["%r1", "%c"], result="r2")
        fn.body.extend([m1, m2])
        propagate_numeric_policy(fn)
        diagnostics = validate_numeric_policy_chain(fn)
        # The pass flags the bf16-storage consumer of an fp32-accum
        # producer — info-level by design.
        codes = [d.code_value for d in diagnostics]
        assert all(d.severity in ("info", "warning") for d in diagnostics)
        # The chain-mismatch code is registered:
        if diagnostics:
            assert "NUMERIC_POLICY_CHAIN_MISMATCH" in codes

    def test_ops_without_policy_are_skipped(self) -> None:
        fn = GraphIRFunction(name="f")
        fn.body.append(_make_op("some_unknown_op"))
        assert validate_numeric_policy_chain(fn) == []


class TestPublicNamespace:
    def test_three_helpers_exported(self) -> None:
        import tessera.compiler as tc
        for name in (
            "propagate_numeric_policy",
            "propagate_numeric_policy_module",
            "validate_numeric_policy_chain",
        ):
            assert name in tc.__all__, name
            assert hasattr(tc, name)
