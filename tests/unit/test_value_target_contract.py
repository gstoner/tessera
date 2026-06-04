from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.value_target_contract import (
    BACKEND_VALUE_TARGET_CONTRACTS,
    REQUIRED_VALUE_CALL_ATTRS,
    VALUE_TARGET_HANDOFF_STEPS,
    backend_family_for_target,
    is_complete_value_call_record,
    missing_value_call_attrs,
    value_contract_for_target,
)


def _complete_call(op: str) -> dict[str, object]:
    return {
        "op": op,
        "op_kind": "batched_gemm",
        "symbol": "backend_symbol",
        "status": "executable",
        "abi": "backend_abi",
        "dtype": "f32",
        "framework": "backend_framework",
    }


def test_stage18_handoff_chain_is_backend_neutral() -> None:
    assert VALUE_TARGET_HANDOFF_STEPS == (
        "graph_ir_op",
        "registered_tile_op",
        "value_target_ir_call",
        "backend_executor_adapter",
        "numerical_proof",
    )
    assert REQUIRED_VALUE_CALL_ATTRS == (
        "op_kind",
        "symbol",
        "status",
        "abi",
        "dtype",
        "framework",
    )


def test_stage18_backend_kernel_call_mnemonics_are_pinned() -> None:
    assert BACKEND_VALUE_TARGET_CONTRACTS["apple_gpu"].kernel_call_op == (
        "tessera_apple.gpu.kernel_call")
    assert BACKEND_VALUE_TARGET_CONTRACTS["nvidia"].kernel_call_op == (
        "tessera_nvidia.kernel_call")
    assert BACKEND_VALUE_TARGET_CONTRACTS["rocm"].kernel_call_op == (
        "tessera_rocm.kernel_call")


@pytest.mark.parametrize("target,family", [
    ("apple_gpu", "apple_gpu"),
    ("nvidia", "nvidia"),
    ("nvidia_sm90", "nvidia"),
    ("nvidia_sm120", "nvidia"),
    ("rocm", "rocm"),
    ("rocm_gfx942", "rocm"),
])
def test_stage18_target_aliases_resolve_to_backend_family(
        target: str, family: str) -> None:
    assert backend_family_for_target(target) == family
    assert value_contract_for_target(target) is BACKEND_VALUE_TARGET_CONTRACTS[family]


@pytest.mark.parametrize("target,op", [
    ("apple_gpu", "tessera_apple.gpu.kernel_call"),
    ("nvidia_sm90", "tessera_nvidia.kernel_call"),
    ("rocm_gfx942", "tessera_rocm.kernel_call"),
])
def test_stage18_complete_value_call_record_accepts_backend_specific_op(
        target: str, op: str) -> None:
    call = _complete_call(op)
    assert missing_value_call_attrs(call, target=target) == ()
    assert is_complete_value_call_record(call, target=target)


def test_stage18_value_call_record_rejects_wrong_backend_op() -> None:
    call = _complete_call("tessera_apple.gpu.kernel_call")
    assert "op" in missing_value_call_attrs(call, target="nvidia_sm90")
    assert not is_complete_value_call_record(call, target="nvidia_sm90")


def test_stage18_value_call_record_rejects_missing_required_attrs() -> None:
    call = _complete_call("tessera_rocm.kernel_call")
    del call["symbol"]
    call["framework"] = ""
    missing = missing_value_call_attrs(call, target="rocm_gfx942")
    assert missing == ("symbol", "framework")
    assert not is_complete_value_call_record(call, target="rocm_gfx942")


def test_stage18_docs_pin_backend_neutral_chain_and_backend_ops() -> None:
    root = Path(__file__).resolve().parents[2]
    value_contract = (
        root / "docs/spec/VALUE_TARGET_IR_CONTRACT.md"
    ).read_text(encoding="utf-8")
    target_spec = (root / "docs/spec/TARGET_IR_SPEC.md").read_text(
        encoding="utf-8")
    for text in (value_contract, target_spec):
        assert "Graph IR op" in text
        assert "registered Tile op" in text
        assert "value Target IR call" in text
        assert "backend executor adapter" in text
        assert "numerical proof" in text
        assert "tessera_apple.gpu.kernel_call" in text
        assert "tessera_nvidia.kernel_call" in text
        assert "tessera_rocm.kernel_call" in text
