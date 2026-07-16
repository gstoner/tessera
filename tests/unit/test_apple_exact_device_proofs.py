"""Host-free APPLE-TEST-2 / APPLE-REG-1 proof-ladder drift guards."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._support.apple import assert_native_apple_gpu, assert_native_apple_jit


ROOT = Path(__file__).resolve().parents[2]


def _test_function(node_id: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    relative_path, test_name = node_id.split("::", 1)
    path = ROOT / relative_path
    assert path.is_file(), node_id
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
            return node
    raise AssertionError(f"missing test node {node_id}")


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        current = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        names.add(".".join(reversed(parts)))
    return names


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and (getattr(child.func, "id", None) == name or getattr(child.func, "attr", None) == name)
        for child in ast.walk(node)
    )


def test_native_provenance_assertion_rejects_reference_fallback() -> None:
    with pytest.raises(AssertionError):
        assert_native_apple_gpu({
            "ok": True,
            "execution_kind": "reference_cpu",
            "execution_mode": "metal_runtime",
        })


def test_native_jit_assertion_rejects_a_reference_callable() -> None:
    class ReferenceCallable:
        execution_kind = "reference_cpu"

        @staticmethod
        def runtime_artifact():
            raise AssertionError("reference callable must fail before artifact access")

    with pytest.raises(AssertionError):
        assert_native_apple_jit(ReferenceCallable())


def test_exact_device_proof_registry_connects_runtime_abi_envelope_and_tests() -> None:
    from tessera._apple_gpu_dispatch import APPLE_ABI
    from tessera.compiler.apple_exact_device_proofs import EXACT_DEVICE_PROOFS
    from tessera.compiler.apple_gpu_envelope import runtime_ops
    from tessera.compiler.execution_matrix import lookup

    assert EXACT_DEVICE_PROOFS
    paths = [proof.compiler_path for proof in EXACT_DEVICE_PROOFS]
    assert len(paths) == len(set(paths)), f"duplicate compiler paths: {paths}"
    assert all(proof.cohort for proof in EXACT_DEVICE_PROOFS)
    envelope = runtime_ops()
    for proof in EXACT_DEVICE_PROOFS:
        row = lookup("apple_gpu", proof.compiler_path)
        assert row is not None, proof.compiler_path
        assert row.executable and row.execution_kind == "native_gpu", proof.compiler_path
        assert row.execution_mode == "metal_runtime", proof.compiler_path
        if proof.compiler_envelope:
            assert set(proof.op_names) <= envelope, proof.compiler_path
        assert set(proof.runtime_symbols) <= set(APPLE_ABI), proof.compiler_path

        native = _test_function(proof.native_test)
        assert "pytest.mark.hardware_apple_gpu" in _decorator_names(native), proof.native_test
        assert _calls(native, "assert_native_apple_gpu"), proof.native_test
        fallback = _test_function(proof.fallback_test)
        assert _calls(fallback, "assert_reference_cpu"), proof.fallback_test
