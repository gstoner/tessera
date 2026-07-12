"""Exact-target x86 proofs for the curated conformance matrix.

Every helper launches a declared native x86 execution-matrix lane and checks
both its provenance and its output against numpy. Composition remains multiple
native kernels; no fused-kernel claim is made.
"""

from __future__ import annotations

import numpy as np
import pytest


def _runtime_or_skip():
    from tessera import runtime as rt

    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, path: str, op_name: str, operands: tuple[str, ...]):
    return rt.RuntimeArtifact(metadata={
        "target": "x86",
        "compiler_path": path,
        "executable": True,
        "execution_kind": "native_cpu",
        "execution_mode": "cpu_avx512",
        "arg_names": list(operands),
        "output_name": "out",
        "ops": [{
            "op_name": op_name,
            "result": "out",
            "operands": list(operands),
            "kwargs": {},
        }],
    })


def _launch(rt, path: str, op_name: str, operands: tuple[str, ...], args):
    result = rt.launch(_artifact(rt, path, op_name, operands), args)
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == path
    assert result["execution_kind"] == "native_cpu"
    return np.asarray(result["output"], dtype=np.float32)


def _matmul(rt, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _launch(
        rt, "x86_matmul_family_compiled", "tessera.matmul", ("a", "b"),
        (a, b),
    )


def _relu(rt, x: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(x)
    return _launch(
        rt, "x86_binary_compiled", "tessera.maximum", ("x", "zero"),
        (x, zeros),
    )


def _softmax(rt, x: np.ndarray) -> np.ndarray:
    return _launch(
        rt, "x86_softmax_compiled", "tessera.softmax", ("x",), (x,),
    )


def _inputs(seed: int = 20260712):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((16, 24)).astype(np.float32),
        rng.standard_normal((24, 12)).astype(np.float32),
    )


def test_x86_matmul_matches_numpy_with_native_provenance() -> None:
    rt = _runtime_or_skip()
    a, b = _inputs()
    np.testing.assert_allclose(_matmul(rt, a, b), a @ b, rtol=1e-3, atol=1e-3)


def test_x86_relu_matches_numpy_with_native_provenance() -> None:
    rt = _runtime_or_skip()
    a, b = _inputs()
    x = a @ b
    np.testing.assert_allclose(_relu(rt, x), np.maximum(x, 0), rtol=0, atol=0)


def test_x86_matmul_relu_composition_matches_numpy() -> None:
    rt = _runtime_or_skip()
    a, b = _inputs()
    output = _relu(rt, _matmul(rt, a, b))
    np.testing.assert_allclose(output, np.maximum(a @ b, 0), rtol=1e-3, atol=1e-3)


def test_x86_matmul_softmax_composition_matches_numpy() -> None:
    rt = _runtime_or_skip()
    a, b = _inputs()
    output = _softmax(rt, _matmul(rt, a, b))
    scores = a @ b
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = exp / exp.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(output, expected, rtol=2e-4, atol=2e-5)
