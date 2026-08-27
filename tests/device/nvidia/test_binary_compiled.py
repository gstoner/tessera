"""Exact-SM120 compiler-emitted binary arithmetic certificates."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _artifact(op_name: str):
    from tessera import runtime as rt

    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_binary_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["a", "b"],
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": ["a", "b"],
            "kwargs": {},
        }],
    })


def _any(rng, shape):
    return (rng.standard_normal(shape) * 1.5).astype(np.float32)


def _positive(rng, shape):
    return (rng.random(shape) * 4.0 + 0.05).astype(np.float32)


def _sample(op_name, rng, shape):
    if op_name == "tessera.pow":
        return _positive(rng, shape), _any(rng, shape) * np.float32(0.5)
    if op_name in {"tessera.div", "tessera.mod", "tessera.floor_div"}:
        divisor = _positive(rng, shape)
        divisor *= rng.choice((-1.0, 1.0), size=shape).astype(np.float32)
        return _any(rng, shape), divisor
    return _any(rng, shape), _any(rng, shape)


_REFERENCES = {
    "tessera.add": np.add,
    "tessera.sub": np.subtract,
    "tessera.mul": np.multiply,
    "tessera.div": np.divide,
    "tessera.pow": np.power,
    "tessera.maximum": np.maximum,
    "tessera.minimum": np.minimum,
    "tessera.mod": np.mod,
    "tessera.floor_div": np.floor_divide,
}


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("op_name", tuple(_REFERENCES))
@pytest.mark.parametrize("storage,tolerance", (
    ("f32", 2.0e-5), ("f16", 5.0e-3), ("bf16", 4.0e-2),
))
def test_sm120_binary_math_matches_fp32_oracle(op_name, storage, tolerance):
    rt = require_nvidia_mma_runtime()
    dtype = np.float32 if storage == "f32" else np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(1200 + len(op_name) + len(storage))
    a, b = _sample(op_name, rng, (3, 5, 17))
    a = a.astype(dtype)
    b = b.astype(dtype)
    result = rt.launch(_artifact(op_name), (a, b))
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_gpu"
    assert result["compiler_path"] == "nvidia_binary_compiled"
    with np.errstate(all="ignore"):
        expected = _REFERENCES[op_name](
            a.astype(np.float32), b.astype(np.float32)
        ).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(result["output"], np.float32), expected,
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_sm120_minmax_propagate_nan_and_order_signed_zero():
    rt = require_nvidia_mma_runtime()
    a = np.array([0.0, -0.0, np.nan, 1.0], np.float32)
    b = np.array([-0.0, 0.0, 1.0, np.nan], np.float32)
    for op_name, tie_sign in (
        ("tessera.maximum", False), ("tessera.minimum", True),
    ):
        result = rt.launch(_artifact(op_name), (a, b))
        assert result["ok"] is True, result.get("reason")
        output = np.asarray(result["output"], np.float32)
        np.testing.assert_array_equal(np.isnan(output), [False, False, True, True])
        np.testing.assert_array_equal(np.signbit(output[:2]), [tie_sign, tie_sign])


def test_nvidia_binary_contract_rejects_shape_and_dtype_mismatch_before_launch():
    from tessera import runtime as rt

    with pytest.raises(ValueError, match="matching shapes"):
        rt._execute_nvidia_compiled_binary(
            _artifact("tessera.add"),
            (np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32)),
        )
    with pytest.raises(ValueError, match="matching dtypes"):
        rt._execute_nvidia_compiled_binary(
            _artifact("tessera.add"),
            (np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float16)),
        )
