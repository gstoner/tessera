"""Native Apple GPU local f32 top-1 MoE compute vertical slice.

Routing/grouping remains host-side metadata work.  Each nonempty expert block
is a checked native MPS f32 matmul; unsupported contracts explicitly retain the
public NumPy reference path.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt
from tests._support.apple import assert_native_apple_gpu, assert_reference_cpu


@ts.jit(target="apple_gpu")
def _jit_moe(x, experts):
    return ts.ops.moe(x, experts)


def _art(extras: list[str] | None = None):
    extras = extras or []
    names = ["x", "experts", *extras]
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_moe_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.moe", "result": "o", "operands": names,
                 "kwargs": {"extras": extras}}],
    })


def _launch(x, experts, *extras, names: list[str] | None = None):
    result = rt.launch(_art(names), (x, experts, *extras))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "apple_gpu_moe_compiled"
    return np.asarray(result["output"]), result


@pytest.mark.parametrize("shape", [(11, 5, 7), (2, 3, 4, 6)])
def test_local_moe_f32_matches_oracle_for_explicit_top1_routes(shape):
    *lead, in_dim, out_dim = shape
    rng = np.random.default_rng(sum(shape))
    x = rng.standard_normal((*lead, in_dim)).astype(np.float32)
    experts = rng.standard_normal((4, in_dim, out_dim)).astype(np.float32)
    route = rng.integers(-4, 4, size=lead, dtype=np.int64)
    out, _ = _launch(x, experts, route, names=["route"])
    np.testing.assert_allclose(out, ts.ops.moe(x, experts, route=route),
                               rtol=2e-5, atol=2e-5)


def test_local_moe_f32_scores_and_round_robin_match_oracle():
    rng = np.random.default_rng(72)
    x = rng.standard_normal((3, 4, 5)).astype(np.float32)
    experts = rng.standard_normal((3, 5, 6)).astype(np.float32)
    scores = rng.standard_normal((3, 4, 3)).astype(np.float32)
    scored, _ = _launch(x, experts, scores, names=["scores"])
    default, _ = _launch(x, experts)
    np.testing.assert_allclose(scored, ts.ops.moe(x, experts, scores=scores),
                               rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(default, ts.ops.moe(x, experts),
                               rtol=2e-5, atol=2e-5)


def test_local_moe_jit_routes_through_the_apple_gpu_compiler_envelope():
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    experts = np.arange(60, dtype=np.float32).reshape(3, 5, 4)
    np.testing.assert_allclose(np.asarray(_jit_moe(x, experts)), ts.ops.moe(x, experts))
    metadata = _jit_moe.runtime_artifact().metadata
    assert metadata["compiler_path"] == "apple_gpu_mps"
    assert metadata["execution_mode"] == "metal_runtime"
    assert "JIT_COMPILED_TARGET_RUNTIME" in metadata["diagnostics"][0]


def test_local_moe_non_f32_and_strided_route_use_reference_cpu_override():
    x = np.ones((4, 3), np.float64)
    experts = np.ones((2, 3, 5), np.float64)
    out, result = _launch(x, experts)
    assert_reference_cpu(result)
    np.testing.assert_allclose(out, ts.ops.moe(x, experts))

    xf = np.ones((4, 3), np.float32)
    ef = np.ones((2, 3, 5), np.float32)
    route = np.arange(8, dtype=np.int64)[::2]
    assert not route.flags.c_contiguous
    out, result = _launch(xf, ef, route, names=["route"])
    assert_reference_cpu(result)
    np.testing.assert_allclose(out, ts.ops.moe(xf, ef, route=route))


def test_local_moe_execution_matrix_declares_native_lane():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("apple_gpu", "apple_gpu_moe_compiled")
    assert row is not None
    assert row.execution_kind == "native_gpu"


@pytest.mark.hardware_apple_gpu
def test_local_moe_f32_reports_native_gpu_on_metal():
    rng = np.random.default_rng(73)
    x = rng.standard_normal((7, 5)).astype(np.float32)
    experts = rng.standard_normal((3, 5, 4)).astype(np.float32)
    route = np.asarray([0, 2, 1, 2, 0, 1, 2], np.int64)
    out, result = _launch(x, experts, route, names=["route"])
    assert_native_apple_gpu(result, compiler_path="apple_gpu_moe_compiled")
    np.testing.assert_allclose(out, ts.ops.moe(x, experts, route=route),
                               rtol=2e-5, atol=2e-5)
