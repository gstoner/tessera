"""Public E2E-REAL-6C attention reverse authority and exact-target proof."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tessera as ts
from tessera._jit_boundary import TesseraJitError
from tessera.compiler.attention_contract import (
    reference_attention_backward_split_reduced,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt


@ts.jit(target="x86", autodiff="reverse", wrt=("q", "key", "value"))
def _x86_attention(q, key, value):
    return ts.ops.gqa_attention(
        q,
        key,
        value,
        num_query_heads=4,
        num_kv_heads=2,
        scale=0.25,
        causal=True,
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("q", "key", "value"))
def _rocm_attention(q, key, value):
    return ts.ops.gqa_attention(
        q,
        key,
        value,
        num_query_heads=4,
        num_kv_heads=2,
        scale=0.25,
        causal=True,
    )


def _inputs(dtype):
    rng = np.random.default_rng(20260817)
    q = rng.normal(0.0, 0.2, (1, 4, 9, 16)).astype(dtype)
    key = rng.normal(0.0, 0.2, (1, 2, 11, 16)).astype(dtype)
    value = rng.normal(0.0, 0.2, key.shape).astype(dtype)
    dout = rng.normal(0.0, 0.2, q.shape).astype(dtype)
    return q, key, value, dout


def _assert_authority(proof, *, target_consumer: str) -> None:
    assert proof["implementation"] == "family_plugin"
    assert proof["frontend_authority"] == "tracer"
    assert proof["family"] == "attention_backward"
    assert proof["schedule_consumer"] == "schedule.attention_backward"
    assert proof["tile_consumer"] == "tile.attention_backward_kernel"
    assert proof["target_consumer"] == target_consumer
    for field in (
        "source_graph_ir_digest",
        "schedule_artifact_hash",
        "tile_program_digest",
        "native_image_digest",
        "artifact_hash",
    ):
        assert len(proof[field]) == 64


def test_public_x86_attention_vjp_consumes_exact_scheduled_package() -> None:
    from tessera import runtime as rt

    if find_tessera_opt() is None or not rt._x86_elementwise_available():
        pytest.skip("production tessera-opt and AVX-512 image are required")
    q, key, value, dout = _inputs(np.float32)
    actual = _x86_attention.native_backward(
        q, key, value, out_cotangents=dout
    )
    expected = reference_attention_backward_split_reduced(
        dout,
        q,
        key,
        value,
        split_count=2,
        scale=0.25,
        causal=True,
    )
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-4, atol=4e-5)
    _assert_authority(
        _x86_attention.last_backward_execution,
        target_consumer="x86.avx512_attention_backward",
    )
    assert _x86_attention.last_frontend_differential.contract[
        "permitted_effect_ops"
    ] == ["tessera.gqa_attention"]


@pytest.mark.compiler_rocm
@pytest.mark.hardware_rocm
def test_public_gfx1151_attention_vjp_consumes_prebuilt_program() -> None:
    from tessera import runtime as rt

    if find_tessera_opt() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("production tessera-opt and a WSL-visible gfx1151 are required")
    q, key, value, dout = _inputs(np.float16)
    actual = _rocm_attention.native_backward(
        q, key, value, out_cotangents=dout
    )
    expected = reference_attention_backward_split_reduced(
        dout,
        q,
        key,
        value,
        split_count=2,
        scale=0.25,
        causal=True,
    )
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=3e-2, atol=4e-3)
    _assert_authority(
        _rocm_attention.last_backward_execution,
        target_consumer="rocm.gfx1151_attention_backward_program",
    )
    assert _rocm_attention.last_frontend_differential.contract[
        "permitted_effect_ops"
    ] == ["tessera.gqa_attention"]


def test_attention_plugin_fails_closed_without_tracer_lineage() -> None:
    from tessera.compiler.graph_ir import IROp
    from tessera.compiler.native_vjp_plugins import execute_native_vjp_family

    op = IROp(
        result="out",
        op_name="tessera.flash_attn",
        operands=["%q", "%key", "%value"],
        operand_types=["tensor<1x2x4x16xf32>"] * 3,
        kwargs={"causal": True},
    )
    arrays = [np.zeros((1, 2, 4, 16), np.float32) for _ in range(3)]
    with pytest.raises(TesseraJitError, match="tracer-owned source Graph IR"):
        execute_native_vjp_family(
            source=op,
            target="x86",
            ordered_inputs=arrays,
            arg_names=("q", "key", "value"),
            source_arg_names=("q", "key", "value"),
            out_cotangents=arrays[0],
            wrt_names=("q",),
            source_graph_ir=None,
        )


def test_attention_package_rejects_stale_parent_and_tile_lineage() -> None:
    from tessera.compiler.graph_ir import IROp
    from tessera.compiler.native_attention_vjp import (
        build_native_attention_vjp_package,
    )

    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt is required")
    q, key, value, dout = _inputs(np.float32)
    source = IROp(
        result="out",
        op_name="tessera.flash_attn",
        operands=["%q", "%key", "%value"],
        operand_types=[
            "tensor<1x4x9x16xf32>",
            "tensor<1x2x11x16xf32>",
            "tensor<1x2x11x16xf32>",
        ],
        kwargs={"scale": 0.25, "causal": True},
    )
    package = build_native_attention_vjp_package(
        source_graph_ir="module { /* traced attention parent */ }",
        source=source,
        target="x86",
        ordered_inputs=(q, key, value),
        arg_names=("q", "key", "value"),
        source_arg_names=("q", "key", "value"),
        out_cotangent=dout,
    )
    with pytest.raises(ValueError, match="source Graph digest is stale"):
        replace(package, source_graph_ir_digest="0" * 64).validate()
    with pytest.raises(ValueError, match="exact Tile artifact"):
        replace(
            package, native=replace(package.native, tile_ir="module {}")
        ).validate()
