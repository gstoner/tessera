from __future__ import annotations

import os

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_native
from tessera.compiler import scheduled_depth_attention
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt


def _module(*, eps: float = 1.0e-6, sources: int = 7) -> GraphIRModule:
    query = IRType("tensor<8xf32>", ("8",), "fp32")
    source_type = IRType(
        f"tensor<{sources}x3x8xf32>", (str(sources), "3", "8"), "fp32"
    )
    output = IRType("tensor<3x8xf32>", ("3", "8"), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="scheduled_depth_attention",
                args=[IRArg("query", query), IRArg("sources", source_type)],
                result_types=[output],
                body=[
                    IROp(
                        result="result",
                        op_name="tessera.depth_attn",
                        operands=["%query", "%sources"],
                        operand_types=[str(query), str(source_type)],
                        result_type=str(output),
                        kwargs={
                            "eps": eps,
                            "numeric_policy": {
                                "storage": "fp32",
                                "softmax": "fp32",
                                "accum": "fp32",
                            },
                        },
                    )
                ],
                return_values=["%result"],
            )
        ]
    )


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize(
    ("target", "arch", "workgroup"),
    [("x86", "zen5-avx512", 1), ("rocm_gfx1151", "gfx1151", 256)],
)
def test_depth_attention_lowers_to_one_exact_artifact(
    target: str, arch: str, workgroup: int
) -> None:
    artifact = scheduled_depth_attention.lower_scheduled_depth_attention(
        _module(), target=target
    )
    assert artifact.architecture == arch
    assert artifact.workgroup_size == workgroup
    assert artifact.source_count == 7
    assert artifact.rows == 3
    assert artifact.width == 8
    assert artifact.schedule_ir.count("schedule.depth_attention") == 1
    assert artifact.tile_ir.count("tile.depth_attention_kernel") == 1
    assert "tessera.depth_attn" not in artifact.tile_ir
    assert "schedule." not in artifact.tile_ir


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_depth_attention_digest_binds_epsilon_and_shape() -> None:
    baseline = scheduled_depth_attention.lower_scheduled_depth_attention(
        _module(), target="x86"
    )
    different_eps = scheduled_depth_attention.lower_scheduled_depth_attention(
        _module(eps=1.0e-5), target="x86"
    )
    different_shape = scheduled_depth_attention.lower_scheduled_depth_attention(
        _module(sources=9), target="x86"
    )
    assert len({baseline.schedule_digest, different_eps.schedule_digest,
                different_shape.schedule_digest}) == 3


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_depth_attention_schedule_tampering_fails_closed() -> None:
    artifact = scheduled_depth_attention.lower_scheduled_depth_attention(
        _module(), target="x86"
    )
    tampered = artifact.schedule_ir.replace("source_tile = 7", "source_tile = 1")
    with pytest.raises(RuntimeError, match="altered after hashing"):
        scheduled_depth_attention.run_tessera_opt(
            find_tessera_opt(), tampered, "--tessera-schedule-to-tile"
        )


@pytest.mark.parametrize("target", ["rocm_gfx1200", "rocm_gfx1250", "cuda"])
def test_depth_attention_unowned_architectures_fail_closed(target: str) -> None:
    assert not scheduled_depth_attention.supports_scheduled_depth_attention(
        _module(), target=target
    )


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_rocm_package_consumes_exact_depth_attention_artifact(monkeypatch) -> None:
    artifact = scheduled_depth_attention.lower_scheduled_depth_attention(
        _module(), target="rocm_gfx1151"
    )
    monkeypatch.setattr(
        rocm_native,
        "_compile_scheduled_depth_attention_tile_ir",
        lambda tile: (
            ("target", "backend", b"hsaco", "compiler", "toolchain", (), "cold")
            if tile == artifact.tile_ir
            else pytest.fail("depth-attention Tile artifact was resynthesized")
        ),
    )
    package = rocm_native.package_scheduled_depth_attention(
        artifact, pipeline_name="tessera-lower-to-rocm"
    )
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.abi_id == rocm_native.GFX1151_DEPTH_ATTN_F32_ABI
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert package.descriptor.provenance["algorithm"] == (
        "rms_key_online_softmax_stats_v1"
    )
    assert package.descriptor.provenance["merge"] == (
        "max_shifted_pairwise_merge_v1"
    )


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_driver_records_depth_attention_schedule_tile_target_lineage(monkeypatch) -> None:
    monkeypatch.setattr(
        rocm_native,
        "_compile_scheduled_depth_attention_tile_ir",
        lambda tile: (
            "target",
            "backend",
            b"hsaco",
            "compiler",
            "toolchain",
            (),
            "cold",
        ),
    )
    bundle = compile_graph_module(
        _module(),
        source_origin="block-attnres-phase5",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.schedule is not None and "schedule.depth_attention" in bundle.schedule.text
    assert bundle.tile is not None and "tile.depth_attention_kernel" in bundle.tile.text
    assert bundle.target_ir is not None and bundle.target_ir.text == "target"
    assert bundle.lineage_complete
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["work_item"] == (
        "BLOCK-ATTNRES-ROCM-1"
    )


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
def test_gfx1151_depth_attention_matches_reference() -> None:
    from tessera._block_attnres_ops import depth_attn

    bundle = compile_graph_module(
        _module(sources=7),
        source_origin="block-attnres-phase5-exact-device",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    rng = np.random.default_rng(554)
    query = np.ascontiguousarray(rng.normal(size=(8,)), dtype=np.float32)
    sources = np.ascontiguousarray(rng.normal(size=(7, 3, 8)), dtype=np.float32)
    output = np.zeros((3, 8), dtype=np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(
            metadata={"target": "rocm_gfx1151"},
            native_image=bundle.native_image,
            launch_descriptor=bundle.launch_descriptor,
            tile_ir=bundle.tile.text,
            target_ir=bundle.target_ir.text,
        ),
        {"query": query, "sources": sources, "result": output},
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(output, depth_attn(query, sources), rtol=3e-5, atol=3e-5)
