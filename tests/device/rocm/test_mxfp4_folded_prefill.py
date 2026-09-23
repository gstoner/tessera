"""Exact-device proof for the opted-in gfx1201 folded MXFP4 ABI."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import (
    GFX_MXFP4_W4A8_FOLDED_SAFE_EPILOGUE_ABI,
    package_mxfp4_folded_prefill, prepare_folded_weights,
)
from tessera.compiler.rocm_mxfp4_folded_carrier import (
    package_folded_scaled_wmma_target_ir,
)
from tessera.compiler.rocm_mxfp4_folded_frontend import (
    compile_folded_scaled_matmul,
)
from tests._support import rocm_isa


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize(
    "shape",
    [
        (65, 48, 64),
        (257, 80, 192),
        (256, 80, 128),
        (256, 5120, 8704),
        (1024, 17408, 5120),
    ],
)
def test_frontend_folded_carrier_broad_and_prefill_shapes(
    shape: tuple[int, int, int],
) -> None:
    """Exercise an authored Graph call, not a replayed fixture."""
    assert rt._rocm_live_arch() == "gfx1201"
    tessera_opt = os.environ.get("TESSERA_OPT")
    assert tessera_opt, "frontend proof requires TESSERA_OPT"
    m, n, k = shape
    a = np.full((m, k), 0x38, dtype=np.uint8)  # E4M3 +1
    a_scale = np.ones(m, dtype=np.float32)
    codes = np.ones((n, k), dtype=np.uint8)  # E2M1 +0.5
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert folded.lossless and folded.inexact_value_count == 0
    # Independent exact K32 sample; the full oracle is analytic for this
    # uniform payload, avoiding a prohibitively large CPU GEMM.
    exact_sample = mx.exact_weights(codes[:2], scales[:, :2])
    np.testing.assert_array_equal(exact_sample, np.full((2, k), 0.5))
    program = compile_folded_scaled_matmul(
        a, a_scale, folded, tessera_opt=Path(tessera_opt),
        allow_approximate=True,
    )
    receipt = program.route_receipt
    package = program.package
    assert package.descriptor.provenance["staging_policy"] == "unconditional_k64"
    assert receipt["schedule_hash"] == package.descriptor.provenance["schedule_hash"]
    assert receipt["abi_id"] == package.descriptor.abi_id
    assert receipt["hsaco_sha256"] == hashlib.sha256(package.image.payload).hexdigest()
    assert receipt["hsaco_sha256"] == package.image.payload_digest
    assert receipt["artifact_image_digest"] == package.image.image_digest
    assert receipt["fold_lossless"] is True
    assert receipt["fold_inexact_value_count"] == 0
    assert receipt["numeric_policy"] == "folded_row_reference_explicit_approximate"
    assert package.target_ir.count("tessera_rocm.scaled_wmma_gemm") == 1
    assert 'physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1"' in package.target_ir
    assert 'k_step_schedule = "isolated_k_stage"' in package.target_ir
    assert receipt["selected_schedule"] == {
        "block_m": 256, "block_n": 64, "block_k": 64,
        "tile_m_per_wave": 4, "tile_n_per_wave": 2,
    }
    rocm_isa.assert_selected(
        package.image.payload, chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8",
        what="frontend folded MXFP4 prefill",
    )
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact,
        {"buffers": {
            "a": a, "b_folded": folded.weight_bytes,
            "a_scale": a_scale, "row_reference": folded.row_reference,
            "output": output,
        }, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str,
    )
    np.testing.assert_array_equal(
        output, np.full((m, n), k * 0.5, dtype=ml_dtypes.bfloat16),
    )


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_frontend_folded_carrier_uses_declared_lossy_oracle() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    tessera_opt = os.environ.get("TESSERA_OPT")
    assert tessera_opt, "frontend proof requires TESSERA_OPT"
    m, n, k = 256, 80, 128
    codes = np.zeros((n, k), dtype=np.uint8)
    codes[:, :32] = 1
    codes[:, 32:64] = 2
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    scales[0, :] = 116
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert not folded.lossless and folded.inexact_value_count > 0
    a = np.full((m, k), 0x38, dtype=np.uint8)
    a_scale = np.ones(m, dtype=np.float32)
    program = compile_folded_scaled_matmul(
        a, a_scale, folded, tessera_opt=Path(tessera_opt),
        allow_approximate=True,
    )
    assert program.route_receipt["fold_lossless"] is False
    assert program.route_receipt["fold_inexact_value_count"] > 0
    package = program.package
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact,
        {"buffers": {
            "a": a, "b_folded": folded.weight_bytes,
            "a_scale": a_scale, "row_reference": folded.row_reference,
            "output": output,
        }, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"], json.dumps(result, default=str)
    approximate = mx.folded_weights(folded).sum(axis=1)
    exact = mx.exact_weights(codes, scales).sum(axis=1)
    assert np.any(approximate != exact)
    np.testing.assert_array_equal(
        output,
        np.broadcast_to(
            approximate.astype(ml_dtypes.bfloat16), (m, n),
        ),
    )


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("shape", [(65, 48, 64), (256, 80, 128)])
def test_folded_prefill_matches_its_declared_approximate_oracle(
    shape: tuple[int, int, int],
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = shape
    a = np.full((m, k), 0x38, dtype=np.uint8)  # E4M3 +1
    a_scale = np.ones(m, dtype=np.float32)
    codes = np.zeros((n, k), dtype=np.uint8)
    codes[:, :32] = 1
    if k > 64:
        codes[:, 32:64] = 2
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    scales[0, :] = 116  # shifted first group underflows in E4M3
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert not folded.lossless and folded.inexact_value_count > 0
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
    )
    assert package.descriptor.provenance["staging_policy"] == "unconditional_k64"
    rocm_isa.assert_selected(
        package.image.payload, chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8",
        what="folded MXFP4 BM256/TM4 prefill",
    )
    assert package.descriptor.provenance["fold_inexact_value_count"] > 0
    buffers = {
        "a": a, "b_folded": folded.weight_bytes,
        "a_scale": a_scale, "row_reference": folded.row_reference,
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str
    )
    expected = np.full((m, n), 32 if k > 64 else 0, dtype=ml_dtypes.bfloat16)
    np.testing.assert_array_equal(buffers["output"], expected)
    exact = mx.exact_weights(codes, scales)
    assert np.any(exact != mx.folded_weights(folded))


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("reference_code,activation_scale", [
    (127, 1.0),
    (254, 2.0 ** -127),
])
def test_safe_folded_epilogue_is_exact_and_binds_activation_bytes(
    reference_code: int, activation_scale: float,
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = 65, 48, 64
    codes = np.ones((n, k), dtype=np.uint8)
    scales = np.full((k // 32, n), reference_code, dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert folded.lossless
    a_scale = np.full(m, activation_scale, dtype=np.float32)
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
        safe_epilogue_scales=a_scale,
    )
    assert package.descriptor.abi_id == GFX_MXFP4_W4A8_FOLDED_SAFE_EPILOGUE_ABI
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        target_ir=package.target_ir,
    )
    buffers = {
        "a": np.full((m, k), 0x38, dtype=np.uint8),
        "b_folded": folded.weight_bytes,
        "a_scale": a_scale,
        "row_reference": folded.row_reference,
        "output": output,
    }
    result = rt.launch(
        artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"], json.dumps(result, default=str)
    np.testing.assert_array_equal(
        output, np.full((m, n), 32.0, dtype=ml_dtypes.bfloat16),
    )
    changed = dict(buffers, a_scale=a_scale.copy())
    changed["a_scale"][0] *= 2
    rejected = rt.launch(
        artifact, {"buffers": changed, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert not rejected["ok"]
    assert "scale certificate" in json.dumps(rejected, default=str)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_folded_prefill_rejects_changed_load_time_payload() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = 65, 48, 64
    codes = np.ones((n, k), dtype=np.uint8)
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
    )
    changed = folded.weight_bytes.copy()
    changed[0, 0] ^= 1
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(artifact, {
        "buffers": {
            "a": np.ones((m, k), dtype=np.uint8),
            "b_folded": changed,
            "a_scale": np.ones(m, dtype=np.float32),
            "row_reference": folded.row_reference,
            "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
        },
        "scalars": {"M": m, "N": n, "K": k},
    })
    assert not result["ok"]
    assert "weight_sha256" in json.dumps(result, default=str)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_folded_prefill_combines_canceling_scales_before_accumulator() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = 65, 48, 32
    codes = np.full((n, k), 2, dtype=np.uint8)
    scales = np.full((1, n), 254, dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert folded.lossless
    assert np.all(folded.row_reference == 254)
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
    )
    assert package.descriptor.provenance["staging_policy"] == "guarded_k32_tail"
    buffers = {
        "a": np.full((m, k), 0x38, dtype=np.uint8),
        "b_folded": folded.weight_bytes,
        "a_scale": np.full(m, np.float32(2.0 ** -127), dtype=np.float32),
        "row_reference": folded.row_reference,
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"], json.dumps(result, default=str)
    np.testing.assert_array_equal(
        buffers["output"], np.full((m, n), 32, dtype=ml_dtypes.bfloat16),
    )


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("case", ("zero_partial", "finite_after_overflow"))
def test_folded_prefill_extreme_scale_product(case: str) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = 65, 48, 64
    codes = np.zeros((n, k), dtype=np.uint8)
    scales = np.full((2, n), 254, dtype=np.uint8)
    if case == "finite_after_overflow":
        codes[:, :32] = 1  # E2M1 0.5; delta 8 folds to E4M3 2^-9.
        scales[0, :] = 246
        activation = 0x01  # E4M3 2^-9.
        activation_scale = np.float32(2.0 ** 9)
        expected = np.float32(2.0 ** 123)
    else:
        activation = 0x38
        activation_scale = np.float32(2.0 ** 127)
        expected = np.float32(0)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    assert folded.lossless and np.all(folded.row_reference == 254)
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
    )
    buffers = {
        "a": np.full((m, k), activation, dtype=np.uint8),
        "b_folded": folded.weight_bytes,
        "a_scale": np.full(m, activation_scale, dtype=np.float32),
        "row_reference": folded.row_reference,
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"], json.dumps(result, default=str)
    np.testing.assert_array_equal(
        buffers["output"], np.full((m, n), expected, dtype=ml_dtypes.bfloat16),
    )


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_folded_graph_pipeline_materializes_and_executes() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    tessera_opt = os.environ.get("TESSERA_OPT")
    assert tessera_opt, "folded Graph pipeline proof requires TESSERA_OPT"
    fixture = (
        Path(__file__).resolve().parents[3]
        / "tests/tessera-ir/phase2/e2e_folded_mxfp4_rocm_target.mlir"
    )
    common = [
        tessera_opt, "--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
    ]
    tile_ir = subprocess.run(
        [*common, str(fixture)], check=True, capture_output=True, text=True,
    ).stdout
    target_ir = subprocess.run(
        [*common, "--lower-tile-to-rocm=arch=gfx1201", str(fixture)],
        check=True, capture_output=True, text=True,
    ).stdout
    m, n, k = 65, 48, 64
    codes = np.ones((n, k), dtype=np.uint8)  # E2M1 0.5, E8M0 unit scale.
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), np.full((2, n), 127, dtype=np.uint8),
        allow_approximate=True,
    )
    assert folded.lossless
    package = package_folded_scaled_wmma_target_ir(
        tile_ir, target_ir, folded, allow_approximate=True,
    )
    assert package.descriptor.provenance["schedule_hash"]
    assert package.descriptor.provenance["physical_contract"] == (
        "rocm_mxfp4_w4a8_folded_prefill_v1"
    )
    rocm_isa.assert_selected(
        package.image.payload, chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8",
        what="generic folded MXFP4 W4A8 route",
    )
    buffers = {
        "a": np.full((m, k), 0x38, dtype=np.uint8),
        "b_folded": folded.weight_bytes,
        "a_scale": np.ones(m, dtype=np.float32),
        "row_reference": folded.row_reference,
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"], json.dumps(result, default=str)
    np.testing.assert_array_equal(
        buffers["output"], np.full((m, n), 32, dtype=ml_dtypes.bfloat16),
    )
