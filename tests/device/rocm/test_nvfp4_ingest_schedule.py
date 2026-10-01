"""Exact gfx1201 Graph/Schedule/Tile proof for NVFP4 checkpoint ingest."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler import rocm_nvfp4_ingest as ingest
from tessera.compiler.rocm_mxfp4_native import package_scaled_wmma_target_ir
from tests._support import rocm_isa


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_nvfp4_gate_up_ingest_runs_graph_schedule_tile_package_on_gfx1201():
    assert rt._rocm_live_arch() == "gfx1201"
    root = Path(__file__).resolve().parents[3]
    fixture = root / "tests/tessera-ir/phase2/e2e_scaled_matmul_rocm_target.mlir"
    tessera_opt = os.environ.get("TESSERA_OPT")
    assert tessera_opt, "native Graph/Schedule proof requires TESSERA_OPT"
    common = [
        tessera_opt, "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile",
    ]
    tile_ir = subprocess.run(
        [*common, str(fixture)], check=True, capture_output=True, text=True
    ).stdout
    target_ir = subprocess.run(
        [*common, "--lower-tile-to-rocm=arch=gfx1201", str(fixture)],
        check=True, capture_output=True, text=True,
    ).stdout
    package = package_scaled_wmma_target_ir(tile_ir, target_ir)
    assert package.image.architecture == "gfx1201"
    assert "tessera.schedule_hash" in package.tile_ir
    assert "tessera.schedule_hash" in package.target_ir
    rocm_isa.assert_selected(
        package.image.payload,
        chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8",
        what="NVFP4-ingested Graph/Schedule/Tile W4A8 route",
    )

    k = 64
    rng = np.random.default_rng(1201_873)
    gate_codes = rng.integers(0, 16, size=(9, k), dtype=np.uint8)
    up_codes = rng.integers(0, 16, size=(10, k), dtype=np.uint8)
    gate_scales = np.asarray(
        np.tile(np.asarray([0.5, 1.0, 0.75, 1.5], np.float32), (9, 1)),
        dtype=ml_dtypes.float8_e4m3fn,
    )
    up_scales = np.asarray(
        np.tile(np.asarray([2.0, 1.0, 1.5, 0.5], np.float32), (10, 1)),
        dtype=ml_dtypes.float8_e4m3fn,
    )
    ingested = ingest.ingest_nvfp4_projections((
        ingest.NVFP4Projection("gate", mx.pack_e2m1_codes(gate_codes),
                               gate_scales, 0.5),
        ingest.NVFP4Projection("up", mx.pack_e2m1_codes(up_codes),
                               up_scales, 2.0),
    ))
    assert ingested.shape == (19, k)
    assert ingested.row_offsets == (0, 9, 19)
    assert [item.global_scale for item in ingested.metadata] == [0.5, 2.0]
    assert ingested.numeric_policy()["lossy_steps"] == [ingest.LOSSY_STEP]

    m, n = 17, 19
    a_values = rng.integers(-4, 5, size=(m, k)).astype(np.float32)
    a_f8 = a_values.astype(ml_dtypes.float8_e4m3fn)
    a_scale = np.exp2(rng.integers(-1, 2, size=m)).astype(np.float32)
    a_raw = np.ascontiguousarray(a_f8.view(np.uint8))
    codes = mx.unpack_e2m1_codes(ingested.packed_codes)
    decoded_b = mx.exact_weights(codes, ingested.scale_exponents)
    expected = ((a_f8.astype(np.float32) * a_scale[:, None]) @ decoded_b.T).astype(
        ml_dtypes.bfloat16
    )
    buffers = {
        "a": a_raw,
        # Generic gfx1201 W4A8 ABI names packed B in [K/2,N] order.
        "b_packed": np.ascontiguousarray(ingested.packed_codes.T),
        "a_scale": a_scale,
        "b_scale": ingested.scale_exponents,
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}}
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str
    )
    np.testing.assert_array_equal(buffers["output"], expected)
