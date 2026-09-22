"""Owning-device proof for exact scalar and WMMA gfx1201 MXFP4 W4A8."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_native import (
    package_mxfp4_w4a8_exact,
    package_mxfp4_w4a8_wmma,
    package_scaled_wmma_target_ir,
)
from tests._support import rocm_isa
from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import (
    Case,
    _logical_inputs,
    _sampled_exact_reference,
    _tessera_engine,
)


def _exact_inputs(
    shape: tuple[int, int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    m, n, k = shape
    rng = np.random.default_rng(1201 + m + n + k)
    # Integer-valued E4M3 inputs and power-of-two scales make every FP32
    # product/sum exact; equality after BF16 rounding then catches layout,
    # nibble, scale-plane, and accumulation-boundary defects without a loose
    # floating tolerance hiding them.
    a_values = rng.integers(-4, 5, size=(m, k)).astype(np.float32)
    a_e4m3 = a_values.astype(ml_dtypes.float8_e4m3fn)
    a_raw = np.ascontiguousarray(a_e4m3.view(np.uint8))
    a_scale = np.exp2(rng.integers(-1, 2, size=m)).astype(np.float32)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    packed_checkpoint = mx.pack_e2m1_codes(codes)
    b_scale = rng.integers(124, 131, size=(k // 32, n), dtype=np.uint8)
    b_scale[0, ::7] = 0  # reserved zero-block spelling is executable semantics
    expected = (
        (a_e4m3.astype(np.float32) * a_scale[:, None])
        @ mx.exact_weights(codes, b_scale).T
    ).astype(ml_dtypes.bfloat16)
    return {
        "a": a_raw,
        "b_packed": np.ascontiguousarray(packed_checkpoint.T),
        "a_scale": a_scale,
        "b_scale": b_scale,
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }, expected


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("shape", [(17, 19, 64), (32, 32, 128)])
@pytest.mark.parametrize(
    "package_factory,instruction",
    [
        pytest.param(package_mxfp4_w4a8_exact, None, id="scalar-oracle"),
        pytest.param(
            package_mxfp4_w4a8_wmma,
            "v_wmma_f32_16x16x16_fp8_fp8",
            id="wmma-production",
        ),
    ],
)
def test_exact_mxfp4_w4a8_package_executes_on_gfx1201(
    shape: tuple[int, int, int], package_factory, instruction: str | None
) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = shape
    buffers, expected = _exact_inputs(shape)

    package = package_factory(m, n, k)
    weight_layout = str(package.descriptor.provenance["weight_layout"])
    buffers["b_packed"] = mx.convert_weight_layout(
        np.ascontiguousarray(buffers["b_packed"].T),
        source=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
        destination=weight_layout,
    )
    if instruction is not None:
        rocm_isa.assert_selected(
            package.image.payload,
            chip="gfx1201",
            pattern=r"v_wmma_f32_16x16x16_\w+",
            require=instruction,
            what="exact MXFP4 W4A8",
        )
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact,
        {
            "buffers": buffers,
            "scalars": {"M": m, "N": n, "K": k},
        },
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str
    )
    np.testing.assert_array_equal(buffers["output"], expected)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 48, 64),
        (5, 80, 64),
        (64, 48, 128),
        (65, 48, 64),
        (128, 80, 64),
        (65, 48, 8704),
        (128, 48, 8704),
        (128, 5120, 64),
        (64, 5120, 64),
    ],
)
def test_fragment_abi_boundary_and_ragged_tiles_execute_on_gfx1201(
    shape: tuple[int, int, int],
) -> None:
    """Cover skinny-M boundaries and N tiles that are ragged to larger kernels."""
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = shape
    buffers, expected = _exact_inputs(shape)
    package = package_mxfp4_w4a8_wmma(
        m,
        n,
        k,
        weight_layout=mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    )
    buffers["b_packed"] = mx.convert_weight_layout(
        np.ascontiguousarray(buffers["b_packed"].T),
        source=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
        destination=mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    )
    assert package.descriptor.provenance["weight_layout"] == (
        mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    )
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact,
        {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}},
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str
    )
    np.testing.assert_array_equal(buffers["output"], expected)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_fragment_decode_launch_is_hip_graph_capture_safe() -> None:
    """Loading/conversion/allocation happen before capture; launch only enqueues."""
    assert rt._rocm_live_arch() == "gfx1201"
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    pointer = ctypes.c_void_p
    hip.hipStreamCreate.argtypes = [ctypes.POINTER(pointer)]
    hip.hipStreamBeginCapture.argtypes = [pointer, ctypes.c_int]
    hip.hipStreamEndCapture.argtypes = [pointer, ctypes.POINTER(pointer)]
    hip.hipGraphInstantiate.argtypes = [
        ctypes.POINTER(pointer), pointer, ctypes.POINTER(pointer),
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    hip.hipGraphLaunch.argtypes = [pointer, pointer]

    stream, graph, executable = pointer(), pointer(), pointer()
    assert hip.hipStreamCreate(ctypes.byref(stream)) == 0
    case = Case("decode", 5, 48, 64)
    inputs = _logical_inputs(case)
    engine = _tessera_engine(hip, case, inputs, 1, None, None, stream)
    try:
        assert hip.hipStreamBeginCapture(stream, 0) == 0
        engine.launch()
        assert hip.hipStreamEndCapture(stream, ctypes.byref(graph)) == 0
        assert hip.hipGraphInstantiate(
            ctypes.byref(executable), graph, None, None, 0
        ) == 0
        assert hip.hipGraphLaunch(executable, stream) == 0
        assert hip.hipStreamSynchronize(stream) == 0
        rows, cols, expected = _sampled_exact_reference(case, inputs)
        output = engine.copies[0].download(engine.output_index)
        np.testing.assert_array_equal(output[np.ix_(rows, cols)], expected)
    finally:
        if executable.value:
            hip.hipGraphExecDestroy(executable)
        if graph.value:
            hip.hipGraphDestroy(graph)
        engine.close()
        hip.hipStreamDestroy(stream)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_fragment_decode_does_not_store_into_poisoned_rows_after_m() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    case = Case("decode", 5, 48, 64)
    inputs = _logical_inputs(case)
    # The raw device allocation is deliberately larger than the logical
    # descriptor. Any unguarded row store changes the poison suffix.
    inputs["output"] = np.full((16, case.n), 123.0, dtype=ml_dtypes.bfloat16)
    engine = _tessera_engine(hip, case, inputs, 1, None, None)
    poison = inputs["output"].copy()
    bundle = engine.copies[0]
    try:
        assert hip.hipMemcpy(
            bundle.device[4],
            poison.ctypes.data_as(ctypes.c_void_p),
            int(poison.nbytes),
            1,
        ) == 0
        engine.launch()
        assert hip.hipDeviceSynchronize() == 0
        output = bundle.download(4)
        np.testing.assert_array_equal(
            output[case.m:].view(np.uint16), poison[case.m:].view(np.uint16)
        )
    finally:
        engine.close()


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_generic_scaled_matmul_pipeline_materializes_and_executes_exact_abi() -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    root = Path(__file__).resolve().parents[3]
    fixture = root / "tests/tessera-ir/phase2/e2e_scaled_matmul_rocm_target.mlir"
    tessera_opt = os.environ.get("TESSERA_OPT")
    assert tessera_opt, "generic route proof requires TESSERA_OPT"
    common = [
        tessera_opt,
        "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile",
    ]
    tile_ir = subprocess.run(
        [*common, str(fixture)], check=True, capture_output=True, text=True
    ).stdout
    target_ir = subprocess.run(
        [*common, "--lower-tile-to-rocm=arch=gfx1201", str(fixture)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    package = package_scaled_wmma_target_ir(tile_ir, target_ir)
    rocm_isa.assert_selected(
        package.image.payload,
        chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8",
        what="generic exact MXFP4 W4A8 route",
    )
    shape = (17, 19, 64)
    buffers, expected = _exact_inputs(shape)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    result = rt.launch(
        artifact,
        {
            "buffers": buffers,
            "scalars": {"M": 17, "N": 19, "K": 64},
        },
    )
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(
        result, default=str
    )
    np.testing.assert_array_equal(buffers["output"], expected)
