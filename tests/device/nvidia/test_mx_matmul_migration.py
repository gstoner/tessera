"""Exact-SM120 differential: the retired Graph-authored MX packager vs the Schedule projection.

E2E-REAL-6 (sync ``E2E-REAL-6-sm120-mx-matmul-2026-11-04``). Each FP6/MXFP4
case builds the retired packager (Python-authored Tile IR, launch fields read
from the Graph; ``tests/_support/nvidia_lowp_matmul_baseline.py``, the
Decision #31(a) oracle) and the production package from the replayed native
Graph -> Schedule -> Tile artifact, launches both on the RTX 5070, and
requires bitwise-equal outputs plus agreement with a NumPy oracle of the
UE8M0 K32 block-scaled product. RTX 5070 evidence only; it transfers to no
other part.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tests._support import nvidia_lowp_matmul_baseline as baseline
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_e2e_spine_native import _decode_ue8m0, _pack_nvfp4
from tests.unit.test_nvidia_mx_matmul_migration import mx_module

pytestmark = pytest.mark.hardware_nvidia
PIPELINE = "tessera-nvidia-pipeline-sm120"


@pytest.fixture(autouse=True)
def _sm120():
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    assert rt._nvidia_device_name() == "sm_120"


def _launch(package, bindings):
    artifact = rt.RuntimeArtifact(
        tile_ir=package.tile_ir, target_ir=package.target_ir, metadata={"target": "nvidia_sm120"},
        native_image=package.image, launch_descriptor=package.descriptor)
    result = rt.launch(rt.RuntimeArtifact.from_json(artifact.to_json()), bindings)
    assert result.get("ok"), result
    assert result.get("execution_kind") == "native_gpu", result


@pytest.mark.parametrize("dtype", ["fp6_e2m3", "fp6_e3m2", "fp4_e2m1"])
@pytest.mark.parametrize("shape", [(16, 8, 64), (19, 13, 69), (7, 5, 31), (33, 17, 129)])
def test_mx_matches_retired_packager_bitwise(dtype, shape):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    m, n, k = shape
    module = mx_module(m, n, k, dtype)
    packages = (baseline.baseline_mx_matmul(module, pipeline_name=PIPELINE),
                native.package_native(module, pipeline_name=PIPELINE))
    assert "tessera.schedule_hash" in packages[1].tile_ir
    value_dtype = {
        "fp6_e2m3": ml_dtypes.float6_e2m3fn,
        "fp6_e3m2": ml_dtypes.float6_e3m2fn,
        "fp4_e2m1": ml_dtypes.float4_e2m1fn,
    }[dtype]
    rng = np.random.default_rng(1_104_2028 + m + n + k)
    a_values = (rng.standard_normal((m, k)) * 1.25).astype(value_dtype)
    b_values = (rng.standard_normal((k, n)) * 1.25).astype(value_dtype)
    a_codes, b_codes = a_values.view(np.uint8), b_values.view(np.uint8)
    if dtype == "fp4_e2m1":
        a_physical, b_physical = _pack_nvfp4(a_codes, 1), _pack_nvfp4(b_codes, 0)
    else:
        a_physical, b_physical = np.ascontiguousarray(a_codes), np.ascontiguousarray(b_codes)
    sk = (k + 31) // 32
    choices = np.asarray([124, 126, 127, 128, 129], np.uint8)
    scale_a = np.ascontiguousarray(choices[(np.arange(m)[:, None] + np.arange(sk)[None, :]) % choices.size])
    scale_b = np.ascontiguousarray(choices[(2 * np.arange(sk)[:, None] + np.arange(n)[None, :]) % choices.size])
    outputs = []
    for package in packages:
        c = np.full((m, n), np.nan, np.float32)
        _launch(package, {"a": a_physical, "b": b_physical, "scale_a": scale_a, "scale_b": scale_b,
                          "c": c, "M": m, "N": n, "K": k})
        outputs.append(c)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    a = a_values.astype(np.float32) * np.repeat(_decode_ue8m0(scale_a), 32, axis=1)[:, :k]
    b = b_values.astype(np.float32) * np.repeat(_decode_ue8m0(scale_b), 32, axis=0)[:k, :]
    np.testing.assert_allclose(outputs[1], a @ b, rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize("dtype", ["fp6_e2m3", "fp4_e2m1"])
def test_mx_projected_package_refuses_a_short_scale_view(dtype):
    m, n, k = 16, 8, 64
    package = native.package_native(mx_module(m, n, k, dtype), pipeline_name=PIPELINE)
    physical_k = k if dtype.startswith("fp6") else k // 2
    artifact = rt.RuntimeArtifact(
        tile_ir=package.tile_ir, target_ir=package.target_ir, metadata={"target": "nvidia_sm120"},
        native_image=package.image, launch_descriptor=package.descriptor)
    result = rt.launch(artifact, {
        "a": np.zeros((m, physical_k), np.uint8), "b": np.zeros((physical_k, n), np.uint8),
        "scale_a": np.full((m, 1), 127, np.uint8), "scale_b": np.full((2, n), 127, np.uint8),
        "c": np.zeros((m, n), np.float32), "M": m, "N": n, "K": k})
    assert result["ok"] is False
    assert result["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"
