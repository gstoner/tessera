"""Exact-SM120 differential: retired Graph-authored low-precision matmul packages vs Schedule projections.

E2E-REAL-6 (sync ``E2E-REAL-6-sm120-lowp-matmul-2026-11-04``). Each NVFP4
case builds the retired Graph-authored package
(``tests/_support/nvidia_lowp_matmul_baseline.py``, the Decision #31(a)
oracle) and the production package from the replayed native artifact,
launches both on the RTX 5070, and requires bitwise-equal outputs plus
agreement with a NumPy oracle. INT4 is exact integer arithmetic, so the
production package is held to the exact product. RTX 5070 evidence only; it
transfers to no other part.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tests._support import nvidia_lowp_matmul_baseline as baseline
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_e2e_spine_native import _decode_e2m1, _decode_ue4m3, _pack_int4, _pack_nvfp4
from tests.unit.test_nvidia_lowp_matmul_migration import int4_module, nvfp4_module

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


@pytest.mark.parametrize("shape", [(16, 8, 64), (7, 5, 31), (33, 19, 129)])
@pytest.mark.parametrize("scaled", [False, True])
def test_nvfp4_matches_retired_packager_bitwise(shape, scaled):
    m, n, k = shape
    module = nvfp4_module(m, n, k, scaled=scaled)
    packages = (baseline.baseline_nvfp4_matmul(module, pipeline_name=PIPELINE),
                native.package_native(module, pipeline_name=PIPELINE))
    rng = np.random.default_rng(1_104_2026 + m + n + k)
    a_codes = rng.integers(0, 16, size=(m, k), dtype=np.uint8)
    b_codes = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    sk = (k + 15) // 16
    choices = np.asarray([0x30, 0x31, 0x33, 0x35, 0x38, 0x3A, 0x3D, 0x40, 0x42], np.uint8)
    scale_a = np.ascontiguousarray(choices[(np.arange(m)[:, None] + np.arange(sk)[None, :]) % choices.size])
    scale_b = np.ascontiguousarray(choices[(2 * np.arange(sk)[:, None] + np.arange(n)[None, :]) % choices.size])
    outputs = []
    for package in packages:
        c = np.full((m, n), np.nan, np.float32)
        _launch(package, {"a": _pack_nvfp4(a_codes, 1), "b": _pack_nvfp4(b_codes, 0),
                          "scale_a": scale_a, "scale_b": scale_b, "c": c, "M": m, "N": n, "K": k})
        outputs.append(c)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    a = _decode_e2m1(a_codes) * np.repeat(_decode_ue4m3(scale_a), 16, axis=1)[:, :k]
    b = _decode_e2m1(b_codes) * np.repeat(_decode_ue4m3(scale_b), 16, axis=0)[:k, :]
    np.testing.assert_allclose(outputs[1], a @ b, rtol=0, atol=2e-3)


@pytest.mark.parametrize("shape", [(8, 32, 32), (11, 19, 37), (33, 17, 129)])
def test_int4_projected_package_is_exact(shape):
    m, n, k = shape
    package = native.package_native(int4_module(m, n, k), pipeline_name=PIPELINE)
    rng = np.random.default_rng(1_104_2027 + m + n + k)
    a = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
    b = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
    c = np.full((m, n), -1, np.int32)
    _launch(package, {"a": _pack_int4(a, 1), "b": _pack_int4(b, 0), "c": c, "M": m, "N": n, "K": k})
    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))
