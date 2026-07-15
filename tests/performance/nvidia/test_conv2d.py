"""Serial measured NVIDIA convolution route proofs."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime
from tests.device.nvidia.test_conv2d import _artifact, _reference


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.performance
@pytest.mark.parametrize("route,atol", [
    ("direct", 2e-5), ("shared", 2e-5), ("im2col_tf32", 2e-2)])
def test_live_nvidia_conv2d_performance_routes(route, atol):
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(1400 + len(route))
    x = (rng.standard_normal((1, 12, 11, 8)) * .1).astype(np.float32)
    w = (rng.standard_normal((3, 3, 8, 16)) * .1).astype(np.float32)
    bias = (rng.standard_normal((16,)) * .05).astype(np.float32)
    result = rt.launch(
        _artifact(bias=True, padding=1, route=route), (x, w, bias))
    assert result["ok"] is True, result.get("reason")
    expected = _reference(x, w, bias, 1, 1, 1)
    np.testing.assert_allclose(result["output"], expected, rtol=0, atol=atol)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.performance
def test_live_nvidia_conv2d_size_aware_dispatch_records_measurements():
    rt = require_nvidia_mma_runtime()
    rt._nvidia_conv2d_route_cache.clear()
    rt._nvidia_conv2d_route_evidence.clear()
    rng = np.random.default_rng(1500)
    x = (rng.standard_normal((1, 16, 16, 8)) * .1).astype(np.float32)
    w = (rng.standard_normal((3, 3, 8, 16)) * .1).astype(np.float32)
    result = rt.launch(_artifact(bias=False, padding=1), (x, w))
    assert result["ok"] is True
    assert len(rt._nvidia_conv2d_route_cache) == 1
    evidence = next(iter(rt._nvidia_conv2d_route_evidence.values()))
    assert set(evidence) == {"direct", "shared", "im2col_tf32"}
    winner = next(iter(rt._nvidia_conv2d_route_cache.values()))
    valid_names = {"direct", "shared", "im2col_tf32"}
    assert winner in valid_names and all(v > 0 for v in evidence.values())
