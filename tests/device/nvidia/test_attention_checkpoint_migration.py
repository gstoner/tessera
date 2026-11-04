"""Exact-SM120 differential: retired Graph-authored attention packages vs Tile projections.

E2E-REAL-6 (sync ``E2E-REAL-6-sm120-attention-2026-11-03``). Each case builds
the retired Graph-authored package (``tests/_support/
nvidia_attention_checkpoint_baseline.py``, the Decision #31(a) oracle) and the
production package from the replayed native artifact, launches both on the
RTX 5070, and requires bitwise-equal outputs plus agreement with a NumPy
oracle. RTX 5070 evidence only; it transfers to no other part.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tests._support import nvidia_attention_checkpoint_baseline as baseline
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_checkpoint_native import _reference
from tests.device.nvidia.test_native_recompute_schedule import reference as recompute_reference
from tests.unit.test_nvidia_attention_checkpoint_migration import _saved
from tests.unit.test_nvidia_native_recompute_schedule import recompute_module

pytestmark = pytest.mark.hardware_nvidia
PIPELINE = "tessera-nvidia-pipeline-sm120"
DIMS = ("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv")


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


def _bias_scalars(package, bias):
    names = [s.name for s in package.descriptor.scalars]
    return dict(zip(("BiasB", "BiasH", "BiasQ", "BiasK"), bias.shape)) if "BiasB" in names else {}


@pytest.mark.parametrize("shape", [(1, 2, 1, 3, 4, 4, 3), (2, 4, 2, 5, 7, 8, 6), (1, 4, 2, 16, 16, 32, 32)])
@pytest.mark.parametrize("bias", [None, "full", "broadcast"])
def test_saved_lse_pair_matches_retired_packager_bitwise(shape, bias):
    b, hq, hkv, sq, sk, d, dv = shape
    forward_module, backward_module = _saved(False, shape, bias, False), _saved(True, shape, bias, False)
    forwards = (baseline.baseline_attention_lse(forward_module, pipeline_name=PIPELINE),
                native.package_native(forward_module, pipeline_name=PIPELINE))
    backwards = (baseline.baseline_attention_backward_lse(backward_module, pipeline_name=PIPELINE),
                 native.package_native(backward_module, pipeline_name=PIPELINE))
    rng = np.random.default_rng(1_103_2026)
    q = (rng.normal(size=(b, hq, sq, d)) * .2).astype(np.float32)
    k = (rng.normal(size=(b, hkv, sk, d)) * .2).astype(np.float32)
    v = (rng.normal(size=(b, hkv, sk, dv)) * .2).astype(np.float32)
    do = (rng.normal(size=(b, hq, sq, dv)) * .2).astype(np.float32)
    bias_shape = {"full": (b, hq, sq, sk), "broadcast": (b, hq, 1, 1)}.get(bias)
    extra = {"bias": (rng.normal(size=bias_shape) * .3).astype(np.float32)} if bias_shape else {}
    scalars = dict(zip(DIMS, shape, strict=True))
    outputs = []
    for package in forwards:
        o, lse = np.empty((b, hq, sq, dv), np.float32), np.empty((b, hq, sq), np.float32)
        _launch(package, {"q": q, "k": k, "v": v, "o": o, "row_lse": lse, **scalars, **extra,
                          **(_bias_scalars(package, extra["bias"]) if extra else {})})
        outputs.append((o, lse))
    (old_o, old_lse), (new_o, new_lse) = outputs
    np.testing.assert_array_equal(old_o, new_o)
    np.testing.assert_array_equal(old_lse, new_lse)
    full_bias = np.broadcast_to(extra["bias"], (b, hq, sq, sk)) if extra else None
    ref_o, ref_lse, ref_grads = _reference(q, k, v, do, full_bias)
    np.testing.assert_allclose(new_o, ref_o, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(new_lse, ref_lse, atol=2e-6, rtol=2e-5)
    gradients = []
    for package in backwards:
        dq, dk, dv_out = (np.empty_like(x) for x in (q, k, v))
        _launch(package, {"do": do, "q": q, "k": k, "v": v, "output": new_o, "row_lse": new_lse,
                          "dq": dq, "dk": dk, "dv": dv_out, **scalars, **extra,
                          **(_bias_scalars(package, extra["bias"]) if extra else {})})
        gradients.append((dq, dk, dv_out))
    for old, new, ref in zip(*gradients, ref_grads, strict=True):
        np.testing.assert_array_equal(old, new)
        np.testing.assert_allclose(new, ref, atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("advanced", [False, True])
def test_recompute_matches_retired_python_tile_packager_bitwise(dtype, bias, advanced):
    module = recompute_module(dtype, bias, False)
    policy = dict(scale=.5, causal=True, window=(-1, -1), softcap=0., dropout=0., seed=0)
    if advanced:
        policy.update(scale=.3, window=(2, 0), softcap=.7, dropout=.2, seed=-182400000)
    module.functions[0].body[0].kwargs.update(policy)
    old = baseline.baseline_attention_backward(module, pipeline_name=PIPELINE)
    new = native.package_native(module, pipeline_name=PIPELINE)
    storage = {"fp16": np.float16, "fp32": np.float32}.get(dtype)
    if dtype == "bf16":
        from ml_dtypes import bfloat16
        storage = bfloat16
    rng = np.random.default_rng(1_103_2027)
    q, k, v, do = ((rng.normal(size=shape) * .2).astype(storage)
                   for shape in ((1, 2, 3, 4), (1, 1, 4, 4), (1, 1, 4, 3), (1, 2, 3, 3)))
    extra = {"bias": (rng.normal(size=(1, 2, 3, 4)) * .3).astype(np.float32)} if bias else {}
    scalars = dict(zip(DIMS, (1, 2, 1, 3, 4, 4, 3), strict=True))
    results = []
    for package in (old, new):
        dq, dk, dv = (np.empty_like(x) for x in (q, k, v))
        _launch(package, {"do": do, "q": q, "k": k, "v": v, "dq": dq, "dk": dk, "dv": dv, **scalars, **extra})
        results.append((dq, dk, dv))
    expected = recompute_reference(q, k, v, do, extra.get("bias"), **policy)
    atol, rtol = (2e-6, 2e-4) if dtype == "fp32" else (3e-5, 4e-3) if dtype == "fp16" else (2e-4, 2e-2)
    for old_out, new_out, ref in zip(*results, expected, strict=True):
        np.testing.assert_array_equal(old_out.view(np.uint8), new_out.view(np.uint8))
        np.testing.assert_allclose(new_out.astype(np.float64), ref.astype(storage).astype(np.float64),
                                   atol=atol, rtol=rtol)
