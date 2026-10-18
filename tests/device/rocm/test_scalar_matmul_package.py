"""Exact-device checked scalar packages, changed roots and compiler-free replay."""
from functools import lru_cache
import os
import subprocess

import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler import rocm_native, scheduled_matmul
from tessera.compiler.rocm_scalar_matmul import prepare
from tests.unit.test_rocm_scalar_package import module

pytestmark = [
    pytest.mark.hardware_rocm,
    pytest.mark.skipif(os.environ.get("TESSERA_ROCM_SCALAR_DEVICE_PROOF") != "1",
                       reason="explicit owning ROCm scalar package proof required"),
]


@lru_cache(maxsize=64)
def packaged(arch, half, side, output, orientation):
    graph = module(arch, half, side, output, orientation)
    scheduled = scheduled_matmul.lower_scheduled_matmul(graph, target="rocm_" + arch)
    package = rocm_native.package_scheduled_matmul(scheduled, pipeline_name="tessera-rocm-executable")
    artifact = rt.RuntimeArtifact(graph_ir=scheduled.graph_ir, schedule_ir=scheduled.schedule_ir,
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor)
    return rt.RuntimeArtifact.from_json(artifact.to_json())


def values(half, side, output, orientation, seed):
    import ml_dtypes
    dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}
    a_type, b_type = ("fp32", half) if side == "a" else (half, "fp32")
    rng = np.random.default_rng(seed)
    a = (rng.integers(-8, 9, (17, 35))/8).astype(dtype[a_type])
    b = (rng.integers(-8, 9, (35, 19))/8).astype(dtype[b_type])
    if side == "a":
        a[0, :] = 0
        a[0, :2] = (1 + 2**-13, -1)
        b[:2, 0] = 1
    else:
        b[:, 0] = 0
        b[:2, 0] = (1 + 2**-13, -1)
        a[0, :2] = 1
    # Dyadic data make this exact double oracle independently representable
    # by every ordered FP32 FMA in the finite test envelope.
    reference = (a.astype(np.float64) @ b.astype(np.float64)).astype(dtype[output])
    ta, tb = orientation
    return (np.ascontiguousarray(a.T if ta else a),
            np.ascontiguousarray(b.T if tb else b)), reference


@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("orientation", [(False, False), (False, True), (True, False), (True, True)])
def test_serialized_scalar_package_replays(half, side, output, orientation, monkeypatch):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    assert rt._rocm_live_arch() == arch
    artifact = packaged(arch, half, side, output, orientation)
    names = [binding.name for binding in artifact.launch_descriptor.buffers]
    previous = []
    for seed in (11, 29):
        roots, reference = values(half, side, output, orientation, seed)
        # The logical output is followed by an independently checked canary.
        storage = np.full(reference.size + 16, -123, dtype=reference.dtype)
        out = storage[:reference.size].reshape(reference.shape)
        args = dict(zip(names, (*roots, out), strict=True), M=17, N=19, K=35)
        with monkeypatch.context() as replay:
            replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("replay compiled"))
            result = rt.launch(artifact, args)
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        assert result["profile"]["kernel_elapsed_ms"] is None
        np.testing.assert_array_equal(out, reference)
        assert float(out[0, 0]) == 2**-13
        assert np.all(storage[reference.size:] == -123)
        previous.append((out, reference.copy()))
    for out, expected in previous:
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
def test_native_owner_updates_and_retains_output_snapshots(half, side, monkeypatch):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    artifact = packaged(arch, half, side, "fp16", (True, True))
    roots, reference = values(half, side, "fp16", (True, True), 41)
    with prepare(artifact.native_image, artifact.launch_descriptor, roots) as owner:
        with monkeypatch.context() as replay:
            replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("owner compiled"))
            generation, _ = owner.invoke()
            first = owner.read(generation)[0]
            np.testing.assert_array_equal(first, reference)
            changed, wanted = values(half, side, "fp16", (True, True), 47)
            for old, new in zip(roots, changed, strict=True):
                old[...] = new
            owner.update(roots)
            generation, _ = owner.invoke()
            second = owner.read(generation)[0]
            np.testing.assert_array_equal(second, wanted)
            np.testing.assert_array_equal(first, reference)


def test_invalid_runtime_dimensions_leave_output_unmodified(monkeypatch):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    artifact = packaged(arch, "fp16", "a", "fp32", (False, True))
    roots, reference = values("fp16", "a", "fp32", (False, True), 53)
    output = np.full(reference.shape, -987, reference.dtype)
    names = [binding.name for binding in artifact.launch_descriptor.buffers]
    args = dict(zip(names, (*roots, output), strict=True), M=17, N=19, K=34)
    with monkeypatch.context() as replay:
        replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("invalid replay compiled"))
        result = rt.launch(artifact, args)
    assert not result["ok"]
    assert np.all(output == -987)


def public_fp32(a, b):
    return ts.ops.matmul(a, b, output_dtype="fp32")


def public_fp16_reordered(b, a):
    return ts.ops.matmul(a, b, output_dtype="fp16")


@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("permuted", [False, True])
def test_public_frontend_reaches_scalar_package(half, side, permuted, monkeypatch):
    import tessera as ts
    arch = os.environ["TESSERA_ROCM_CHIP"]
    output = "fp16" if permuted else "fp32"
    roots, expected = values(half, side, output, (False, False), 71)
    fn = ts.jit(target="rocm_" + arch)(public_fp16_reordered if permuted else public_fp32)
    actual = fn(*(roots[::-1] if permuted else roots))
    assert fn.last_fallback_reason is None
    np.testing.assert_array_equal(actual, expected)
    assert fn._cached_artifact.launch_descriptor.abi_id.endswith("mixed_fp32_fma.v1")
    changed, wanted = values(half, side, output, (False, False), 73)
    with monkeypatch.context() as replay:
        replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("warm public call compiled"))
        second = fn(*(changed[::-1] if permuted else changed))
    np.testing.assert_array_equal(second, wanted)
    np.testing.assert_array_equal(actual, expected)
    assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"


def test_changed_artifact_lineage_is_rejected_before_execution():
    from dataclasses import replace
    arch = os.environ["TESSERA_ROCM_CHIP"]
    artifact = packaged(arch, "fp16", "a", "fp32", (False, True))
    roots, reference = values("fp16", "a", "fp32", (False, True), 61)
    output = np.full(reference.shape, -987, reference.dtype)
    names = [binding.name for binding in artifact.launch_descriptor.buffers]
    args = dict(zip(names, (*roots, output), strict=True), M=17, N=19, K=35)
    changed = replace(artifact, tile_ir=artifact.tile_ir + "// changed lineage\n")
    result = rt.launch(changed, args)
    assert not result["ok"] and result["runtime_status"] == "invalid_artifact"
    assert np.all(output == -987)


@pytest.mark.parametrize("half", ["fp16", "bf16"])
def test_checked_images_reuse_new_shapes(half):
    import ml_dtypes
    arch = os.environ["TESSERA_ROCM_CHIP"]
    images = []
    half_type = np.float16 if half == "fp16" else ml_dtypes.bfloat16
    for m, n, k in ((17, 19, 35), (7, 11, 13)):
        graph = module(arch, half, "a", "fp32", (True, True), (m, n, k))
        scheduled = scheduled_matmul.lower_scheduled_matmul(graph, target="rocm_" + arch)
        package = rocm_native.package_scheduled_matmul(scheduled, pipeline_name="tessera-rocm-executable")
        images.append(package.image)
        roots = (np.ones((k, m), np.float32), np.ones((n, k), half_type))
        with prepare(package.image, package.descriptor, roots) as owner:
            generation, _ = owner.invoke()
            np.testing.assert_array_equal(owner.read(generation)[0], np.full((m, n), k, np.float32))
    assert images[0].payload == images[1].payload
    assert images[0].entry_points == images[1].entry_points
    assert images[1].compile_state == "warm_cache"


@pytest.mark.parametrize("half", ["fp16", "bf16"])
def test_fp32_scalar_classification_and_subnormal_preservation(half):
    import ml_dtypes
    arch = os.environ["TESSERA_ROCM_CHIP"]
    artifact = packaged(arch, half, "a", "fp32", (False, False))
    half_type = np.float16 if half == "fp16" else ml_dtypes.bfloat16
    probes = (np.nextafter(np.float32(0), np.float32(1)),
              np.float32(np.finfo(np.float32).max), np.float32(np.inf),
              np.float32(-np.inf), np.float32(np.nan))
    for probe in probes:
        roots = (np.zeros((17, 35), np.float32), np.zeros((35, 19), half_type))
        roots[0][0, 0] = probe
        roots[1][0, 0] = 1
        with prepare(artifact.native_image, artifact.launch_descriptor, roots) as owner:
            generation, _ = owner.invoke()
            actual = owner.read(generation)[0][0, 0]
        if np.isnan(probe):
            assert np.isnan(actual)
        else:
            assert actual == probe


def public_fused_mixed(a, b):
    return ts.ops.matmul(a, b, output_dtype="fp32", activation="relu")


def test_mixed_request_outside_native_contract_does_not_use_eager_fallback():
    arch = os.environ["TESSERA_ROCM_CHIP"]
    roots, _ = values("fp16", "a", "fp32", (False, False), 83)
    fn = ts.jit(target="rocm_" + arch)(public_fused_mixed)
    with pytest.raises(ValueError, match="native ROCm scalar matmul"):
        fn(*roots)
