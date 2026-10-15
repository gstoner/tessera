"""Owning gfx1201 full native expansion route, with bitwise output checks."""
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy, reference_nvfp4_expand
from tessera.compiler.rocm_nvfp4_expand_native import (
    execute_nvfp4_expansion, package_nvfp4_expansion,
)
from tests.unit.test_rocm_graph_nvfp4_expand import graph

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires the gfx1201 owning device")


@pytest.mark.parametrize("k", [16, 48, 80])
def test_native_codes_and_projection_scales(k):
    assert rt._rocm_live_arch() == "gfx1201"
    codes = np.resize(np.arange(256, dtype=np.uint8), (7, k // 2))
    scale_bits = np.resize(np.arange(127, dtype=np.uint8), (7, k // 16))
    scale_bits[0, 0] = 128  # Retain a negative-zero scale.
    scales = scale_bits.view(ml_dtypes.float8_e4m3fn)
    globals_ = np.array([0.000123456789, 0.007654321], np.float64)
    expected = reference_nvfp4_expand(codes, scales, globals_, row_offsets=[0, 3, 7],
                                      numeric_policy=nvfp4_expansion_policy())
    package = package_nvfp4_expansion(graph(k))
    actual = execute_nvfp4_expansion(package, codes, scales, globals_)
    np.testing.assert_array_equal(actual[0].view(np.uint8), expected[0].view(np.uint8))
    np.testing.assert_array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))


@pytest.mark.parametrize("global_", [
    np.float64(2.**-149), np.float64(2.**-126),
    np.float64(1. + 2.**-24), np.nextafter(np.float64(1. + 2.**-24), np.float64(2.)),
    np.float64(np.finfo(np.float32).max),
])
def test_native_scale_rounding_boundaries(global_):
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_nvfp4_expansion(graph(16))
    codes = np.zeros((7, 8), np.uint8)
    scales = np.ones((7, 1), dtype=ml_dtypes.float8_e4m3fn)
    scales[0, 0] = -0.
    globals_ = np.array([global_, global_], np.float64)
    expected = reference_nvfp4_expand(codes, scales, globals_, row_offsets=[0, 3, 7],
                                      numeric_policy=nvfp4_expansion_policy())
    actual = execute_nvfp4_expansion(package, codes, scales, globals_)
    np.testing.assert_array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))


def test_invalid_scale_rejected_before_gpu_submit(monkeypatch):
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_nvfp4_expansion(graph(16))
    import tessera.compiler.rocm_nvfp4_expand_native as native
    monkeypatch.setattr(native, "_execute_arrays",
                        lambda *a, **kw: pytest.fail("invalid input reached GPU submit"))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        execute_nvfp4_expansion(package, np.zeros((7, 8), np.uint8),
            np.full((7, 1), np.nan, dtype=ml_dtypes.float8_e4m3fn), np.ones(2, np.float64))


def test_catalog_graph_and_package_roundtrip():
    from dataclasses import replace
    from tessera.compiler.native_artifact import NativeImageArtifact, LaunchDescriptor
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph
    module = build_nvfp4_expansion_graph(7, 48, [0, 3, 7],
                                        numeric_policy=nvfp4_expansion_policy())
    assert [str(t) for t in module.functions[0].result_types] == [
        "tensor<7x48xf8E4M3FN>", "tensor<3x7xf32>"]
    before = module.to_mlir(target="rocm_gfx1201", canonical=True)
    package = package_nvfp4_expansion(module)
    assert module.to_mlir(target="rocm_gfx1201", canonical=True) == before
    restored = replace(package, native=replace(package.native,
        image=NativeImageArtifact.from_dict(package.native.image.to_dict()),
        descriptor=LaunchDescriptor.from_dict(package.native.descriptor.to_dict())))
    restored.validate()
    for damaged in [
        replace(package, graph_ir=package.graph_ir + " "),
        replace(package, native=replace(package.native, descriptor=replace(
            package.native.descriptor, abi_id="other"))),
        replace(package, native=replace(package.native, descriptor=replace(
            package.native.descriptor, provenance={}))),
    ]:
        with pytest.raises(ValueError):
            damaged.validate()
