"""Exact-device proof of the native sm_120 FP4 weight expansion.

SM120-FP4-EXPAND-NATIVE-20261103 (RTX 5070 on The-Super-Bear). The image is
Graph -> Schedule -> Tile -> LowerTileToNVIDIA -> PTX; both sources are checked
bitwise against ``reference_nvfp4_expand``. gfx1201/gfx1151 results for the
ROCm image are separate proofs and say nothing about this one.
"""
from __future__ import annotations

import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.nvfp4_expansion import (
    mxfp4_expansion_policy, nvfp4_expansion_policy, reference_nvfp4_expand,
)

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]

# (N, K, row_offsets): ragged N against the 256-thread block, unequal
# projections, a one-row projection, and a long-K row.
_NVFP4_SHAPES = [(7, 32, [0, 7]), (300, 256, [0, 1, 100, 300]),
                 (513, 64, [0, 200, 513]), (64, 4096, [0, 32, 64])]
_MXFP4_SHAPES = [(7, 32, [0, 7]), (300, 256, [0, 1, 100, 300]),
                 (513, 96, [0, 200, 513]), (64, 4096, [0, 32, 64])]


def _modules():
    from tessera import runtime
    from tessera.compiler import nvidia_native
    from tessera.compiler import nvidia_nvfp4_expand_native as native

    if not nvidia_native.tools_available():
        pytest.fail("NVIDIA packaging tools unavailable on an NVIDIA device run "
                    "(source scripts/_nvidia_env.sh; build build-nvidia-cuda/)")
    if runtime._load_nvidia_ptx_launch() is None:
        pytest.fail("libtessera_nvidia_ptx_launch.so not loadable on an NVIDIA device run")
    return native


def _codes(rng, n, k):
    codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    # Every E2M1 code in both nibbles of the first row, both signed zeros included.
    first = np.arange(16, dtype=np.uint8)
    codes[0, :8] = first[0::2] | (first[1::2] << 4)
    if k // 2 >= 16:
        codes[0, 8:16] = first[1::2] | (first[0::2] << 4)
    return codes


def _nvfp4_inputs(n, k, offsets, seed):
    rng = np.random.default_rng(seed)
    # Nonnegative finite E4M3 codes (no NaN 0x7F), plus negative zero.
    scales = rng.integers(0, 0x7F, size=(n, k // 16), dtype=np.uint8)
    special = np.array([0x00, 0x80, 0x01, 0x07, 0x08, 0x7E], np.uint8)
    scales.flat[:special.size] = special[:scales.size]
    # 2^-135 drives the smallest E4M3 subnormal to an f32 subnormal (no FTZ).
    globals_ = np.array([1.0, 2.0**-135, 0.37, 3.1e30][:len(offsets) - 1], np.float64)
    return _codes(rng, n, k), scales.view(ml_dtypes.float8_e4m3fn), globals_


def _mxfp4_inputs(n, k, offsets, seed):
    rng = np.random.default_rng(seed)
    exponents = rng.integers(0, 255, size=(n, k // 32), dtype=np.uint8)
    exponents.flat[:3] = np.array([0, 1, 254], np.uint8)[:exponents.size]
    # 2^-127 x 2^-20 stays a nonzero f32 subnormal; 2^127 x 1.0 stays finite.
    globals_ = np.array([1.0, 0.37, 2.0**-20, 0.75][:len(offsets) - 1], np.float64)
    return _codes(rng, n, k), exponents, globals_


def _graph(native, n, k, offsets, policy):
    return native.build_sm120_nvfp4_expansion_graph(n, k, offsets, numeric_policy=policy)


@pytest.mark.parametrize("source", ["nvfp4", "mxfp4"])
def test_expansion_is_bitwise_the_reference(source):
    native = _modules()
    mx = source == "mxfp4"
    policy = mxfp4_expansion_policy() if mx else nvfp4_expansion_policy()
    for n, k, offsets in (_MXFP4_SHAPES if mx else _NVFP4_SHAPES):
        package = native.package_sm120_nvfp4_expansion(_graph(native, n, k, offsets, policy))
        assert package.mxfp4 is mx
        assert package.native.image.architecture == "sm_120a"
        assert package.native.descriptor.abi_id == (
            native.MXFP4_EXPAND_SM120_ABI if mx else native.NVFP4_EXPAND_SM120_ABI)
        metrics = package.native.image.resource_record.metrics
        assert metrics["spill_load_bytes"] == 0 and metrics["spill_store_bytes"] == 0
        codes, scales, globals_ = (_mxfp4_inputs if mx else _nvfp4_inputs)(n, k, offsets, n + k)
        before = [codes.copy(), np.asarray(scales).copy(), globals_.copy()]
        values, plane = native.execute_sm120_nvfp4_expansion(package, codes, scales, globals_)
        expected_values, expected_plane = reference_nvfp4_expand(
            codes, scales, globals_, row_offsets=offsets, numeric_policy=policy)
        assert values.shape == (n, k) and plane.shape == (k // 16, n)
        np.testing.assert_array_equal(values.view(np.uint8), expected_values.view(np.uint8),
                                      err_msg=f"{source} values {n}x{k}")
        np.testing.assert_array_equal(plane.view(np.uint32), expected_plane.view(np.uint32),
                                      err_msg=f"{source} scale plane {n}x{k}")
        if len(offsets) > 2:
            # The equality above covers f32 subnormal scales: no flush to zero.
            assert np.any((plane != 0) & (np.abs(plane) < np.finfo(np.float32).tiny))
        for a, b in zip((codes, np.asarray(scales), globals_), before, strict=True):
            np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


def test_reserved_e8m0_code_is_refused_before_launch(monkeypatch):
    from tessera import runtime

    native = _modules()
    n, k, offsets = 7, 32, [0, 7]
    package = native.package_sm120_nvfp4_expansion(
        _graph(native, n, k, offsets, mxfp4_expansion_policy()))
    codes, exponents, globals_ = _mxfp4_inputs(n, k, offsets, 5)
    exponents[3, 0] = 255

    def no_launch(*_):
        raise AssertionError("a refused expansion reached PTX registration")

    monkeypatch.setattr(runtime, "_register_nvidia_ptx", no_launch)
    with pytest.raises(ValueError, match="E8M0 scale code 255 is reserved"):
        native.execute_sm120_nvfp4_expansion(package, codes, exponents, globals_)


def test_rocm_tile_is_refused_by_the_nvidia_lowering():
    from tessera.compiler.nvidia_native import _tool
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

    _modules()
    graph = build_nvfp4_expansion_graph(16, 32, [0, 16], numeric_policy=nvfp4_expansion_policy(),
                                        target="rocm_gfx1201")
    tool = find_tessera_opt()
    tile = run_tessera_opt(tool, run_tessera_opt(
        tool, graph.to_mlir(target="rocm_gfx1201", canonical=True),
        "--tessera-graph-to-schedule"), "--tessera-schedule-to-tile")
    assert "tile.nvfp4_expand_kernel" in tile and 'arch = "gfx1201"' in tile
    lowered = subprocess.run([str(_tool("tessera-nvidia-opt")), "--tessera-lower-to-nvidia-sm120"],
                             input=tile, capture_output=True, text=True)
    assert lowered.returncode != 0
    assert "NVIDIA_NVFP4_EXPAND_ARCH_MISMATCH" in lowered.stderr


def test_package_refuses_a_non_sm120_graph():
    from tessera.compiler.rocm_nvfp4_expand_native import build_nvfp4_expansion_graph

    native = _modules()
    graph = build_nvfp4_expansion_graph(16, 32, [0, 16], numeric_policy=nvfp4_expansion_policy(),
                                        target="rocm_gfx1201")
    with pytest.raises(ValueError, match="nvidia_sm120 / sm_120"):
        native.package_sm120_nvfp4_expansion(graph)


def test_runtime_launch_resolves_the_matrix_row():
    from tessera.runtime import launch

    native = _modules()
    n, k, offsets = 300, 256, [0, 1, 100, 300]
    package = native.package_sm120_nvfp4_expansion(
        _graph(native, n, k, offsets, mxfp4_expansion_policy()))
    codes, exponents, globals_ = _mxfp4_inputs(n, k, offsets, 77)
    values = np.zeros((n, k), ml_dtypes.float8_e4m3fn)
    plane = np.zeros((k // 16, n), np.float32)
    result = launch(native.expansion_runtime_artifact(package),
                    {"codes": codes, "scales": exponents, "globals": globals_,
                     "values": values, "scale_plane": plane})
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == native.COMPILER_PATH
    assert result["execution_kind"] == "native_gpu"
    output = result["output"]
    assert output["values"] is values and output["scale_plane"] is plane
    expected_values, expected_plane = reference_nvfp4_expand(
        codes, exponents, globals_, row_offsets=offsets, numeric_policy=mxfp4_expansion_policy())
    np.testing.assert_array_equal(output["values"].view(np.uint8), expected_values.view(np.uint8))
    np.testing.assert_array_equal(output["scale_plane"].view(np.uint32),
                                  expected_plane.view(np.uint32))
