"""Exact-device proof for the gfx1151 EGGROLL rank-1 correction.

The host-free fixtures prove the content-addressed Graph -> Schedule -> Tile
contract.  This hardware test goes through ROCDL and a real hsaco launch, then
compares the physical member-keyed RNG and correction arithmetic with the
portable ES reference.
"""

from __future__ import annotations

import ctypes
import os

import pytest

from tessera import rng
from tessera.stdlib.es import low_rank_correction
from tests._support.compiler_tool import require_tessera_opt, run_tessera_opt
from tests.unit.test_rocm_mlir_to_hsaco import _extract_hsaco, _hip

np = pytest.importorskip("numpy")

CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
def _source(population: int, rows: int, in_dim: int, out_dim: int) -> str:
    return f"""
module attributes {{tessera.target = "rocm", tessera.arch = "gfx1151"}} {{
  func.func @es_rank1(
      %x: tensor<{population}x{rows}x{in_dim}xf32>,
      %members: tensor<{population}xi64>, %key: tensor<2xi64>)
      -> tensor<{population}x{rows}x{out_dim}xf32> {{
    %correction = "tessera.es_low_rank_correction"(%x, %members, %key) {{
      out_dim = {out_dim} : i64, rank = 1 : i64, epoch = 3 : i64,
      sigma = 2.000000e-02 : f64, antithetic = true,
      numeric_policy = {{storage = "fp32", accum = "fp32"}}
    }} : (tensor<{population}x{rows}x{in_dim}xf32>, tensor<{population}xi64>,
          tensor<2xi64>) -> tensor<{population}x{rows}x{out_dim}xf32>
    return %correction : tensor<{population}x{rows}x{out_dim}xf32>
  }}
}}
"""


def _compile(source: str) -> bytes:
    generated = run_tessera_opt(
        source,
        "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile",
        "--lower-tile-to-rocm=arch=gfx1151",
        "--generate-rocm-es-low-rank-kernel",
    )
    assert generated.returncode == 0, generated.stderr
    pipeline = (
        "builtin.module(convert-scf-to-cf,"
        "gpu.module(convert-gpu-to-rocdl,reconcile-unrealized-casts),"
        f"rocdl-attach-target{{chip={CHIP}}},gpu-module-to-binary)"
    )
    serialized = run_tessera_opt(generated.stdout, f"--pass-pipeline={pipeline}")
    assert serialized.returncode == 0, serialized.stderr
    hsaco = _extract_hsaco(serialized.stdout)
    assert hsaco[:4] == b"\x7fELF"
    return hsaco


def _launch(hip, hsaco: bytes, arrays: list[np.ndarray]) -> np.ndarray | None:
    if hip.hipInit(0) != 0:
        return None
    module = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), hsaco) != 0:
        return None
    function = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(
        ctypes.byref(function), module, b"es_rank1"
    ) == 0

    device: list[ctypes.c_void_p] = []
    for array in arrays:
        pointer = ctypes.c_void_p()
        assert hip.hipMalloc(ctypes.byref(pointer), array.nbytes) == 0
        device.append(pointer)
        assert hip.hipMemcpy(
            pointer, array.ctypes.data_as(ctypes.c_void_p), array.nbytes, 1
        ) == 0

    arguments: list[ctypes._SimpleCData] = []
    for pointer, array in zip(device, arrays):
        arguments.extend(
            (
                ctypes.c_void_p(pointer.value),
                ctypes.c_void_p(pointer.value),
                ctypes.c_int64(0),
                ctypes.c_int64(array.size),
                ctypes.c_int64(1),
            )
        )
    packed = (ctypes.c_void_p * len(arguments))()
    for index, argument in enumerate(arguments):
        packed[index] = ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)

    launch = hip.hipModuleLaunchKernel
    launch.argtypes = (
        [ctypes.c_void_p]
        + [ctypes.c_uint] * 6
        + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    )
    blocks = arrays[0].shape[0] * arrays[0].shape[1]
    assert launch(
        function, blocks, 1, 1, 256, 1, 1, 0, None, packed, None
    ) == 0
    assert hip.hipDeviceSynchronize() == 0
    result = np.empty_like(arrays[-1])
    assert hip.hipMemcpy(
        result.ctypes.data_as(ctypes.c_void_p), device[-1], result.nbytes, 2
    ) == 0
    for pointer in device:
        hip.hipFree(pointer)
    return result


@pytest.mark.hardware_rocm
@pytest.mark.parametrize(
    "population,rows,in_dim,out_dim",
    ((4, 3, 8, 6), (2, 2, 513, 277)),
)
def test_native_es_low_rank_correction_on_gfx1151(
    population: int, rows: int, in_dim: int, out_dim: int
) -> None:
    require_tessera_opt("--generate-rocm-es-low-rank-kernel")
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable")

    generator = np.random.default_rng(7)
    values = generator.standard_normal(
        (population, rows, in_dim), dtype=np.float32
    )
    members = np.arange(8, 8 + population, dtype=np.int64)
    key = rng.RNGKey(seed_high=0x0123456789ABCDEF, seed_low=0x1234567890ABCDEF)
    key_storage = np.array([key.seed_high, key.seed_low], dtype=np.int64)
    output = np.zeros((population, rows, out_dim), dtype=np.float32)

    result = _launch(
        hip,
        _compile(_source(population, rows, in_dim, out_dim)),
        [values, members, key_storage, output],
    )
    if result is None:
        pytest.skip("gfx1151 module loading is unavailable")
    expected = low_rank_correction(
        values,
        members,
        key,
        out_dim=out_dim,
        rank=1,
        sigma=0.02,
        epoch=3,
        antithetic=True,
    )
    np.testing.assert_allclose(result, expected, rtol=2.0e-5, atol=2.0e-6)
