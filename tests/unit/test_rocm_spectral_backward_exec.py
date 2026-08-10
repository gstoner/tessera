"""Exact-device proof for bounded gfx1151 compound spectral adjoints.

The test compiles the content-addressed Graph -> Schedule -> Tile contract to a
native ROCm kernel, serializes that kernel to hsaco, launches it through HIP,
and compares both gradient outputs with direct numerical oracles.  It is kept
in the hardware lane; host-free CI owns the structural lowering fixtures.
"""

from __future__ import annotations

import ctypes
import os

import pytest

from tests._support.compiler_tool import require_tessera_opt, run_tessera_opt
from tests.unit.test_rocm_mlir_to_hsaco import _extract_hsaco, _hip

np = pytest.importorskip("numpy")

CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")


def _source(kind: str) -> tuple[str, str]:
    if kind == "filter":
        return (
            """
module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @spectral_filter_bwd(
      %dy: tensor<2xcomplex<f32>>, %x: tensor<2xcomplex<f32>>,
      %filter: tensor<2xcomplex<f32>>)
      -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
    %dx, %df = "tessera.spectral_backward"(%dy, %x, %filter) {
      kind = "tessera.spectral_filter", axis = -1 : i64,
      logical_length = 2 : i64, normalization = "backward",
      spectrum_layout = "full_complex"
    } : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>,
         tensor<2xcomplex<f32>>) ->
        (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>)
    return %dx, %df : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
}
""",
            "spectral_filter_bwd",
        )
    return (
        """
module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @spectral_conv_bwd(
      %dy: tensor<4xf32>, %x: tensor<3xf32>, %filter: tensor<2xf32>)
      -> (tensor<3xf32>, tensor<2xf32>) {
    %dx, %df = "tessera.spectral_backward"(%dy, %x, %filter) {
      kind = "tessera.spectral_conv", axis = -1 : i64,
      logical_length = 4 : i64, normalization = "ortho",
      spectrum_layout = "half_spectrum_nyquist_explicit"
    } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>) ->
        (tensor<3xf32>, tensor<2xf32>)
    return %dx, %df : tensor<3xf32>, tensor<2xf32>
  }
}
""",
        "spectral_conv_bwd",
    )


def _compile(kind: str) -> tuple[bytes, str]:
    source, symbol = _source(kind)
    generated = run_tessera_opt(
        source,
        "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile",
        "--lower-tile-to-rocm=arch=gfx1151",
        "--generate-rocm-spectral-backward-kernel",
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
    return hsaco, symbol


def _launch(hip, hsaco: bytes, symbol: str, arrays: list[np.ndarray]):
    if hip.hipInit(0) != 0:
        return None
    module = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), hsaco) != 0:
        return None
    function = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(
        ctypes.byref(function), module, symbol.encode()
    ) == 0
    device = []
    for array in arrays:
        pointer = ctypes.c_void_p()
        assert hip.hipMalloc(ctypes.byref(pointer), array.nbytes) == 0
        device.append(pointer)
        hip.hipMemcpy(
            pointer, array.ctypes.data_as(ctypes.c_void_p), array.nbytes, 1
        )

    args = []
    for pointer, array in zip(device, arrays):
        args.extend(
            (
                ctypes.c_void_p(pointer.value),
                ctypes.c_void_p(pointer.value),
                ctypes.c_int64(0),
                ctypes.c_int64(array.size),
                ctypes.c_int64(1),
            )
        )
    packed = (ctypes.c_void_p * len(args))()
    for index, value in enumerate(args):
        packed[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = (
        [ctypes.c_void_p]
        + [ctypes.c_uint] * 6
        + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    )
    assert launch(function, 1, 1, 1, 256, 1, 1, 0, None, packed, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    outputs = []
    for array, pointer in zip(arrays[-2:], device[-2:]):
        output = np.empty_like(array)
        hip.hipMemcpy(
            output.ctypes.data_as(ctypes.c_void_p), pointer, output.nbytes, 2
        )
        outputs.append(output)
    for pointer in device:
        hip.hipFree(pointer)
    return outputs


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("kind", ("filter", "conv"))
def test_native_compound_spectral_backward_on_gfx1151(kind: str) -> None:
    require_tessera_opt("--generate-rocm-spectral-backward-kernel")
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable")
    if kind == "filter":
        dy = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
        x = np.array([1.0, 4.0, -2.0, 0.25], dtype=np.float32)
        parameter = np.array([3.0, 2.0, -1.0, 5.0], dtype=np.float32)
        dx = np.zeros_like(x)
        dparameter = np.zeros_like(parameter)
        expected = (
            np.array([4.0, -7.0, 14.5, -5.5], dtype=np.float32),
            np.array([-2.0, -9.0, -0.25, -6.125], dtype=np.float32),
        )
    else:
        dy = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = np.array([2.0, -1.0, 3.0], dtype=np.float32)
        parameter = np.array([4.0, 5.0], dtype=np.float32)
        dx = np.zeros_like(x)
        dparameter = np.zeros_like(parameter)
        expected = (
            np.array([7.0, 11.5, 16.0], dtype=np.float32),
            np.array([4.5, 6.5], dtype=np.float32),
        )
    outputs = _launch(
        hip, *_compile(kind), [dy, x, parameter, dx, dparameter]
    )
    if outputs is None:
        pytest.skip("gfx1151 module loading is unavailable")
    np.testing.assert_allclose(outputs[0], expected[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(outputs[1], expected[1], rtol=0.0, atol=1e-6)
