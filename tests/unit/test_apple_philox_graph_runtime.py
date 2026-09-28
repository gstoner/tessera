"""Exact Metal proof for the checked Graph Philox Langevin ABI."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import philox


@pytest.mark.hardware_apple_gpu
def test_graph_philox_status_runs_metal_and_matches_counter_oracle():
    library = os.environ.get("TESSERA_APPLE_GPU_RUNTIME_LIB")
    assert library and Path(library).is_file(), "set TESSERA_APPLE_GPU_RUNTIME_LIB to a fresh Mac runtime"
    lib = ctypes.CDLL(library)
    call = lib.tessera_apple_gpu_ebm_langevin_step_philox_graph_f32_status
    pf = ctypes.POINTER(ctypes.c_float)
    pi = ctypes.POINTER(ctypes.c_int64)
    call.argtypes = [pf, pf, pi, pi, ctypes.c_float, ctypes.c_float,
                     pf, ctypes.c_int32]
    call.restype = ctypes.c_int32

    y = np.linspace(-0.75, 0.75, 16, dtype=np.float32).reshape(2, 8)
    grad = np.linspace(0.2, -0.2, 16, dtype=np.float32).reshape(2, 8)
    seed = np.array([0xCAFE0042DEADBEEF], dtype=np.uint64).view(np.int64)
    counter = np.array([7, 0, 1, 2], dtype=np.int64)
    output = np.full_like(y, np.nan)
    eta, noise = 0.05, 0.1
    status = call(y.ctypes.data_as(pf), grad.ctypes.data_as(pf),
                  seed.ctypes.data_as(pi), counter.ctypes.data_as(pi),
                  eta, noise, output.ctypes.data_as(pf), y.size)
    assert status == 1, "Graph ABI did not execute on Metal"

    key = np.array([0xDEADBEEF, 0xCAFE0042], np.uint32)
    expected = np.empty_like(y).reshape(-1)
    for index, (value, slope) in enumerate(zip(y.flat, grad.flat)):
        ctr = np.array([7 + index, 0, 1, 2], np.uint32)
        words = philox.philox_4x32_10(ctr, key)
        u0 = (float(words[0]) + 0.5) * 2.0 ** -32
        u1 = (float(words[1]) + 0.5) * 2.0 ** -32
        z = np.sqrt(-2.0 * np.log(u0)) * np.cos(2.0 * np.pi * u1)
        expected[index] = float(value) - eta * float(slope) + noise * z
    np.testing.assert_allclose(output.reshape(-1), expected, rtol=1e-4, atol=1e-4)

    invalid = counter.copy()
    invalid[0] = 2 ** 32
    output.fill(np.nan)
    assert call(y.ctypes.data_as(pf), grad.ctypes.data_as(pf),
                seed.ctypes.data_as(pi), invalid.ctypes.data_as(pi),
                eta, noise, output.ctypes.data_as(pf), y.size) == 0
    assert np.isnan(output).all()


@pytest.mark.hardware_apple_gpu
def test_scaled_rope_graph_calls_execute_on_metal():
    library = os.environ.get("TESSERA_APPLE_GPU_RUNTIME_LIB")
    assert library and Path(library).is_file(), "set TESSERA_APPLE_GPU_RUNTIME_LIB to a fresh Mac runtime"
    lib = ctypes.CDLL(library)
    pf = ctypes.POINTER(ctypes.c_float)
    divide = lib.tessera_apple_gpu_mpsgraph_binary_f32_status
    divide.argtypes = [ctypes.c_int32, pf, pf, pf, ctypes.c_int64]
    divide.restype = ctypes.c_int32
    rope = lib.tessera_apple_gpu_rope_f32_status
    rope.argtypes = [pf, pf, pf, ctypes.c_int32, ctypes.c_int32]
    rope.restype = ctypes.c_int32

    x = np.linspace(-0.75, 0.75, 32, dtype=np.float32).reshape(4, 8)
    theta = np.linspace(-0.3, 0.6, 32, dtype=np.float32).reshape(4, 8)
    scale = np.full_like(theta, 2)
    scaled = np.full_like(theta, np.nan)
    output = np.full_like(x, np.nan)
    assert divide(3, theta.ctypes.data_as(pf), scale.ctypes.data_as(pf),
                  scaled.ctypes.data_as(pf), theta.size) == 1
    assert rope(x.ctypes.data_as(pf), scaled.ctypes.data_as(pf),
                output.ctypes.data_as(pf), 4, 8) == 1
    angle = theta[:, 0::2].astype(np.float64) / 2
    even = x[:, 0::2].astype(np.float64)
    odd = x[:, 1::2].astype(np.float64)
    expected = np.empty_like(x)
    expected[:, 0::2] = even * np.cos(angle) - odd * np.sin(angle)
    expected[:, 1::2] = even * np.sin(angle) + odd * np.cos(angle)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)
