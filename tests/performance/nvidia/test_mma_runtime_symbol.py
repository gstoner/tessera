"""Measured resident NVIDIA MMA runtime-symbol ABI proofs."""
from __future__ import annotations
import ctypes
import numpy as np
import pytest
from tests.device.nvidia import test_mma_runtime_symbol as mma
from tests._support.nvidia_numerics import assert_matches

pytestmark = pytest.mark.hardware_nvidia

@pytest.mark.parametrize("dt", mma._DTYPES)
@pytest.mark.performance
def test_device_resident_gemm_stream_and_event_abi(dt):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    mma._load_lib(); a, b, ref = mma._case(dt, np.random.default_rng(1300 + len(dt)), 19, 13, 29)
    dtype_key = {"bf16": "bfloat16", "f16": "float16", "tf32": "float32", "e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}[dt]
    with NvidiaDeviceSession() as session:
        da, db = session.upload(a), session.upload(b); out = session.empty((19, 13), np.float32)
        def launch(): session.gemm(da, db, out, dtype_key)
        launch(); latency = session.measure(launch, reps=8, warmup=2); got = out.numpy()
    assert latency > 0
    assert_matches(got, ref, dt, reduction_length=29)
