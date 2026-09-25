"""CUDA FFT plans are cached per device (host-only; no GPU needed).

A cuFFT plan and its workspace belong to the CUDA device current when the plan
was created. The cache used to key by (kind, batch, length) alone, so a process
that switched devices got the other device's plan back (Codex review on #839;
sync ROCM-EXEC-PIPELINE-2026-09-24, owner NVIDIA-FFT-WORKSPACE-1).
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera import runtime as rt


class _FakeFFTLib:
    """Hands out one handle per plan and records which device created it."""

    def __init__(self):
        self.device = 0
        self.plan_device: dict[int, int] = {}
        self._next = 0x1000

    def tessera_nvidia_fft_current_device(self, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = self.device
        return 0

    def _create(self, batch, length, plan, workspace):
        self._next += 1
        ctypes.cast(plan, ctypes.POINTER(ctypes.c_void_p))[0] = self._next
        ctypes.cast(workspace, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        self.plan_device[self._next] = self.device
        return 0

    tessera_nvidia_fft_plan_create_c2c_f32 = _create
    tessera_nvidia_fft_plan_create_r2c_f32 = _create
    tessera_nvidia_fft_plan_create_c2r_f32 = _create

    def tessera_nvidia_fft_workspace_alloc(self, size, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 0x9000 + self.device
        return 0

    def tessera_nvidia_fft_workspace_free(self, workspace):
        return 0

    def tessera_nvidia_fft_plan_destroy(self, plan):
        return 0

    def _execute(self, plan, *_args):
        # The real library's v3 refusal: a plan runs only on its own device.
        return 0 if self.plan_device[plan.value] == self.device else 3

    tessera_nvidia_fft_execute_c2c_f32 = _execute
    tessera_nvidia_fft_execute_r2c_f32 = _execute
    tessera_nvidia_fft_execute_c2r_f32 = _execute


@pytest.fixture
def fake(monkeypatch):
    lib = _FakeFFTLib()
    monkeypatch.setattr(rt, "_load_nvidia_fft_runtime", lambda: lib)
    monkeypatch.setattr(rt, "_nvidia_fft_plans", __import__("collections").OrderedDict())
    return lib


def test_switching_devices_creates_a_plan_per_device(fake):
    x = np.ones((2, 16), np.complex64)
    rt._nvidia_fft_c2c_rows(x, False, np)
    rt._nvidia_fft_c2c_rows(x, False, np)  # cache hit on device 0
    fake.device = 1
    rt._nvidia_fft_c2c_rows(x, False, np)  # must not reuse device 0's plan
    assert list(rt._nvidia_fft_plans) == [(0, "c2c", 2, 16), (1, "c2c", 2, 16)]
    assert sorted(fake.plan_device.values()) == [0, 1]
    fake.device = 0
    rt._nvidia_fft_c2c_rows(x, False, np)  # back on 0: its own plan again
    assert len(fake.plan_device) == 2


def test_real_transforms_are_keyed_by_device_too(fake):
    rt._nvidia_fft_real_rows(np.ones((1, 8), np.float32), False, None, np)
    fake.device = 1
    rt._nvidia_fft_real_rows(np.ones((1, 8), np.float32), False, None, np)
    assert {key[0] for key in rt._nvidia_fft_plans} == {0, 1}


def test_a_foreign_device_plan_is_reported_not_run(fake, monkeypatch):
    x = np.ones((1, 8), np.complex64)
    rt._nvidia_fft_c2c_rows(x, False, np)
    # Simulate the old defect: a cache that ignores the device.
    monkeypatch.setattr(rt, "_nvidia_fft_device", lambda lib: 0)
    fake.device = 1
    with pytest.raises(RuntimeError, match="belongs to another CUDA device"):
        rt._nvidia_fft_c2c_rows(x, False, np)
