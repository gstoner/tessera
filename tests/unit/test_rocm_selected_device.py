"""Selected-device HIP architecture identity and versioned property ABI."""
import ctypes as ct
from pathlib import Path
import shutil
import subprocess

import pytest

from tessera import runtime as rt


def test_live_arch_tracks_selected_device_and_fails_closed(monkeypatch):
    current = [1]
    failure = [False]
    calls = []
    def device(pointer):
        ct.cast(pointer, ct.POINTER(ct.c_int))[0] = current[0]
        return int(failure[0])
    def properties(pointer, ordinal):
        calls.append(ordinal)
        name = {0: b"gfx1151", 1: b"gfx1201:xnack-", 2: b"gfx90a"}[ordinal]
        ct.memmove(ct.cast(pointer, ct.c_void_p).value + 1160, name + b"\0", len(name) + 1)
        return 0
    class HIP:
        hipGetDevice = staticmethod(device)
        hipGetDevicePropertiesR0600 = staticmethod(properties)
    monkeypatch.setattr(rt, "_load_hip_for_launch", lambda: HIP())
    for ordinal, expected in ((1, "gfx1201"), (0, "gfx1151"), (2, "gfx90a")):
        current[0] = ordinal
        assert rt._rocm_live_arch() == expected
    assert calls == [1, 0, 2]
    failure[0] = True
    assert rt._rocm_live_arch() is None
    assert calls == [1, 0, 2]
    monkeypatch.setattr(rt, "_load_hip_for_launch", lambda: object())
    assert rt._rocm_live_arch() is None



def test_live_arch_memoizes_per_handle_and_ordinal(monkeypatch):
    current = [0]
    calls = []
    def device(pointer):
        ct.cast(pointer, ct.POINTER(ct.c_int))[0] = current[0]
        return 0
    def properties(pointer, ordinal):
        calls.append(ordinal)
        name = {0: b"gfx1201", 1: b"gfx1151"}[ordinal]
        ct.memmove(ct.cast(pointer, ct.c_void_p).value + 1160, name + b"\0", len(name) + 1)
        return 0
    class HIP:
        hipGetDevice = staticmethod(device)
        hipGetDevicePropertiesR0600 = staticmethod(properties)
    handle = HIP()
    monkeypatch.setattr(rt, "_load_hip_for_launch", lambda: handle)
    monkeypatch.setattr(rt, "_rocm_live_arch_cache", (None, {}))
    assert [rt._rocm_live_arch() for _ in range(3)] == ["gfx1201"] * 3
    assert calls == [0]  # hot callers pay the property query once
    current[0] = 1  # the selected ordinal is still re-read every call
    assert rt._rocm_live_arch() == "gfx1151"
    assert calls == [0, 1]
    other = HIP()  # a different HIP handle never inherits the memo
    monkeypatch.setattr(rt, "_load_hip_for_launch", lambda: other)
    assert rt._rocm_live_arch() == "gfx1151"
    assert calls == [0, 1, 1]

def test_physical_attestation_uses_the_selected_rocm_device(monkeypatch):
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setattr(
        rt,
        "_rocm_device_name",
        lambda: pytest.fail("physical attestation must not require the legacy WMMA runtime probe"),
    )
    attestation = rt._physical_execution_attestation(
        target="rocm",
        execution_kind="native_gpu",
        execution_mode="hip_runtime",
        artifact_hash="a" * 64,
        expected_device_arch="gfx1201",
    )
    assert attestation is not None
    assert attestation["device_arch"] == "gfx1201"
    assert len(attestation["digest"]) == 64

    assert rt._physical_execution_attestation(
        target="rocm",
        execution_kind="native_gpu",
        execution_mode="hip_runtime",
        artifact_hash="a" * 64,
        expected_device_arch="gfx1151",
    ) is None


def test_versioned_hip_property_layout_matches_headers(tmp_path):
    compiler = shutil.which("c++")
    headers = Path("/opt/rocm/include")
    if compiler is None or not (headers / "hip/hip_runtime_api.h").exists():
        pytest.skip("requires HIP headers and C++ compiler")
    source = tmp_path / "layout.cpp"
    source.write_text('''
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <cstddef>
static_assert(sizeof(hipDeviceProp_tR0600) == 1472);
static_assert(alignof(hipDeviceProp_tR0600) == 8);
static_assert(offsetof(hipDeviceProp_tR0600, gcnArchName) == 1160);
int main() {}
''')
    subprocess.run([compiler, "-I", str(headers), str(source), "-o", str(tmp_path / "layout")], check=True)
