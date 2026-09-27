"""Compiler-built device-clock markers (sync WSL-TIMING-ADMISSION-2026-09-26).

The marker is an empty kernel stamped by `--tessera-device-clock-span`; a
calibration window brackets N launches of the unmodified clean image with two
marker launches sharing one span buffer. These tests cover what a host without
the device can check: the target gate and the stamped marker IR. The image and
its ISA check run on the owning device's host (gfx1151/gfx1201 on their
ROCm boxes; sm_120 on The-Super-Bear).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler.native_device_clock import (
    MARKER_ENTRY, VALIDATED_MARKER_TARGETS, _MARKER_MODULE, build_device_clock_marker)
from tessera.compiler.scheduled_matmul import find_tessera_opt


@pytest.mark.parametrize("backend, chip", [
    ("nvidia", "sm_121"), ("nvidia", "sm_90"), ("nvidia", "sm_120a"), ("rocm", "gfx942"),
    ("rocm", ""), ("rocm", "sm_120"), ("nvidia", "gfx1201")])
def test_marker_is_built_only_for_validated_targets(backend, chip) -> None:
    """Each target needs its own exact-device validation: sm_120 was validated
    on The-Super-Bear (NVIDIA-GLOBALTIMER-MARKER-2026-09-26), gfx1151/gfx1201
    on their own boxes. Anything else refuses rather than ship unvalidated."""
    with pytest.raises(ValueError, match="validated for ROCm gfx1151/gfx1201 and NVIDIA sm_120 only"):
        build_device_clock_marker(compiler=Path("unused"), llvm_bin=Path("unused"),
                                  backend=backend, chip=chip)


def test_validated_targets_are_exact() -> None:
    assert set(VALIDATED_MARKER_TARGETS) == {
        ("rocm", "gfx1151"), ("rocm", "gfx1201"), ("nvidia", "sm_120")}


@pytest.mark.skipif(find_tessera_opt() is None, reason="native compiler required")
def test_the_stamped_nvidia_marker_reads_globaltimer_twice_into_one_span() -> None:
    from tessera.compiler.native_gpu_storage import _run
    stamped = _run(Path(find_tessera_opt()), "--tessera-device-clock-span=backend=nvidia",
                   source=_MARKER_MODULE)
    assert len(re.findall(
        r'llvm\.call_intrinsic "llvm\.nvvm\.read\.ptx\.sreg\.globaltimer"', stamped)) == 2
    assert "llvm.atomicrmw umin" in stamped and "llvm.atomicrmw umax" in stamped
    assert stamped.count("gpu.barrier") == 2


def test_the_sass_check_requires_two_globaltimer_reads_and_two_span_atomics(
        monkeypatch, tmp_path) -> None:
    """The NVIDIA image check reads the SASS the driver loads (a marker with
    one read, or a non-atomic span update, would time nothing)."""
    import subprocess
    from tessera.compiler import native_device_clock as ndc
    from tessera.compiler import native_gpu_storage

    good = ("CS2R R4, SR_GLOBALTIMERLO ;\nREDG.E.MIN.64.STRONG.SYS desc[UR4][R2.64], R4 ;\n"
            "CS2R R4, SR_GLOBALTIMERLO ;\nREDG.E.MAX.64.STRONG.SYS desc[UR4][R2.64], R4 ;\n")
    monkeypatch.setattr(native_gpu_storage, "_cuda_disassembler", lambda: tmp_path / "cuobjdump")

    def fake(stdout):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(ndc.subprocess, "run", fake(good))
    ndc._require_globaltimer_and_atomics(b"fatbin")
    for bad in (good.replace("CS2R R4, SR_GLOBALTIMERLO ;\nREDG.E.MAX", "REDG.E.MAX"),
                good.replace("REDG.E.MAX.64", "STG.E.64")):
        monkeypatch.setattr(ndc.subprocess, "run", fake(bad))
        with pytest.raises(ValueError, match="must read %globaltimer twice"):
            ndc._require_globaltimer_and_atomics(b"fatbin")
    monkeypatch.setattr(native_gpu_storage, "_cuda_disassembler", lambda: None)
    with pytest.raises(ValueError, match="needs cuobjdump"):
        ndc._require_globaltimer_and_atomics(b"fatbin")


@pytest.mark.skipif(find_tessera_opt() is None, reason="native compiler required")
def test_the_stamped_marker_reads_the_clock_twice_into_one_span() -> None:
    from tessera.compiler.native_gpu_storage import _run
    stamped = _run(Path(find_tessera_opt()), "--tessera-device-clock-span=backend=rocm",
                   source=_MARKER_MODULE)
    assert f"gpu.func @{MARKER_ENTRY}(%arg0: !llvm.ptr<1>) kernel" in stamped
    assert len(re.findall(r'llvm\.call_intrinsic "llvm\.readsteadycounter"', stamped)) == 2
    assert "llvm.atomicrmw umin" in stamped and "llvm.atomicrmw umax" in stamped
    assert stamped.count("gpu.barrier") == 2
