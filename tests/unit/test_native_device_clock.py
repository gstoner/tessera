"""Compiler-built device-clock markers (sync WSL-TIMING-ADMISSION-2026-09-26).

The marker is an empty kernel stamped by `--tessera-device-clock-span`; a
calibration window brackets N launches of the unmodified clean image with two
marker launches sharing one span buffer. These tests cover what a host without
the device can check: the target gate and the stamped marker IR. The image and
its ISA check run on the owning ROCm host.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler.native_device_clock import (
    MARKER_ENTRY, _MARKER_MODULE, build_device_clock_marker)
from tessera.compiler.scheduled_matmul import find_tessera_opt


@pytest.mark.parametrize("backend, chip", [("nvidia", "sm_120"), ("rocm", "gfx942"), ("rocm", "")])
def test_marker_is_built_only_for_validated_targets(backend, chip) -> None:
    """NVIDIA's %globaltimer marker is owed on Super-Bear; it must refuse,
    not ship unvalidated."""
    with pytest.raises(ValueError, match="validated for ROCm gfx1151/gfx1201 only"):
        build_device_clock_marker(compiler=Path("unused"), llvm_bin=Path("unused"),
                                  backend=backend, chip=chip)


@pytest.mark.skipif(find_tessera_opt() is None, reason="native compiler required")
def test_the_stamped_marker_reads_the_clock_twice_into_one_span() -> None:
    from tessera.compiler.native_gpu_storage import _run
    stamped = _run(Path(find_tessera_opt()), "--tessera-device-clock-span=backend=rocm",
                   source=_MARKER_MODULE)
    assert f"gpu.func @{MARKER_ENTRY}(%arg0: !llvm.ptr<1>) kernel" in stamped
    assert len(re.findall(r'llvm\.call_intrinsic "llvm\.readsteadycounter"', stamped)) == 2
    assert "llvm.atomicrmw umin" in stamped and "llvm.atomicrmw umax" in stamped
    assert stamped.count("gpu.barrier") == 2
