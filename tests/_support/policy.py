"""Canonical test-layer and environment policy.

Test *layer* describes what a test proves.  Environment markers describe what
it needs to produce that proof.  Keeping those axes independent avoids the old
failure mode where a file under ``tests/unit`` could silently become a compiler,
device, or benchmark test depending on what happened to be installed locally.
"""

from __future__ import annotations

from enum import Enum


class TestLayer(str, Enum):
    UNIT = "unit"
    COMPILER = "compiler"
    INTEGRATION = "integration"
    DEVICE = "device"
    PERFORMANCE = "performance"
    AUDIT = "audit"


MARKERS: dict[str, str] = {
    "compiler_tool": (
        "requires a built external compiler tool such as tessera-opt or mlir-opt"
    ),
    "integration": "crosses a process, package, runtime, or component boundary",
    "performance": (
        "measures wall-clock/device performance; excluded from the CPU PR lane"
    ),
    "hardware_apple_gpu": "requires a Darwin host with Metal hardware",
    "hardware_nvidia": "requires an NVIDIA GPU with the CUDA toolkit",
    "hardware_rocm": "requires an AMD GPU with the ROCm toolkit",
}


PR_MARKER_EXPRESSION = (
    "not slow and not performance and not hardware_apple_gpu "
    "and not hardware_nvidia and not hardware_rocm"
)
