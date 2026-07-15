"""Release-gate target-selection contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "release_gate_under_test", ROOT / "scripts" / "release_gate.py")
assert _SPEC is not None and _SPEC.loader is not None
release_gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = release_gate
_SPEC.loader.exec_module(release_gate)


def test_apple_release_gate_selects_the_registered_hardware_marker():
    gate = next(
        gate for gate in release_gate._APPLE_GPU_GATES
        if gate.name == "apple_gpu_hardware_marked_tests"
    )
    pytest_index = gate.cmd.index("pytest")
    marker_index = gate.cmd.index("-m", pytest_index + 1) + 1
    assert gate.cmd[marker_index] == "hardware_apple_gpu"
