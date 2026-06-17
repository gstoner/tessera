"""G3 drift guard — the C++ Tile→Apple GPU pass's per-op `status` must match the
Python runtime envelope (the single source of truth).

`TileToApple.cpp` tags each lowered `tessera_apple.gpu.metal_kernel` with
`status = "metal_runtime"` (runtime-executable) or `"artifact_only"`. That set is
a hand-maintained mirror of `driver._APPLE_GPU_{MPS,MSL,MPSGRAPH}_OPS`. This test
runs the real `tessera-opt` pass over a `tile.mock` per envelope op + a
non-envelope op and asserts the two agree — so adding a runtime op in `driver.py`
without updating the C++ pass (or vice-versa) fails CI.

Skips when `tessera-opt` isn't built (e.g. Linux runners without the MLIR build).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tessera.compiler import driver

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT = REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _find_opt() -> str | None:
    if _DEFAULT.is_file() and os.access(_DEFAULT, os.X_OK):
        return str(_DEFAULT)
    return shutil.which("tessera-opt")


_OPT = _find_opt()
pytestmark = pytest.mark.skipif(
    _OPT is None, reason="tessera-opt not built; run `cmake --build build --target tessera-opt`")


def _runtime_envelope() -> set[str]:
    # P1 (2026-06-09) — the C++ ``kRuntimeOps`` list is now GENERATED from
    # the envelope registry (`apple_gpu_envelope.py`) via
    # `scripts/generate_apple_runtime_ops_table.py`, so the drift gate covers
    # the FULL runtime envelope (this also closed glass-jaw #10 — projection
    # and reduction ops previously lacked C++ entries).
    return set(driver._APPLE_GPU_RUNTIME_OPS)


def _lower_and_parse(sources: list[str]) -> dict[str, str]:
    """Run tile-to-apple_gpu over a tile.mock per source op; return {source: status}."""
    lines = ["module {"]
    for i, s in enumerate(sources):
        lines.append(f'  "tile.mock"() {{source = "{s}", result = "v{i}", '
                     f"ordinal = {i} : i64}} : () -> ()")
    lines.append("}")
    proc = subprocess.run(
        [_OPT, "-", "-tessera-lower-to-apple_gpu", "--allow-unregistered-dialect"],
        input="\n".join(lines), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    # Each metal_kernel op carries source + status in its (alphabetically sorted)
    # attr dict; source < status, so a per-op non-greedy match pairs them.
    out = {}
    for m in re.finditer(
            r'metal_kernel.*?source = "(tessera\.[^"]+)".*?status = "([a-z_]+)"',
            proc.stdout, re.DOTALL):
        out[m.group(1)] = m.group(2)
    return out


def test_runtime_ops_lower_to_metal_runtime_status():
    envelope = sorted(_runtime_envelope())
    status = _lower_and_parse(envelope)
    missing = [op for op in envelope if op not in status]
    assert not missing, f"pass did not emit a metal_kernel status for: {missing}"
    wrong = {op: status[op] for op in envelope if status[op] != "metal_runtime"}
    assert not wrong, (
        "C++ Tile→Apple pass disagrees with the Python runtime envelope "
        f"(driver._APPLE_GPU_*): these envelope ops were NOT tagged metal_runtime: "
        f"{wrong}. Update isAppleGpuRuntimeOp in TileToApple.cpp to match driver.py.")


def test_non_envelope_op_is_artifact_only():
    # An op outside the envelope must stay artifact_only. (`tessera.add` joined
    # the binary lane; `tessera.transpose`/`tessera.gather` joined the structural
    # data-mover lanes — use `tessera.reshape`, a true-view metadata layout op
    # that is never accelerator-routed.)
    assert "tessera.reshape" not in _runtime_envelope()
    status = _lower_and_parse(["tessera.reshape"])
    assert status.get("tessera.reshape") == "artifact_only", status
