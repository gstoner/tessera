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
    # Task C (2026-06-01) — added _APPLE_GPU_CONV_OPS so conv2d / conv3d
    # are part of the drift gate. The C++ ``kRuntimeOps`` list in
    # ``TileToApple.cpp`` now includes conv2d + conv3d to match.
    # L-series linalg family (2026-06-02) — cholesky + tri_solve are wired
    # through the C++ Tile→Apple GPU pass (isAppleGpuRuntimeOp) with real MSL
    # kernels, so the full `_APPLE_GPU_LINALG_OPS` set joins the drift gate.
    # The other family members (cholesky_solve/lu/qr/svd) lower GPU-side as
    # artifacts (CPU LAPACK path is real); they join here when their GPU runtime
    # wiring lands.  _APPLE_GPU_PROJECTION_OPS / _APPLE_GPU_REDUCTION_OPS still
    # need their own C++ entries (glass-jaw #10).
    return (set(driver._APPLE_GPU_MPS_OPS) | set(driver._APPLE_GPU_MSL_OPS)
            | set(driver._APPLE_GPU_MPSGRAPH_OPS)
            | set(driver._APPLE_GPU_CONV_OPS)
            | set(driver._APPLE_GPU_LINALG_OPS))


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
    # the MPSGraph binary lane in batch 1 — use `tessera.gather`, a host-class
    # layout op that is never accelerator-routed.)
    assert "tessera.gather" not in _runtime_envelope()
    status = _lower_and_parse(["tessera.gather"])
    assert status.get("tessera.gather") == "artifact_only", status
