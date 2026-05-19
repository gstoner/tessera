"""Regression: the IR-pipeline tutorial example must run end-to-end.

Findings audit (2026-05-19) caught this demo crashing on a stale
``JitFn.uses_compiled_path`` attribute that was renamed to the
``is_executable`` / ``is_reference_execution`` / ``is_native_execution``
trio with a ``lowering_artifacts()`` accessor.  The example is the
canonical "look at all four IR layers" walkthrough; if it breaks,
documentation drifts away from the actual API.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "examples" / "compiler" / "ir_pipeline_tutorial" / "tessera_ir_pipeline_demo.py"


def test_ir_pipeline_tutorial_runs() -> None:
    """``python examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py``
    must complete with rc=0 and print the headline labels the README
    talks about — that's the contract for the public tutorial."""
    assert DEMO.is_file(), f"missing demo: {DEMO}"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}/python{os.pathsep}{REPO_ROOT}{os.pathsep}"
        f"{env.get('PYTHONPATH', '')}"
    )
    proc = subprocess.run(
        [sys.executable, str(DEMO)],
        capture_output=True, text=True, timeout=60,
        env=env, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"demo exited rc={proc.returncode}:\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )
    # Spot-check the headline labels the demo prints.  Drift in any
    # of these means the demo is talking about an attribute the
    # JitFn class no longer exposes.
    for marker in (
        "execution_kind:",
        "is_executable:",
        "is_reference_execution:",
        "is_native_execution:",
        "lowering_artifacts()",
        "==== GRAPH IR ====",
    ):
        assert marker in proc.stdout, (
            f"missing demo marker {marker!r} in stdout:\n{proc.stdout}"
        )
