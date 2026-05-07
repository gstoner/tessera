#!/usr/bin/env python3
"""Validation environment discovery for Tessera developer workflows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_VENV = Path.home() / "venv" / "bin" / "python"
REPO_VENV = ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class ValidationEnvironment:
    python: str
    python_version: str
    pytest: str
    cmake: str
    cxx: str
    llvm_config: str
    mlir_opt: str
    tmpdir: str
    tmpdir_writable: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def choose_python(
    *,
    env: dict[str, str] | None = None,
    root: Path = ROOT,
    home: Path | None = None,
) -> str:
    """Choose the Python used by validation without running expensive checks."""

    env = env or os.environ
    if env.get("PYTHON"):
        return env["PYTHON"]
    home = home or Path.home()
    user_venv = home / "venv" / "bin" / "python"
    if user_venv.exists() and os.access(user_venv, os.X_OK):
        return str(user_venv)
    repo_venv = root / ".venv" / "bin" / "python"
    if repo_venv.exists() and os.access(repo_venv, os.X_OK):
        return str(repo_venv)
    return env.get("PYTHON_FALLBACK", "python3")


def collect_environment(python: str | None = None) -> ValidationEnvironment:
    python = python or choose_python()
    tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return ValidationEnvironment(
        python=python,
        python_version=_run_text([python, "-V"]),
        pytest=_run_text([python, "-m", "pytest", "--version"]),
        cmake=_tool_version("cmake", "--version"),
        cxx=_tool_version(os.environ.get("CXX", "c++"), "--version"),
        llvm_config=_tool_version("llvm-config", "--version"),
        mlir_opt=shutil.which("mlir-opt") or "missing",
        tmpdir=tmpdir,
        tmpdir_writable=os.access(tmpdir, os.W_OK),
    )


def print_grouped_report(env: ValidationEnvironment) -> None:
    print("==> Validation environment")
    print(f"python: {env.python}")
    print(f"python_version: {env.python_version}")
    print(f"pytest: {env.pytest}")
    print(f"cmake: {env.cmake}")
    print(f"cxx: {env.cxx}")
    print(f"llvm_config: {env.llvm_config}")
    print(f"mlir_opt: {env.mlir_opt}")
    print(f"tmpdir: {env.tmpdir} writable={env.tmpdir_writable}")


def _tool_version(tool: str, *args: str) -> str:
    if shutil.which(tool) is None and "/" not in tool:
        return "missing"
    return _run_text([tool, *args])


def _run_text(cmd: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            list(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return f"error: {exc}"
    text = (proc.stdout or "").strip().splitlines()
    return text[0] if text else f"exit {proc.returncode}"


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    env = collect_environment()
    if "--json" in argv:
        print(json.dumps(env.to_dict(), sort_keys=True))
    else:
        print_grouped_report(env)
    if "No module named pytest" in env.pytest or env.pytest.startswith("error:"):
        return 1
    if not env.tmpdir_writable:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
