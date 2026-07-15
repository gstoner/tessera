"""One source of truth for compiler tools and child-process environments."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"


def _tool_path(env_name: str, *candidates: Path | str) -> Path | None:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    executable = shutil.which(env_name.lower().replace("_", "-"))
    return Path(executable) if executable else None


@dataclass(frozen=True)
class CompilerToolchain:
    """Discovered host compiler tools; requirement checks skip consistently."""

    tessera_opt: Path | None
    mlir_opt: Path | None

    @classmethod
    def discover(cls) -> "CompilerToolchain":
        return cls(
            tessera_opt=_tool_path(
                "TESSERA_OPT",
                REPO_ROOT / "build/tools/tessera-opt/tessera-opt",
            ),
            mlir_opt=_tool_path(
                "MLIR_OPT",
                "/usr/lib/llvm-22/bin/mlir-opt",
                "/opt/homebrew/opt/llvm/bin/mlir-opt",
            ),
        )

    def require_tessera_opt(self) -> Path:
        if self.tessera_opt is None:
            pytest.skip(
                "compiler-tool test requires tessera-opt; build it or set TESSERA_OPT"
            )
        return self.tessera_opt

    def require_mlir_opt(self) -> Path:
        if self.mlir_opt is None:
            pytest.skip(
                "compiler-tool test requires MLIR 22 mlir-opt; set MLIR_OPT"
            )
        return self.mlir_opt


def python_subprocess_environment(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an inherited environment where the source package is importable."""

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    entries = [str(PYTHON_ROOT), str(REPO_ROOT)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    if overrides:
        env.update(overrides)
    return env
