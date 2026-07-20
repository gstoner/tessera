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
CUDA_BIN_DIRS = (Path("/usr/local/cuda/bin"), Path("/usr/local/cuda-13.3/bin"))


def ensure_cuda_bin_on_path() -> Path | None:
    """Make the canonical WSL CUDA toolkit visible to this process.

    NVIDIA hosts commonly install CUDA under ``/usr/local/cuda`` without
    adding it to non-interactive WSL shells.  Prefer an existing user PATH
    entry; otherwise prepend the first real toolkit directory so subprocesses
    (pytest fixtures, NVRTC compilation, and benchmark recorders) inherit it.
    """
    entries = os.environ.get("PATH", "").split(os.pathsep)
    for root in CUDA_BIN_DIRS:
        if (root / "nvcc").is_file():
            if str(root) not in entries:
                os.environ["PATH"] = os.pathsep.join([str(root), *filter(None, entries)])
            return root
    return None


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


def nvidia_cuda_tool(name: str) -> Path | None:
    """Find a CUDA executable from PATH or the supported host installations."""

    ensure_cuda_bin_on_path()
    executable = shutil.which(name)
    if executable:
        return Path(executable)
    return next((path for path in (root / name for root in CUDA_BIN_DIRS)
                 if path.is_file()), None)


@dataclass(frozen=True)
class CompilerToolchain:
    """Discovered host compiler tools; requirement checks skip consistently."""

    tessera_opt: Path | None
    mlir_opt: Path | None
    nvidia_opt: Path | None = None

    @classmethod
    def discover(cls) -> "CompilerToolchain":
        return cls(
            tessera_opt=_tool_path(
                "TESSERA_OPT",
                REPO_ROOT / "build/tools/tessera-opt/tessera-opt",
            ),
            mlir_opt=_tool_path(
                "MLIR_OPT",
                "/usr/lib/llvm-23/bin/mlir-opt",
                "/opt/rocm/core/lib/llvm/bin/mlir-opt",
                "/opt/homebrew/opt/llvm@23/bin/mlir-opt",
            ),
            nvidia_opt=_tool_path(
                "TESSERA_NVIDIA_OPT",
                REPO_ROOT / "build-nvidia-cuda/src/compiler/codegen"
                / "tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt",
                REPO_ROOT / "build/src/compiler/codegen"
                / "tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt",
                REPO_ROOT / "build-nvidia/src/compiler/codegen"
                / "tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt",
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
                "compiler-tool test requires MLIR 23 mlir-opt; set MLIR_OPT"
            )
        return self.mlir_opt

    def require_nvidia_opt(self) -> Path:
        if self.nvidia_opt is None:
            pytest.skip(
                "compiler-tool test requires tessera-nvidia-opt; build it or set "
                "TESSERA_NVIDIA_OPT"
            )
        return self.nvidia_opt


def python_subprocess_environment(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an inherited environment where the source package is importable."""

    env = os.environ.copy()
    cuda_bin = ensure_cuda_bin_on_path()
    if cuda_bin is not None:
        env["PATH"] = os.environ["PATH"]
    existing = env.get("PYTHONPATH")
    entries = [str(PYTHON_ROOT), str(REPO_ROOT)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    if overrides:
        env.update(overrides)
    return env
