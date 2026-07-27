"""One source of truth for compiler tools and child-process environments."""

from __future__ import annotations

import os
import platform
import shutil
import re
import subprocess
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
CUDA_BIN_DIRS = (Path("/usr/local/cuda/bin"), Path("/usr/local/cuda-13.3/bin"))


def is_wsl() -> bool:
    """Return whether the current process is running under Windows Subsystem for Linux."""

    release = platform.release().casefold()
    return "microsoft" in release or "wsl" in release or "WSL_INTEROP" in os.environ


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

    def require_tessera_opt(self, *passes: str) -> Path:
        """Return tessera-opt, skipping unless it registers every named pass.

        `tessera-opt` is a *variable* binary: which passes it registers depends
        on how it was configured, and a repo commonly has more than one build
        directory. Without this check a test that needs an Apple pass will pick
        up whichever binary is found first — a lean artifact driver, or simply
        a stale build predating the pass — and fail with
        `Unknown command line argument '--tessera-...'`, which reads like a
        broken test rather than a build-selection problem.

        Pass the pass names (without the leading `--`) that the test drives.
        """
        if self.tessera_opt is None:
            pytest.skip(
                "compiler-tool test requires tessera-opt; build it or set TESSERA_OPT"
            )
        missing = [name for name in passes
                   if name not in self._registered_passes(self.tessera_opt)]
        if missing:
            pytest.skip(
                f"{self.tessera_opt} does not register {', '.join(missing)} "
                f"(build profile: {self._build_profile(self.tessera_opt)}). "
                "Point TESSERA_OPT at a build configured with the owning "
                "backend, or rebuild this one."
            )
        return self.tessera_opt

    @staticmethod
    @lru_cache(maxsize=8)
    def _registered_passes(tool: Path) -> frozenset[str]:
        """Pass names this binary registers, read once per tool path."""
        try:
            help_text = subprocess.run(
                [str(tool), "--help"], capture_output=True, text=True,
                check=False, timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return frozenset()
        return frozenset(re.findall(r"--([A-Za-z0-9][\w-]*)", help_text))

    @staticmethod
    @lru_cache(maxsize=8)
    def _build_profile(tool: Path) -> str:
        """`--tessera-build-info` summary, or a note that the build predates it."""
        try:
            result = subprocess.run(
                [str(tool), "--tessera-build-info"], capture_output=True,
                text=True, check=False, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        if result.returncode != 0 or not result.stdout.strip():
            return "unknown (binary predates --tessera-build-info)"
        return " ".join(result.stdout.split())

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
