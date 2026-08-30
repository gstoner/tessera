"""One source of truth for compiler tools and child-process environments."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
CUDA_BIN_DIRS = (Path("/usr/local/cuda/bin"), Path("/usr/local/cuda-13.3/bin"))
# WSL2 ships the NVIDIA driver shim (nvidia-smi, libcuda.so) here rather than in
# a system bin directory, and only an interactive shell puts it on PATH.
NVIDIA_DRIVER_DIRS = (Path("/usr/lib/wsl/lib"),)


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


def ensure_nvidia_driver_on_path() -> Path | None:
    """Make the NVIDIA driver shim visible to this process.

    Under WSL2 ``nvidia-smi`` lives in ``/usr/lib/wsl/lib``, which the
    interactive ``.bashrc`` adds and a non-interactive shell -- an
    ``ssh <host> <cmd>``, a CI step, a bare ``pytest`` -- never sees. Because
    the device gate probes for ``nvidia-smi`` by name, its absence from PATH
    made every NVIDIA device test SKIP while the run still exited 0: measured
    2026-08-30 on The-Super-Bear as *454 passed, 395 skipped*, with a healthy
    RTX 5070, /dev/dxg and CUDA 13.3 present the whole time. Reporting that as
    sm_120 evidence would assert a hardware result that never ran, so the gate
    repairs its own PATH rather than trusting the caller to remember.
    """
    entries = os.environ.get("PATH", "").split(os.pathsep)
    for root in NVIDIA_DRIVER_DIRS:
        if (root / "nvidia-smi").is_file():
            if str(root) not in entries:
                os.environ["PATH"] = os.pathsep.join([str(root), *filter(None, entries)])
            return root
    return None


def nvidia_gpu_is_plausibly_present() -> bool:
    """Whether this host looks like it has an NVIDIA GPU, ignoring PATH.

    Used to tell "no GPU here, skip honestly" apart from "a GPU is sitting
    right there and the environment is hiding it", which is a misconfiguration
    worth shouting about rather than skipping past.
    """
    if any((root / "nvidia-smi").is_file() for root in NVIDIA_DRIVER_DIRS):
        return True
    return any(
        Path(node).exists() for node in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/dxg")
    )


def rocm_gpu_is_plausibly_present() -> bool:
    """Whether this host looks like it has an AMD ROCm GPU, ignoring PATH.

    Deliberately does NOT probe ``/dev/dxg``. Under WSL2 that node is the
    generic GPU paravirtualisation device and is present for NVIDIA hosts too,
    so trusting it would claim a ROCm device on The-Super-Bear. The honest
    signals are the KFD node and an installed toolkit root.
    """
    if Path("/dev/kfd").exists():
        return True
    return any(
        (Path(root) / "bin/rocminfo").is_file()
        for root in ("/opt/rocm", "/opt/rocm/core")
    )


def apple_metal_is_plausibly_present() -> bool:
    """Whether this host is an Apple-silicon Mac, which always has Metal."""
    return platform.system() == "Darwin" and platform.machine().startswith("arm")


def amx_is_plausibly_present() -> bool:
    """Whether this host advertises Intel AMX tile support.

    No box in the current fleet has AMX (Zen 5 has AVX-512 but AMX is
    Intel-only), so this is expected to be False everywhere today. It exists so
    that the day an AMX host appears, its lanes cannot silently skip.
    """
    try:
        return "amx_tile" in Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return False


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
        from tests._support import compiler_tool

        return cls(
            # `compiler_tool` owns driver resolution for the whole tree — two
            # search orders is how a fixture and a test end up running
            # different binaries in the same session.
            tessera_opt=compiler_tool.tessera_opt_path(),
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
        """Return a tessera-opt registering every named pass, else skip.

        Both the capability check and the *selection* it feeds belong to
        `tests._support.compiler_tool` — the tool's registered pass set depends
        on how it was configured, and duplicating that knowledge is how two
        call sites end up disagreeing about what a binary can do.

        This field holds only the *preferred* driver. Checking it alone would
        skip on a host that can run the test: a developer may hold a lean
        in-repo build alongside a fuller binary on PATH, and preference is not
        capability. So when the preferred driver cannot run `passes` we fall
        through to the resolver, which walks every candidate and takes the
        first capable one. The missing-binary skip stays here because this
        dataclass can be constructed with `tessera_opt=None` directly,
        independent of what the resolver would find.
        """
        from tests._support import compiler_tool

        if self.tessera_opt is None:
            pytest.skip(
                "compiler-tool test requires tessera-opt; build it or set TESSERA_OPT"
            )
        if not passes:
            return self.tessera_opt
        if compiler_tool.capability_skip_reason(self.tessera_opt, *passes) is None:
            return self.tessera_opt
        return compiler_tool.require_tessera_opt(*passes)

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
