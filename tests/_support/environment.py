"""One source of truth for compiler tools and child-process environments."""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
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

    Every signal must be NVIDIA-*specific*. ``/dev/dxg`` was accepted here
    until 2026-08-30 and is not: under WSL2 it is the generic GPU
    paravirtualisation node, present whatever the vendor. Measured that day --
    Princess-Luna (AMD gfx1151, no NVIDIA anything) has ``/dev/dxg`` and
    therefore claimed an NVIDIA GPU, which would fail any session on that box
    that collected NVIDIA lanes and honestly skipped them. The WSL driver shim
    (``/usr/lib/wsl/lib/nvidia-smi``) is vendor-specific and is what The-Super-Bear
    is recognised by; Princess-Luna does not have it.
    """
    if any((root / "nvidia-smi").is_file() for root in NVIDIA_DRIVER_DIRS):
        return True
    return any(Path(node).exists() for node in ("/dev/nvidiactl", "/dev/nvidia0"))


@functools.lru_cache(maxsize=1)
def rocm_gpu_is_plausibly_present() -> bool:
    """Whether this host actually has an AMD ROCm GPU, ignoring PATH.

    Two things this must NOT infer a device from, both of which produce a
    false claim that then fails an otherwise-valid run:

    * ``/dev/dxg`` -- under WSL2 that is the generic GPU paravirtualisation
      node and is present on the NVIDIA box too, so it would claim a ROCm
      device on The-Super-Bear.
    * **an installed toolkit** -- a build or NVIDIA host may carry
      ``/opt/rocm`` purely to compile, with no AMD device anywhere.

    ``/dev/kfd`` is a genuine signal (the amdgpu/KFD driver creates it, not
    the toolkit) but is *not sufficient on its own*: under WSL2 it does not
    exist at all, and Princess-Luna -- the fleet's only ROCm box -- has no
    ``/dev/kfd`` while running gfx1151 happily. Requiring it would disable
    this probe precisely where it is needed.

    So the authority is ``rocminfo`` itself: run it and require a
    ``Device Type: GPU`` agent. A toolkit-only host reports CPU agents alone
    or fails outright. Cached, because the answer cannot change within a
    session and the subprocess is not free.
    """
    if Path("/dev/kfd").exists():
        return True
    binary = shutil.which("rocminfo") or next(
        (
            str(candidate)
            for candidate in (
                Path(root) / "bin/rocminfo" for root in ("/opt/rocm", "/opt/rocm/core")
            )
            if candidate.is_file()
        ),
        None,
    )
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    return any(
        line.split(":", 1)[1].strip() == "GPU"
        for line in result.stdout.splitlines()
        if line.strip().startswith("Device Type:")
    )


def apple_metal_is_plausibly_present() -> bool:
    """Whether this host is an Apple-silicon Mac, which always has Metal."""
    return platform.system() == "Darwin" and platform.machine().startswith("arm")


def avx512_is_plausibly_present() -> bool:
    """Whether this host advertises AVX-512.

    This is the x86 lane that actually exists. AMX is retired (superseded by
    ACE) and no fleet box has it, so `hardware_amx` can never stand in for
    "x86 hardware" -- doing so skips the test on Princess-Luna, the only host
    that can run it. `avx512f` is the base feature every other AVX-512 subset
    implies, so it is the right thing to key on.
    """
    try:
        return "avx512f" in Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def apple_metal4_is_plausibly_present() -> bool:
    """Whether a *Metal 4* runtime is actually available, not merely Metal.

    Being Apple silicon is the wrong question for the Metal 4 lane. Metal 4
    additionally needs a runtime that reports the capability -- macOS 26.5.1
    exposes Metal 4.0, and parts of the surface (8-bit matrix ops) are gated
    behind macOS 27.0 -- so a Metal-capable Mac can honestly skip every
    ``metal4`` test. Judging that lane by the generic Apple-silicon probe
    would turn a correct capability skip into a session failure.

    Mirrors the gate in ``tests._support.apple.require_apple_metal4`` so the
    presence probe and the skip decision cannot drift apart. The import is
    lazy and broadly guarded because this runs at session end on hosts where
    the runtime may not import at all.
    """
    if not apple_metal_is_plausibly_present():
        return False
    try:
        from tessera import runtime

        return bool(runtime.apple_gpu_metal4_caps().get("available"))
    except Exception:
        return False


def amx_is_plausibly_present() -> bool:
    """Whether this host advertises Intel AMX tile support.

    Expected to be False everywhere, permanently. **AMX is a dead end and is
    not a Tessera target**: it was Intel-only, no fleet box has it (Zen 5 has
    AVX-512; the Core Ultra 7 265F has neither), and it is superseded by ACE
    (AI Compute Extensions), the joint AMD/Intel matrix spec. This probe is
    not a roadmap placeholder -- it exists only so the `hardware_amx` marker
    already gating `tests/device/x86/test_amx_*.py` has a consumer
    (Decision #29). The x86 lane that needs marker coverage is AVX-512.
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
