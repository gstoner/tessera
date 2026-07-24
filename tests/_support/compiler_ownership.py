"""Backend ownership and capability selection for compiler-artifact tests."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from tests._support.policy import (
    APPLE_HOST_FREE_COMPILER_EXPRESSION,
    ROCM_HOST_FREE_COMPILER_EXPRESSION,
)


_BACKEND_CACHE_KEYS = {
    "apple": "TESSERA_BUILD_APPLE_BACKEND",
    "nvidia": "TESSERA_BUILD_NVIDIA_BACKEND",
    "rocm": "TESSERA_BUILD_ROCM_BACKEND",
}

COMPILER_TEST_PLATFORM_ENV = "TESSERA_COMPILER_TEST_PLATFORM"

# An owner marker is an executable-host requirement, not merely a source-tree
# label.  Keep the display name in the skip reason so a mixed compiler suite
# tells its operator exactly where the excluded proof belongs.
_OWNER_MARKER_PLATFORMS = (
    ("compiler_avx512", "avx512", "AVX512"),
    ("compiler_x86", "x86", "X86"),
    ("compiler_cuda", "cuda", "CUDA"),
    ("compiler_nvidia", "cuda", "CUDA"),
    ("compiler_rocm", "rocm", "ROCm"),
    ("compiler_apple", "apple", "Apple"),
    ("compiler_cpu", "x86", "X86"),
)

_PLATFORM_ALIASES = {
    "apple": "apple",
    "metal": "apple",
    "cuda": "cuda",
    "nvidia": "cuda",
    "rocm": "rocm",
    "x86": "x86",
    "avx512": "avx512",
}


@dataclass(frozen=True)
class CompilerBuildCapabilities:
    """Compiler backend families explicitly enabled in one CMake build."""

    apple: bool
    nvidia: bool
    rocm: bool

    @classmethod
    def from_cmake_cache(cls, build_dir: Path) -> "CompilerBuildCapabilities":
        cache = build_dir / "CMakeCache.txt"
        if not cache.is_file():
            raise FileNotFoundError(f"CMake cache not found: {cache}")
        values: dict[str, bool] = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            for family, key in _BACKEND_CACHE_KEYS.items():
                prefix = f"{key}:BOOL="
                if line.startswith(prefix):
                    values[family] = line.removeprefix(prefix).strip().upper() == "ON"
        missing = sorted(set(_BACKEND_CACHE_KEYS) - set(values))
        if missing:
            raise ValueError(
                f"CMake cache does not declare compiler backend option(s): {', '.join(missing)}"
            )
        return cls(**values)

    def as_dict(self) -> dict[str, bool]:
        return {"apple": self.apple, "nvidia": self.nvidia, "rocm": self.rocm}


def apple_host_free_compiler_expression() -> str:
    """Select compiler proofs supplied by an Apple-only compiler build."""

    return APPLE_HOST_FREE_COMPILER_EXPRESSION


def rocm_host_free_compiler_expression() -> str:
    """Select generic and ROCm-owned proofs supported by a ROCm-only build."""

    return ROCM_HOST_FREE_COMPILER_EXPRESSION


def selected_compiler_test_platform() -> str | None:
    """Return the explicitly selected compiler-test platform, if any."""

    raw = os.environ.get(COMPILER_TEST_PLATFORM_ENV)
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower().replace("-", "")
    try:
        return _PLATFORM_ALIASES[normalized]
    except KeyError as error:
        choices = ", ".join(("Apple", "CUDA", "ROCm", "X86", "AVX512"))
        raise ValueError(
            f"{COMPILER_TEST_PLATFORM_ENV}={raw!r} is invalid; choose {choices}"
        ) from error


def compiler_test_required_platform(item: object) -> tuple[str, str] | None:
    """Return the canonical and display platform required by an owned test."""

    get_closest_marker = getattr(item, "get_closest_marker")
    for marker, platform, display_name in _OWNER_MARKER_PLATFORMS:
        if get_closest_marker(marker) is not None:
            return platform, display_name
    return None


def compiler_platform_skip_reason(required_display_name: str) -> str:
    """Stable, countable reason for a test owned by another compiler system."""

    return (
        f"compiler-test platform mismatch: requires {required_display_name}; "
        f"run on a {required_display_name} system"
    )
