"""Backend ownership and capability selection for compiler-artifact tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests._support.policy import APPLE_HOST_FREE_COMPILER_EXPRESSION


_BACKEND_CACHE_KEYS = {
    "apple": "TESSERA_BUILD_APPLE_BACKEND",
    "nvidia": "TESSERA_BUILD_NVIDIA_BACKEND",
    "rocm": "TESSERA_BUILD_ROCM_BACKEND",
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
