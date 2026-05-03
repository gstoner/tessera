#!/usr/bin/env python3
"""Verify Tessera version declarations stay aligned."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _match(pattern: str, text: str, label: str) -> str:
    found = re.search(pattern, text, re.MULTILINE)
    if not found:
        raise SystemExit(f"missing version declaration: {label}")
    return found.group(1)


def main() -> int:
    cmake = _match(r"project\(\s*Tessera\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", _read("CMakeLists.txt"), "CMake project")
    pyproject = _match(r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', _read("pyproject.toml"), "pyproject")
    init_py = _match(r'^__version__\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', _read("python/tessera/__init__.py"), "python package")

    header = _read("src/runtime/include/tessera/tsr_version.h")
    major = _match(r"#define\s+TESSERA_VERSION_MAJOR\s+([0-9]+)", header, "runtime major")
    minor = _match(r"#define\s+TESSERA_VERSION_MINOR\s+([0-9]+)", header, "runtime minor")
    patch = _match(r"#define\s+TESSERA_VERSION_PATCH\s+([0-9]+)", header, "runtime patch")
    runtime = f"{major}.{minor}.{patch}"

    versions = {
        "CMakeLists.txt": cmake,
        "pyproject.toml": pyproject,
        "python/tessera/__init__.py": init_py,
        "src/runtime/include/tessera/tsr_version.h": runtime,
    }
    unique = set(versions.values())
    if len(unique) != 1:
        details = "\n".join(f"  {path}: {version}" for path, version in versions.items())
        raise SystemExit(f"Tessera version declarations drifted:\n{details}")
    print(f"Tessera version declarations aligned: {cmake}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
