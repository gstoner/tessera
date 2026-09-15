"""Locate the matched LLVM/MLIR 23 companion tools on any fleet host.

Every native lane shells out to ``mlir-opt`` / ``mlir-translate`` / ``llc`` /
``llvm-link`` from the *same* LLVM major the driver was built against. The
canonical prefixes differ per host -- apt.llvm.org on the Ubuntu/WSL2 boxes,
Homebrew's ``llvm`` keg on the Mac, a from-source assertions prefix on
Tajasarus -- so a hard-coded ``/usr/lib/llvm-23/bin`` turned every native test
into a false failure off Ubuntu (2026-09-15). Resolution order:

1. ``TESSERA_LLVM_BIN`` (a bin directory) when set.
2. A per-tool override such as ``MLIR_OPT`` / ``LLC`` when the caller passes it.
3. The canonical fleet prefixes, in order, when their ``llvm-config`` reports
   major 23 (or when no ``llvm-config`` is present alongside).
4. ``llvm-config`` on ``PATH`` reporting major 23.
5. The bare tool on ``PATH``.

Nothing here fabricates a toolchain: a miss returns ``None`` and the caller
decides whether that is a skip or an error.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

REQUIRED_MAJOR = 23

CANONICAL_BIN_DIRS: tuple[str, ...] = (
    "/usr/lib/llvm-23/bin",            # apt.llvm.org (Princess-Luna, Super-Bear)
    "/opt/homebrew/opt/llvm/bin",      # Homebrew keg, LLVM 23 since 2026-08-28 (Mac)
    "/opt/homebrew/opt/llvm@23/bin",
    "/usr/local/opt/llvm/bin",
)


def _major_of(llvm_config: Path) -> int | None:
    try:
        out = subprocess.run([str(llvm_config), "--version"], capture_output=True,
                             text=True, timeout=20, check=False).stdout.strip()
        return int(out.split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _accept(bin_dir: Path) -> bool:
    if not bin_dir.is_dir():
        return False
    cfg = bin_dir / "llvm-config"
    if cfg.is_file():
        return _major_of(cfg) == REQUIRED_MAJOR
    return True  # canonical dir without llvm-config (minimal package) -- trust the path


@lru_cache(maxsize=1)
def llvm_bin_dir() -> Path | None:
    """The bin directory holding the matched LLVM 23 tools, or ``None``."""
    configured = os.environ.get("TESSERA_LLVM_BIN")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_dir() else None
    for cand in CANONICAL_BIN_DIRS:
        if _accept(Path(cand)):
            return Path(cand)
    found = shutil.which("llvm-config")
    if found and _major_of(Path(found)) == REQUIRED_MAJOR:
        return Path(found).resolve().parent
    return None


def find_llvm_tool(name: str, env: str | None = None) -> Path | None:
    """Path to ``name`` (e.g. ``mlir-opt``), honouring an optional per-tool env var."""
    if env:
        configured = os.environ.get(env)
        if configured:
            path = Path(configured).expanduser()
            return path if path.is_file() else None
    bin_dir = llvm_bin_dir()
    if bin_dir is not None and (bin_dir / name).is_file():
        return bin_dir / name
    found = shutil.which(name)
    return Path(found) if found else None
