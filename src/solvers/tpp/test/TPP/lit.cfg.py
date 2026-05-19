"""Lit configuration for the TPP solver fixtures.

Discovers `*.mlir` files and substitutes `tessera-opt`.  The
substitution is the absolute path to a working build of
`tessera-opt` so the suite runs cleanly without the caller
needing to export ``PATH`` first.  We probe (in order):

  1. ``$TESSERA_OPT`` environment variable (caller override),
  2. ``<repo>/build/tools/tessera-opt/tessera-opt`` (the
     canonical local build location used by
     ``cmake --build build --target tessera-opt``),
  3. plain ``tessera-opt`` (relies on the caller's PATH).

The same probe order is used for ``FileCheck`` — the lit RUN
lines pipe into it, so it has to be findable too.
"""

import os
import shutil
import lit.formats


# Common Homebrew / system LLVM install locations to probe when
# the binary isn't on PATH.  Pinned to MLIR 21 — see CLAUDE.md
# build-pin note.
_LLVM_BIN_HINTS = (
    "/opt/homebrew/opt/llvm@21/bin",
    "/opt/homebrew/opt/llvm/bin",
    "/usr/local/opt/llvm@21/bin",
    "/usr/local/opt/llvm/bin",
    "/usr/lib/llvm-21/bin",
)


def _resolve(env_var: str, repo_relative: str, fallback: str) -> str:
    """Pick the first available location for a build binary."""
    if env_var in os.environ:
        return os.environ[env_var]
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    if repo_relative:
        candidate = os.path.join(repo_root, repo_relative)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    on_path = shutil.which(fallback)
    if on_path is not None:
        return on_path
    for hint in _LLVM_BIN_HINTS:
        candidate = os.path.join(hint, fallback)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return fallback


config.name = "Tessera-TPP"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.environment['PATH'] = os.environ.get('PATH', '')

_TESSERA_OPT = _resolve(
    "TESSERA_OPT", "build/tools/tessera-opt/tessera-opt", "tessera-opt",
)
_FILECHECK = _resolve("FILECHECK", "", "FileCheck")

# `-allow-unregistered-dialect` matches the parent suite's
# substitution at tests/tessera-ir/lit.cfg.py — TPP fixtures inherit
# that convention so an in-place migration to the unified suite
# wouldn't change their RUN lines.
config.substitutions.append(
    ('tessera-opt', f'"{_TESSERA_OPT}" -allow-unregistered-dialect')
)
config.substitutions.append(('FileCheck', f'"{_FILECHECK}"'))
