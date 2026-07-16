"""Lit configuration for the TPP solver fixtures.

Discovers `*.mlir` files and substitutes `tessera-opt`.  The
substitution is the absolute path to a working build of
`tessera-opt` so the suite runs cleanly without the caller
needing to export ``PATH`` first.  We probe (in order):

  1. ``$TESSERA_OPT`` environment variable (caller override),
  2. ``<repo>/build/tools/tessera-opt/tessera-opt`` (the
     canonical local build location used by
     ``cmake --build build --target tessera-opt``),
  3. plain ``tessera-opt`` resolved against the caller's PATH,
  4. common LLVM 23 bin directories such as Homebrew, TheRock, and
     ``/usr/lib/llvm-23/bin``,
  5. plain ``tessera-opt`` as the literal substitution (leaves any
     resolution failure to the test process).

The same probe order is used for ``FileCheck`` — the lit RUN
lines pipe into it, so it has to be findable too.
"""

import os
import shutil
import lit.formats


# Common Homebrew / system LLVM install locations to probe when
# the binary isn't on PATH. Pinned to LLVM/MLIR 23.
_LLVM_BIN_HINTS = (
    "/opt/homebrew/opt/llvm/bin",
    "/opt/homebrew/opt/llvm@23/bin",
    "/usr/local/opt/llvm/bin",
    "/usr/local/opt/llvm@23/bin",
    "/opt/rocm/core/lib/llvm/bin",
    "/usr/lib/llvm-23/bin",
)


def _resolve(env_var: str, repo_relative: str, fallback: str) -> str:
    """Pick the first available location for a build binary."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    if env_var in os.environ:
        override = os.environ[env_var]
        # Caller-provided overrides must absolutize against the repo
        # root.  Lit runs each fixture from its ``Output/`` subdir, so
        # a relative path like ``build/tools/tessera-opt/tessera-opt``
        # would resolve to ``Output/build/tools/.../tessera-opt`` and
        # fail.  Honor absolute paths verbatim; expand a leading ``~``;
        # otherwise join against the repo root.
        if override.startswith("~"):
            override = os.path.expanduser(override)
        if not os.path.isabs(override):
            override = os.path.abspath(os.path.join(repo_root, override))
        return override
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
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = ['.mlir']
config.environment['PATH'] = os.environ.get('PATH', '')

# Resolution order for the opt driver:
#   1. explicit $TESSERA_OPT override (verbatim / repo-relative),
#   2. the dedicated standalone TPP driver `tessera-tpp-opt` ($TESSERA_TPP_OPT
#      or on PATH) — preferred because it always has the TPP passes,
#   3. the monorepo `tessera-opt` (only carries TPP when built with
#      TESSERA_HAVE_TPP).
# `tessera-tpp-opt` wins over a generic `tessera-opt` because the latter may be
# a monorepo build without the TPP dialect/passes registered.
if "TESSERA_OPT" in os.environ:
    _TESSERA_OPT = _resolve("TESSERA_OPT", "", "tessera-opt")
else:
    _TESSERA_OPT = os.environ.get("TESSERA_TPP_OPT") or shutil.which(
        "tessera-tpp-opt")
    if not (_TESSERA_OPT and os.path.isfile(_TESSERA_OPT)):
        _TESSERA_OPT = _resolve(
            "TESSERA_OPT", "build/tools/tessera-opt/tessera-opt", "tessera-opt")
_FILECHECK = _resolve("FILECHECK", "", "FileCheck")

# `-allow-unregistered-dialect` matches the parent suite's
# substitution at tests/tessera-ir/lit.cfg.py — TPP fixtures inherit
# that convention so an in-place migration to the unified suite
# wouldn't change their RUN lines.
config.substitutions.append(
    ('tessera-opt', f'"{_TESSERA_OPT}" -allow-unregistered-dialect')
)
config.substitutions.append(('FileCheck', f'"{_FILECHECK}"'))
