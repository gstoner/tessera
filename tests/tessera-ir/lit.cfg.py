
# -*- Python -*-
"""Lit configuration for the Tessera IR phase 2–8 fixtures.

Discovers ``*.mlir`` files and substitutes ``tessera-opt``.  The
substitution prefers an absolute path so the suite runs cleanly
without the caller needing to export ``PATH`` first.  Probe order:

  1. ``$TESSERA_OPT`` environment variable (caller override),
  2. ``<repo>/build/tools/tessera-opt/tessera-opt`` (the canonical
     local build location used by
     ``cmake --build build --target tessera-opt``),
  3. plain ``tessera-opt`` resolved against the caller's PATH,
  4. common LLVM 22 bin directories such as Homebrew and
     ``/usr/lib/llvm-22/bin``,
  5. plain ``tessera-opt`` as the literal substitution (leaves any
     resolution failure to the test process).

``FileCheck`` follows the same probe order so the lit RUN lines
can pipe into it on a host without LLVM bin on PATH.
"""

import os, shutil, subprocess, lit.formats


# Common Homebrew / system LLVM install locations to probe when
# the binary isn't on PATH.  Pinned to MLIR 22.
_LLVM_BIN_HINTS = (
    "/opt/homebrew/opt/llvm/bin",
    "/opt/homebrew/opt/llvm@22/bin",
    "/usr/local/opt/llvm/bin",
    "/usr/local/opt/llvm@22/bin",
    "/usr/lib/llvm-22/bin",
)


def _resolve(env_var: str, repo_relative: str, fallback: str) -> str:
    """Pick the first available location for a build binary."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
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


config.name = "Tessera-IR v0.3.1"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.environment['PATH'] = os.environ.get('PATH', '')

_TESSERA_OPT = _resolve(
    "TESSERA_OPT", "build/tools/tessera-opt/tessera-opt", "tessera-opt",
)
_FILECHECK = _resolve("FILECHECK", "", "FileCheck")

config.substitutions.append(
    ('tessera-opt', f'"{_TESSERA_OPT}" -allow-unregistered-dialect')
)
config.substitutions.append(('%tessera_strict_opt', f'"{_TESSERA_OPT}"'))
config.substitutions.append(('FileCheck', f'"{_FILECHECK}"'))

# Probe tessera-opt for optional backends so per-target fixtures can
# REQUIRE the right feature. We only mark the feature as available if
# the corresponding pipeline alias is actually registered in this build.
def _opt_help_contains(needle: str) -> bool:
    if not (os.path.isfile(_TESSERA_OPT) and os.access(_TESSERA_OPT, os.X_OK)):
        return False
    try:
        out = subprocess.run([_TESSERA_OPT, "--help"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return False
    return needle in out

if _opt_help_contains("tessera-lower-to-metalium"):
    config.available_features.add("tessera-metalium-backend")
if _opt_help_contains("tessera-lower-to-rocm"):
    config.available_features.add("tessera-rocm-backend")
