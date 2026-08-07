#!/usr/bin/env bash
# Shared interpreter resolver for Tessera's shell gates. Source it; it exports
# TESSERA_PY (and PYTHON, for scripts that already honor that name).
#
#   source "$(dirname "$0")/_python_env.sh"
#   "${TESSERA_PY}" scripts/check_spec_sync.py
#
# Why this exists
# ---------------
# The gate scripts import `tessera.compiler.*`, which imports numpy. A bare
# `python3` is NOT sufficient: whichever python3 happens to be first on PATH may
# have no numpy, and then a gate that would have PASSED dies on
# `ModuleNotFoundError` and reads as drift. That is the worst failure mode for a
# gate — a green check reported as red pushes people toward
# TESSERA_SKIP_DOC_DRIFT=1, which skips a check that was actually fine.
#
# Concretely: on the Mac fleet box the first `python3` on PATH is the python.org
# framework build (no numpy) while the working interpreter is Homebrew's; on the
# Ubuntu/Strix Halo box it is the project-local `.venv`. So the resolver picks by
# *capability* (can it import numpy?) rather than by path.
#
# Resolution order — first candidate that can `import numpy` wins:
#   1. $TESSERA_PYTHON   explicit override (checked, not blindly trusted)
#   2. $PYTHON           pre-existing convention in check_generated_docs.sh
#   3. $VIRTUAL_ENV      the activated venv, if any
#   4. <repo>/.venv      the Ubuntu setup_ubuntu.sh venv
#   5. python3 on PATH   the common case when it is already correct
#   6. versioned/Homebrew fallbacks
#
# If nothing qualifies, this FAILS CLOSED with an actionable message rather than
# letting a gate report a false negative.

_tessera_py_repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || (
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
  )
}

# Does this interpreter exist and satisfy the gates' import requirements?
_tessera_py_usable() {
  local py="${1:-}"
  [ -n "${py}" ] || return 1
  command -v "${py}" >/dev/null 2>&1 || return 1
  "${py}" -c 'import numpy' >/dev/null 2>&1
}

_tessera_py_resolve() {
  local root candidate
  root="$(_tessera_py_repo_root)"

  # An explicit override is still verified — a stale TESSERA_PYTHON pointing at
  # an interpreter without numpy should say so, not fail obscurely later.
  if [ -n "${TESSERA_PYTHON:-}" ]; then
    if _tessera_py_usable "${TESSERA_PYTHON}"; then
      printf '%s\n' "${TESSERA_PYTHON}"
      return 0
    fi
    echo "[tessera] TESSERA_PYTHON=${TESSERA_PYTHON} cannot import numpy; ignoring it." >&2
  fi

  for candidate in \
      "${PYTHON:-}" \
      "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python3}" \
      "${root}/.venv/bin/python3" \
      python3 \
      python3.14 python3.13 python3.12 python3.11 python3.10 \
      /opt/homebrew/bin/python3 \
      /usr/local/bin/python3 \
      /usr/bin/python3
  do
    if _tessera_py_usable "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! TESSERA_PY="$(_tessera_py_resolve)"; then
  cat >&2 <<'MSG'
[tessera] No usable python3 found — every candidate failed `import numpy`.

  The gates import tessera.compiler.*, which needs numpy. This is an
  environment problem, NOT a drift failure, so do not reach for
  TESSERA_SKIP_DOC_DRIFT=1 — that would skip a check you have not run.

  Fix (pick the one matching your box):
    Ubuntu / Strix Halo : source .venv/bin/activate      # or: bash scripts/setup_ubuntu.sh
    macOS (Homebrew)    : export TESSERA_PYTHON=/opt/homebrew/bin/python3
    any                 : export TESSERA_PYTHON=/path/to/python3   # must have numpy

  Verify with:  "$TESSERA_PYTHON" -c 'import numpy; print(numpy.__version__)'
MSG
  return 1 2>/dev/null || exit 1
fi

export TESSERA_PY
# Exported so child scripts that already honor $PYTHON (check_generated_docs.sh)
# inherit the same interpreter instead of re-resolving to a different one.
export PYTHON="${TESSERA_PY}"
