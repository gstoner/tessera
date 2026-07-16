#!/usr/bin/env bash
# install_test_deps.sh — install the Python tooling needed to RUN the Tessera
# test suite (unit + lit + lint + type-check) into the active Python env.
#
# This is the single source of truth for the test-tooling dependency set. It is
# also what scripts/setup_ubuntu.sh calls for its venv pip step, so the two never
# drift. Linux-focused but harmless on macOS.
#
# It does NOT install the C++ build toolchain (LLVM/MLIR 23, cmake, ninja) — for
# that use scripts/setup_ubuntu.sh. It DOES check for the LLVM lit helpers
# (FileCheck / not) the MLIR lit suite needs and tells you how to get them.
#
# Usage:
#   bash scripts/install_test_deps.sh            # into the active python3
#   bash scripts/install_test_deps.sh --venv     # create/use ./.venv first
#   PYTHON=python3.11 bash scripts/install_test_deps.sh
#   bash scripts/install_test_deps.sh --check     # verify only, install nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
USE_VENV=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --venv)  USE_VENV=1 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;34m[install-test-deps]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install-test-deps] WARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install-test-deps] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Runtime numerics the tests import. numpy is capped <2.2: numpy>=2.2 ships
# stubs that break the mypy ratchet under python_version=3.10 (matches the macOS
# env and setup_ubuntu.sh). Bump only with a baseline refresh.
RUNTIME=( "numpy>=2.0,<2.2" scipy ml_dtypes pyyaml click rich tqdm )
# Test + lint + type tooling — mirrors pyproject [project.optional-dependencies]
# dev, plus `lit` (the LLVM test runner) for the MLIR fixtures under tests/.
TOOLING=( pytest pytest-cov pytest-timeout pytest-xdist hypothesis mypy ruff black isort flake8 lit )

# ---------------------------------------------------------------------------
if [[ $USE_VENV -eq 1 ]]; then
  VENV="${REPO_ROOT}/.venv"
  [[ -d "$VENV" ]] || { say "Creating venv at $VENV"; "$PY" -m venv "$VENV"; }
  # shellcheck disable=SC1091
  . "${VENV}/bin/activate"
  PY=python
fi

command -v "$PY" >/dev/null 2>&1 || die "python interpreter '$PY' not found (set PYTHON=...)"
say "Target interpreter: $("$PY" -c 'import sys; print(sys.executable)')"

if [[ $CHECK_ONLY -eq 0 ]]; then
  say "Upgrading pip"
  "$PY" -m pip install --upgrade pip >/dev/null
  say "Installing runtime numerics + test tooling (${#RUNTIME[@]}+${#TOOLING[@]} packages)"
  "$PY" -m pip install "${RUNTIME[@]}" "${TOOLING[@]}"
fi

# ---------------------------------------------------------------------------
# Verify: importable runtime deps + active pytest plugins. Non-fatal here (the
# `if` condition is exempt from set -e) so we report every gap — Python AND the
# LLVM helpers below — before exiting.
say "Verifying Python tooling"
if "$PY" - <<'PYV'
import importlib.util as u, sys
mods = {
    "numpy": "numpy", "scipy": "scipy", "ml_dtypes": "ml_dtypes",
    "pytest": "pytest", "pytest-cov": "pytest_cov",
    "pytest-timeout": "pytest_timeout", "pytest-xdist": "xdist",
    "lit": "lit",
}
missing = [name for name, mod in mods.items() if u.find_spec(mod) is None]
if missing:
    print("  MISSING:", ", ".join(missing)); sys.exit(1)
import numpy, scipy, ml_dtypes
print(f"  ok: numpy {numpy.__version__}, scipy {scipy.__version__}, "
      f"ml_dtypes {ml_dtypes.__version__}")
print("  ok: pytest + cov + timeout + xdist + lit importable")
PYV
then PY_OK=1; else PY_OK=0; fi

# CLI lint/type tools (installed as console scripts).
for tool in ruff mypy black isort flake8; do
  command -v "$tool" >/dev/null 2>&1 || warn "$tool not on PATH (pip installed it; ensure the env's bin/ is on PATH)"
done

# ---------------------------------------------------------------------------
# The MLIR lit suite needs FileCheck + `not` from LLVM — NOT pip-installable.
say "Checking LLVM lit helpers (FileCheck / not) for the MLIR fixtures"
if command -v FileCheck >/dev/null 2>&1 && command -v not >/dev/null 2>&1; then
  say "  ok: FileCheck + not on PATH ($(command -v FileCheck))"
else
  # Common case on a dev box: they ARE installed (apt llvm-NN-tools / Homebrew
  # llvm) but the LLVM bin/ isn't on PATH. Find a bindir that has BOTH and print
  # the exact export — far more useful than "reinstall LLVM".
  FOUND_BIN=""
  # Prefer an llvm-config's own bindir (matches the version we build against).
  for cfg in llvm-config-${LLVM_VERSION:-23} llvm-config; do
    command -v "$cfg" >/dev/null 2>&1 || continue
    d="$("$cfg" --bindir 2>/dev/null || true)"
    if [[ -x "$d/FileCheck" && -x "$d/not" ]]; then FOUND_BIN="$d"; break; fi
  done
  if [[ -z "$FOUND_BIN" ]]; then
    for d in /usr/lib/llvm-*/bin /usr/local/opt/llvm/bin /opt/homebrew/opt/llvm/bin; do
      if [[ -x "$d/FileCheck" && -x "$d/not" ]]; then FOUND_BIN="$d"; break; fi
    done
  fi
  if [[ -n "$FOUND_BIN" ]]; then
    warn "FileCheck + not are INSTALLED at ${FOUND_BIN} but not on PATH. Add it:"
    warn "    export PATH=\"${FOUND_BIN}:\$PATH\""
  else
    warn "MLIR lit tests (tests/tessera-ir, tests/.../test/) need FileCheck + 'not'"
    warn "from LLVM. Install them via:  bash scripts/setup_ubuntu.sh   (apt.llvm.org"
    warn "LLVM 23), or 'apt-get install llvm-23-tools', then add its bin/ to PATH."
  fi
  warn "Pure-Python unit tests (pytest tests/unit) do NOT need these."
fi

if [[ "${PY_OK}" -ne 1 ]]; then
  die "Python test tooling incomplete (see MISSING above)."$'\n'"  Fix: bash scripts/install_test_deps.sh   (drop --check to install)."
fi

say "Done. Run the suite with:"
echo "    export PYTHONPATH=${REPO_ROOT}/python"
echo "    python scripts/run_unit_tests.py            # auto-sized parallel unit run"
echo "    python -m lit tests/tessera-ir/ -v          # MLIR lit (needs FileCheck/not)"
