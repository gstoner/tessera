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
# Runtime numerics the tests import.
#
# **The `<2.2` numpy cap was removed 2026-09-01, and it was not cosmetic.** It
# existed because numpy>=2.2 ships PEP 695 `type` statements in its stubs that
# broke the mypy ratchet under python_version=3.10 — a reason that stopped
# applying on 2026-08-28, when the ratchet was fixed with `follow_imports`
# overrides in pyproject instead (CLAUDE.md records the lift; this file was
# never updated). Every Ubuntu box provisioned in between got numpy 2.1.3.
#
# Measured cost of that gap: on Princess-Luna, same commit and same build, only
# numpy differing — 16 failed at 2.1.3, 9 failed at 2.5.2. Nine failures were
# the pin, including a family that presented as an empty reference array
# (`operands could not be broadcast together with shapes (524288,) (0,)`) and
# read like a compiler defect. The floor is 2.5 because that is the version
# actually verified; the macOS box, which never runs this script, is on 2.5.2
# and sweeps clean.
RUNTIME=( "numpy>=2.5" scipy ml_dtypes pyyaml click rich tqdm )
# Test + lint + type tooling — mirrors pyproject [project.optional-dependencies]
# dev, plus `lit` (the LLVM test runner) for the MLIR fixtures under tests/.
#
# `lit` is bounded to the LLVM major we build against. It is pip-installed, so
# an old resolution SHADOWS the correct `/usr/lib/llvm-<N>/bin/lit`: The
# Super-Bear was running pip lit 18.1.8 in its venv against LLVM 23 fixtures
# while the matching 23.1.0 runner sat unused on disk.
TOOLING=( pytest pytest-cov pytest-timeout pytest-xdist hypothesis mypy ruff black isort flake8 "lit>=23,<24" )

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
  # --upgrade is load-bearing, not tidiness. Without it pip leaves an
  # already-satisfied unpinned package alone, so two boxes provisioned by this
  # same script weeks apart drift permanently and never re-converge. Measured
  # 2026-09-01: Princess-Luna and The Super-Bear disagreed on isort (9.0.1 vs
  # 8.0.1), ruff (0.16.5 vs 0.16.4), scipy (1.18.1 vs 1.18.0) and lit (23.1.0
  # vs 18.1.8) with identical LLVM/MLIR and an identical invocation of this
  # file. Re-running it must actually converge an existing env.
  "$PY" -m pip install --upgrade "${RUNTIME[@]}" "${TOOLING[@]}"
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
# Print the versions that have actually drifted between fleet boxes, so a
# mismatch is visible in the provisioning log instead of being discovered
# later as a test failure. A silent install tells you nothing; this is the
# cheapest place to make two machines comparable.
import importlib.metadata as md
# pip-managed only. The lint/type tools are reported below from the BINARY
# that will actually run, because on macOS they come from Homebrew rather
# than pip and would read as MISSING here while being perfectly present.
drifty = ("numpy", "scipy", "pytest", "hypothesis", "lit", "coverage")
seen = []
for name in drifty:
    try:
        seen.append(f"{name} {md.version(name)}")
    except md.PackageNotFoundError:
        seen.append(f"{name} MISSING")
print("  versions: " + ", ".join(seen))
PYV
then PY_OK=1; else PY_OK=0; fi

# CLI lint/type tools (installed as console scripts, or from the system package
# manager on macOS). Report the version of the binary that will actually run —
# these are the tools that drifted between fleet boxes, and a version in the
# provisioning log is what makes two machines comparable after the fact.
for tool in ruff mypy black isort flake8; do
  if command -v "$tool" >/dev/null 2>&1; then
    say "  $tool $("$tool" --version 2>&1 | head -1) ($(command -v "$tool"))"
  else
    warn "$tool not on PATH (pip installed it; ensure the env's bin/ is on PATH)"
  fi
done

# ---------------------------------------------------------------------------
# The MLIR lit suite needs FileCheck + `not` from LLVM — NOT pip-installable.
say "Checking LLVM lit helpers (FileCheck / not) for the MLIR fixtures"
if command -v FileCheck >/dev/null 2>&1 && command -v not >/dev/null 2>&1; then
  say "  ok: FileCheck + not on PATH ($(command -v FileCheck))"
else
  # Common case on a dev box: LLVM 23 is installed but its bin/ is not on PATH.
  # Find a canonical bindir that has BOTH tools and print
  # the exact export — far more useful than "reinstall LLVM".
  FOUND_BIN=""
  # Prefer an llvm-config's own bindir (matches the version we build against).
  for cfg in llvm-config-23; do
    command -v "$cfg" >/dev/null 2>&1 || continue
    d="$("$cfg" --bindir 2>/dev/null || true)"
    if [[ -x "$d/FileCheck" && -x "$d/not" ]]; then FOUND_BIN="$d"; break; fi
  done
  if [[ -z "$FOUND_BIN" ]]; then
    for d in /usr/lib/llvm-23/bin /usr/local/opt/llvm@23/bin /opt/homebrew/opt/llvm@23/bin; do
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
