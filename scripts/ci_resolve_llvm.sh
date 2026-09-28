#!/usr/bin/env bash
# Resolve the LLVM/MLIR toolchain a hosted CI lane will build against, or FAIL.
#
# Owner decision (2026-09-27, sync FOUNDATION-BATCH-2-2026-09-27): hosted CI
# accepts ANY patch release in the fleet pin's major.minor series (23.1.x),
# records the exact version it ran, and FAILS -- never skips -- when no such
# toolchain is present. The fleet boxes keep the EXACT pin in
# cmake/TesseraToolchainPins.cmake; this script is the CI-only tolerance and
# pairs with `-DTESSERA_LLVM_PIN_MODE=minor` at configure time.
#
# Why this exists: until 2026-09-27 the `lit` and `rocm-serialize` lanes
# compared apt.llvm.org's rolling 23.1.2 against the exact 23.1.1 pin, printed
# a ::warning, skipped configure/build/test, and reported SUCCESS having tested
# nothing (push run 36347063229 on main). A toolchain mismatch is a lane
# failure or a recorded tolerance -- never a green skip.
#
# Usage:
#   scripts/ci_resolve_llvm.sh --lane <name> [--require-lld] [--manifest <path>]
#
# Environment:
#   TESSERA_CI_LLVM_PREFIX  toolchain prefix (default /usr/lib/llvm-23)
#   GITHUB_OUTPUT           when set, receives llvm_version / mlir_version /
#                           llvm_prefix / fleet_pin / pin_match
#   GITHUB_STEP_SUMMARY     when set, receives a markdown record of the result
#
# Exit status: 0 only when a matched 23.1.x LLVM + MLIR (+ ld.lld if asked) is
# present and usable. Anything else is exit 1 with a ::error annotation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PINS_FILE="$REPO_ROOT/cmake/TesseraToolchainPins.cmake"

lane=""
require_lld=0
manifest=""
while (( $# > 0 )); do
  case "$1" in
    --lane) lane="${2:-}"; shift 2 ;;
    --require-lld) require_lld=1; shift ;;
    --manifest) manifest="${2:-}"; shift 2 ;;
    *) echo "::error ::ci_resolve_llvm.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$lane" ]]; then
  echo "::error ::ci_resolve_llvm.sh: --lane is required" >&2
  exit 2
fi

prefix="${TESSERA_CI_LLVM_PREFIX:-/usr/lib/llvm-23}"

summary() {
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf '%s\n' "$@" >> "$GITHUB_STEP_SUMMARY"
  fi
}

fail() {
  local why="$1"
  echo "::error ::[$lane] LLVM/MLIR toolchain unusable: $why -- this lane FAILS rather than skipping (hosted CI accepts any ${series:-<pin series>}.x patch; see scripts/ci_resolve_llvm.sh)" >&2
  summary "### $lane: LLVM/MLIR toolchain" "" \
    "**FAILED** -- $why" "" \
    "Fleet pin: \`${fleet_pin:-unknown}\`; CI accepts \`${series:-?}.x\`; prefix \`$prefix\`."
  exit 1
}

fleet_pin="$(sed -n \
  's/^set(TESSERA_REQUIRED_LLVM_VERSION[[:space:]]*"\([0-9.]*\)".*/\1/p' \
  "$PINS_FILE" 2>/dev/null || true)"
if [[ ! "$fleet_pin" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  series=""
  fail "could not read TESSERA_REQUIRED_LLVM_VERSION from cmake/TesseraToolchainPins.cmake (got '${fleet_pin}')"
fi
series="${fleet_pin%.*}"

version_of() {
  # Print the first X.Y.Z in the tool's --version output, or nothing.
  local tool="$1"
  [[ -x "$tool" ]] || return 0
  "$tool" --version 2>/dev/null \
    | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true
}

llvm_version="$(version_of "$prefix/bin/llvm-config")"
mlir_version="$(version_of "$prefix/bin/mlir-opt")"

[[ -n "$llvm_version" ]] || fail "no runnable llvm-config under $prefix/bin"
[[ -n "$mlir_version" ]] || fail "no runnable mlir-opt under $prefix/bin (LLVM=$llvm_version)"
[[ -f "$prefix/lib/cmake/llvm/LLVMConfig.cmake" ]] \
  || fail "LLVMConfig.cmake missing under $prefix/lib/cmake/llvm (LLVM=$llvm_version)"
[[ -f "$prefix/lib/cmake/mlir/MLIRConfig.cmake" ]] \
  || fail "MLIRConfig.cmake missing under $prefix/lib/cmake/mlir (MLIR=$mlir_version)"

if [[ "${llvm_version%.*}" != "$series" ]]; then
  fail "LLVM $llvm_version is outside the accepted ${series}.x series"
fi
# A mixed-patch LLVM/MLIR pair is rejected in CI exactly as on the fleet: the
# passes would compile against a different MLIR than the tools report.
if [[ "$mlir_version" != "$llvm_version" ]]; then
  fail "mixed LLVM/MLIR pair (LLVM=$llvm_version, MLIR=$mlir_version)"
fi

lld_version=""
if (( require_lld )); then
  lld_version="$(version_of "$prefix/bin/ld.lld")"
  [[ -n "$lld_version" ]] || fail "lane requires ld.lld and none runs under $prefix/bin"
  if [[ "$lld_version" != "$llvm_version" ]]; then
    fail "ld.lld $lld_version does not match LLVM $llvm_version"
  fi
fi

if [[ "$llvm_version" == "$fleet_pin" ]]; then
  pin_match="exact"
else
  pin_match="series"
  echo "::notice ::[$lane] running LLVM/MLIR $llvm_version under the CI ${series}.x tolerance; the fleet pin is $fleet_pin, so this result is not a fleet-comparable measurement"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "llvm_version=$llvm_version"
    echo "mlir_version=$mlir_version"
    echo "llvm_prefix=$prefix"
    echo "fleet_pin=$fleet_pin"
    echo "pin_match=$pin_match"
  } >> "$GITHUB_OUTPUT"
fi

summary "### $lane: LLVM/MLIR toolchain" "" \
  "| field | value |" "|---|---|" \
  "| LLVM | \`$llvm_version\` |" \
  "| MLIR | \`$mlir_version\` |" \
  "| ld.lld | \`${lld_version:-not required}\` |" \
  "| prefix | \`$prefix\` |" \
  "| fleet pin | \`$fleet_pin\` |" \
  "| match | $pin_match (CI accepts \`${series}.x\`) |" ""

if [[ -n "$manifest" ]]; then
  mkdir -p "$(dirname "$manifest")"
  cat > "$manifest" <<EOF
{
  "schema": "tessera.ci_toolchain.v1",
  "lane": "$lane",
  "llvm_version": "$llvm_version",
  "mlir_version": "$mlir_version",
  "lld_version": "${lld_version}",
  "llvm_prefix": "$prefix",
  "fleet_pin": "$fleet_pin",
  "accepted_series": "${series}.x",
  "pin_match": "$pin_match",
  "commit": "${GITHUB_SHA:-}",
  "run_id": "${GITHUB_RUN_ID:-}",
  "run_attempt": "${GITHUB_RUN_ATTEMPT:-}"
}
EOF
fi

echo "[$lane] LLVM/MLIR $llvm_version at $prefix (fleet pin $fleet_pin, match=$pin_match)"
