#!/usr/bin/env bash
# Tessera mypy ratchet — track the legacy type-error baseline and
# fail only when new errors land.
#
# Two parallel ratchets share this script:
#
#   * Standard policy (default) — reads scripts/mypy_baseline.txt;
#     runs mypy with whatever pyproject.toml declares.  Baseline=0
#     as of the 2026-05-19 cleanup sprint.
#
#   * Strict policy (MYPY_STRICT=1) — reads
#     scripts/mypy_strict_baseline.txt; runs mypy with
#     --check-untyped-defs to size the next ratchet frontier.
#     Baseline=23 as of 2026-05-19.
#
# Usage:
#   scripts/mypy_ratchet.sh                                  # standard
#   scripts/mypy_ratchet.sh --update-baseline               # write standard baseline
#   MYPY_STRICT=1 scripts/mypy_ratchet.sh                   # strict
#   MYPY_STRICT=1 scripts/mypy_ratchet.sh --update-baseline # write strict baseline
#
# Exit codes:
#   0  — error count is at or below the baseline (clean)
#   1  — error count has *increased* (regression)
#   2  — internal / config error
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MYPY_STRICT="${MYPY_STRICT:-0}"
if [[ "$MYPY_STRICT" != "0" ]]; then
  BASELINE_FILE="$REPO_ROOT/scripts/mypy_strict_baseline.txt"
  MYPY_EXTRA_ARGS=(--check-untyped-defs)
  RATCHET_LABEL="mypy ratchet (strict)"
else
  BASELINE_FILE="$REPO_ROOT/scripts/mypy_baseline.txt"
  MYPY_EXTRA_ARGS=()
  RATCHET_LABEL="mypy ratchet"
fi

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "error: missing baseline file: $BASELINE_FILE" >&2
  exit 2
fi
BASELINE=$(grep -E '^BASELINE=' "$BASELINE_FILE" | cut -d= -f2)
if [[ -z "$BASELINE" ]]; then
  echo "error: BASELINE not set in $BASELINE_FILE" >&2
  exit 2
fi

cd "$REPO_ROOT"

# Resolve the mypy invocation.  Caller may pass:
#   * a path to a mypy binary in $MYPY (e.g., venv/bin/mypy), or
#   * a multi-word invocation like "python -m mypy" in $MYPY.
# If $MYPY is unset, probe known locations.
MYPY="${MYPY:-}"
if [[ -z "$MYPY" ]]; then
  for cand in \
    "/Users/gregorystoner/venv/bin/mypy" \
    "venv/bin/mypy" \
    "$(command -v mypy || true)"; do
    if [[ -x "$cand" ]]; then MYPY="$cand"; break; fi
  done
fi
if [[ -z "$MYPY" ]]; then
  echo "warning: mypy not found; skipping ratchet (install via 'pip install -e .[dev]')" >&2
  exit 0
fi
# Validate: either a single executable path, or a multi-word command
# whose first token is executable / resolvable.
read -ra MYPY_CMD <<< "$MYPY"
if ! command -v "${MYPY_CMD[0]}" >/dev/null 2>&1 && [[ ! -x "${MYPY_CMD[0]}" ]]; then
  echo "warning: \$MYPY not executable: $MYPY  (skipping ratchet)" >&2
  exit 0
fi

REPORT_FILE=$(mktemp -t tessera-mypy-XXXXXX.txt)
trap 'rm -f "$REPORT_FILE"' EXIT

set +e
# Guard MYPY_EXTRA_ARGS against `set -u` when the array is empty
# (Bash 4.4+ syntax for "expand only if defined and non-empty").
"${MYPY_CMD[@]}" python/tessera/ --config-file pyproject.toml ${MYPY_EXTRA_ARGS[@]+"${MYPY_EXTRA_ARGS[@]}"} > "$REPORT_FILE" 2>&1
MYPY_RC=$?
set -e

# Parse the trailing summary line: "Found N errors in M files ..."
COUNT=$(grep -oE 'Found [0-9]+ error' "$REPORT_FILE" | head -1 | grep -oE '[0-9]+' || true)
COUNT=${COUNT:-0}

if [[ "${1:-}" == "--update-baseline" ]]; then
  echo "current ${RATCHET_LABEL} error count: $COUNT (was $BASELINE)"
  sed -i.bak "s/^BASELINE=.*/BASELINE=$COUNT/" "$BASELINE_FILE"
  rm -f "${BASELINE_FILE}.bak"
  echo "updated $BASELINE_FILE"
  exit 0
fi

echo "[${RATCHET_LABEL}] errors=$COUNT  baseline=$BASELINE  (mypy rc=$MYPY_RC)"
if (( COUNT > BASELINE )); then
  echo "" >&2
  echo "FAIL: ${RATCHET_LABEL} error count increased ($BASELINE -> $COUNT)." >&2
  echo "      New errors (or any error above the baseline) must be" >&2
  echo "      fixed or the baseline updated explicitly via:" >&2
  if [[ "$MYPY_STRICT" != "0" ]]; then
    echo "        MYPY_STRICT=1 scripts/mypy_ratchet.sh --update-baseline" >&2
  else
    echo "        scripts/mypy_ratchet.sh --update-baseline" >&2
  fi
  echo "" >&2
  echo "      Full mypy output:" >&2
  cat "$REPORT_FILE" >&2
  exit 1
fi
if (( COUNT < BASELINE )); then
  echo "[${RATCHET_LABEL}] note: error count decreased ($BASELINE -> $COUNT)."
  echo "                    Consider running with --update-baseline."
fi
exit 0
