#!/usr/bin/env bash
# Glass-jaw #12 (2026-06-01) — generated-doc drift gate + regenerator.
#
# Single source of truth for "which generated docs must stay fresh".
# Two modes:
#
#   scripts/check_generated_docs.sh            # CI / pre-commit drift gate
#                                              # (default): fail on any stale doc
#   scripts/check_generated_docs.sh --write    # sprint-finish regenerator:
#                                              # rewrite every generated doc
#
# Wired into `.pre-commit-config.yaml` (pre-commit) and the CI audit
# lane, so a registry/manifest change without a matching regen can't
# land. To add a generated doc, append one DOCS entry below — both
# modes pick it up automatically.
#
# Each DOCS entry is "label|<check args>|<write args>", all relative to
# `python -m`. The check args must support `--check` (exit non-zero on
# drift); the write args must (re)write the doc on disk.
#
# NOT covered here (gated elsewhere, intentionally):
#   * gpu_target_map — per-`--target` (no single global doc); drift
#                      gated by tests/unit/test_gpu_target_maps.py
set -uo pipefail

# Resolve repo root from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON:-python3}"

MODE="check"
if [ "${1:-}" = "--write" ]; then
  MODE="write"
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--write]" >&2
  exit 2
fi

# label | <check args> | <write args>
DOCS=(
  "op_target_conformance|-m tessera.cli.conformance_matrix --check|-m tessera.cli.conformance_matrix --render"
  "e2e_op_coverage|-m tessera.cli.e2e_coverage --check|-m tessera.cli.e2e_coverage --render"
  "apple_target_map|-m tessera.cli.apple_target_map --check|-m tessera.cli.apple_target_map --render"
  "support_table|-m tessera.compiler.audit support_table --check|-m tessera.compiler.audit support_table"
  "s_series_status|-m tessera.compiler.s_series_status --check|-m tessera.compiler.s_series_status"
  "verifier_coverage|-m tessera.compiler.audit verifier_coverage --check|-m tessera.compiler.audit verifier_coverage --write"
  "runtime_abi|-m tessera.compiler.audit runtime_abi --check|-m tessera.compiler.audit runtime_abi --write"
)

rc=0

if [ "${MODE}" = "write" ]; then
  echo "Regenerating all generated docs..." >&2
  for entry in "${DOCS[@]}"; do
    label="${entry%%|*}"
    rest="${entry#*|}"
    write_cmd="${rest#*|}"
    if ${PY} ${write_cmd} >/dev/null 2>&1; then
      echo "wrote: ${label}"
    else
      echo "FAILED to regenerate: ${label}" >&2
      ${PY} ${write_cmd} >&2 || true
      rc=1
    fi
  done
  if [ "${rc}" -eq 0 ]; then
    echo "" >&2
    echo "All generated docs regenerated. Commit the source + regenerated docs." >&2
  fi
  exit "${rc}"
fi

# Default: drift-gate mode.
for entry in "${DOCS[@]}"; do
  label="${entry%%|*}"
  rest="${entry#*|}"
  check_cmd="${rest%%|*}"
  if ! ${PY} ${check_cmd} >/dev/null 2>&1; then
    echo "DRIFT: ${label} is stale — regenerate it." >&2
    # Re-run visibly so the operator sees the precise regen hint.
    ${PY} ${check_cmd} >&2 || true
    rc=1
  else
    echo "ok: ${label}"
  fi
done

if [ "${rc}" -ne 0 ]; then
  echo "" >&2
  echo "Generated-doc drift detected. Regenerate everything in one shot with" >&2
  echo "  scripts/check_generated_docs.sh --write" >&2
  echo "then commit both the source + regenerated docs." >&2
fi
exit "${rc}"
