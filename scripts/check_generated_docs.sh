#!/usr/bin/env bash
# Glass-jaw #12 (2026-06-01) — generated-doc drift gate.
#
# Runs every audit-doc `--check` that a manifest / registry edit can
# drift. Any drift exits non-zero with the regen command. Wired into
# both `.pre-commit-config.yaml` (pre-commit) and the CI audit lane so
# a manifest change without a matching `--render` can't land.
#
# Single source of truth for "which generated docs must stay fresh":
# add a new doc's `--check` invocation here and both pre-commit + CI
# pick it up.
#
# NOT covered here (gated elsewhere, intentionally):
#   * gpu_target_map      — per-`--target` (no single global doc); its
#                           drift is gated by tests/unit/test_gpu_target_maps.py
#   * runtime_abi         — no CLI `--check`; gated by
#                           tests/unit/test_runtime_abi_audit.py
#   * standalone primitive coverage / support-table dashboards beyond
#     support_table — gated by their own unit drift tests.
set -uo pipefail

# Resolve repo root from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON:-python3}"

# (label, module + args) pairs — each must support `--check`.
CHECKS=(
  "op_target_conformance|-m tessera.cli.conformance_matrix --check"
  "e2e_op_coverage|-m tessera.cli.e2e_coverage --check"
  "apple_target_map|-m tessera.cli.apple_target_map --check"
  "support_table|-m tessera.compiler.audit support_table --check"
  "s_series_status|-m tessera.compiler.s_series_status --check"
)

rc=0
for entry in "${CHECKS[@]}"; do
  label="${entry%%|*}"
  cmd="${entry#*|}"
  if ! ${PY} ${cmd} >/dev/null 2>&1; then
    echo "DRIFT: ${label} is stale — regenerate it." >&2
    # Re-run visibly so the operator sees the precise regen hint.
    ${PY} ${cmd} >&2 || true
    rc=1
  else
    echo "ok: ${label}"
  fi
done

if [ "${rc}" -ne 0 ]; then
  echo "" >&2
  echo "Generated-doc drift detected. Regenerate the stale doc(s) with" >&2
  echo "the matching '--render' (or 'python -m tessera.compiler.audit" >&2
  echo "support_table') and commit both the source + regenerated doc." >&2
fi
exit "${rc}"
