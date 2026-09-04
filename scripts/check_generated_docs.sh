#!/usr/bin/env bash
# Generated-doc drift gate + regenerator.
#
# Thin wrapper over the single registry in
# `python/tessera/compiler/generated_docs.py`, which is the one source of
# truth for *which* docs are generated, *how* they regenerate, and *which*
# artifact (CSV when present, else Markdown) the drift gate byte-compares.
#
#   scripts/check_generated_docs.sh            # CI / pre-commit drift gate
#   scripts/check_generated_docs.sh --write    # sprint-finish: regenerate all
#
# Wired into `.pre-commit-config.yaml` and the CI audit lane. To add or
# retire a dashboard, edit the registry — never this script.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

# Resolve an interpreter that can import numpy rather than trusting PATH order —
# a python3 without it turns a passing gate into a spurious drift failure.
# Honors $TESSERA_PYTHON / $PYTHON first. See scripts/_python_env.sh.
# shellcheck source=_python_env.sh
. "${SCRIPT_DIR}/_python_env.sh" || exit 1
PY="${TESSERA_PY}"

case "${1:-}" in
  "")
    exec "${PY}" -m tessera.compiler.generated_docs --check
    ;;
  --refresh-coverage)
    # Regenerate from the resolved source tree; never merge generated rows.
    # Refuse if any authored input is still conflicted.
    while IFS= read -r conflict; do
      case "$conflict" in
        docs/audit/generated/test_coverage.md|docs/audit/generated/test_coverage.csv|docs/audit/generated/docs_freshness.md) ;;
        *) echo "Resolve authored conflict before regeneration: $conflict" >&2; exit 1 ;;
      esac
    done < <(git diff --name-only --diff-filter=U)
    "${PY}" -m tessera.compiler.generated_docs --write test_coverage docs_freshness || exit 1
    exec "${PY}" -m tessera.compiler.generated_docs --check test_coverage docs_freshness
    ;;
  --write)
    exec "${PY}" -m tessera.compiler.generated_docs --write
    ;;
  *)
    echo "usage: $0 [--write|--refresh-coverage]" >&2
    exit 2
    ;;
esac
