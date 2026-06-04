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
PY="${PYTHON:-python3}"

case "${1:-}" in
  "")
    exec "${PY}" -m tessera.compiler.generated_docs --check
    ;;
  --write)
    exec "${PY}" -m tessera.compiler.generated_docs --write
    ;;
  *)
    echo "usage: $0 [--write]" >&2
    exit 2
    ;;
esac
