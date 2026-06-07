#!/usr/bin/env bash
# Activate Tessera's committed git hooks (in .githooks/) for this clone.
#
# One-time, idempotent. Points git at the version-controlled .githooks/ dir so
# every clone runs the same gates without depending on the `pre-commit` tool.
#
#   bash scripts/install-git-hooks.sh
#
# Today this wires the pre-push generated-doc drift gate (.githooks/pre-push).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

chmod +x .githooks/* 2>/dev/null || true
git config core.hooksPath .githooks

echo "Installed: core.hooksPath -> .githooks"
echo "Active hooks:"
for h in .githooks/*; do
  [ -f "$h" ] && echo "  - $(basename "$h")"
done
echo ""
echo "The pre-push hook runs (before each push):"
echo "  1. scripts/check_spec_sync.py      — op-catalog<->PYTHON_API_SPEC + generated-md registry (<1s)"
echo "  2. scripts/check_generated_docs.sh — generated audit-dashboard drift (~30s)"
echo "Bypass once with: TESSERA_SKIP_DOC_DRIFT=1 git push"
