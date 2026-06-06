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
echo "The pre-push hook runs scripts/check_generated_docs.sh (~30s) before each push."
echo "Bypass once with: TESSERA_SKIP_DOC_DRIFT=1 git push"
