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
echo "  1. scripts/check_spec_sync.py      — op-catalog<->PYTHON_API_SPEC + generated-md registry + docs-freshness (<1s)"
echo "  2. scripts/check_generated_docs.sh — generated audit-dashboard drift (~30s)"
echo ""
echo "Interpreter: resolved by capability (must import numpy), not PATH order —"
echo "see scripts/_python_env.sh. Override with:  export TESSERA_PYTHON=/path/to/python3"

# Report the resolved interpreter now, so a broken environment surfaces at
# install time rather than as a confusing 'drift' failure on the first push.
if . "${repo_root}/scripts/_python_env.sh" 2>/dev/null; then
  echo "  resolved: ${TESSERA_PY}  (numpy $(${TESSERA_PY} -c 'import numpy;print(numpy.__version__)' 2>/dev/null))"
else
  echo "  WARNING: no python3 with numpy found — the hook will block until this is fixed."
  echo "           Run 'bash scripts/setup_ubuntu.sh' or set TESSERA_PYTHON (see above)."
fi
echo ""
echo "Bypass once with: TESSERA_SKIP_DOC_DRIFT=1 git push"
echo "  (only for a genuine emergency — if a gate fails because no interpreter"
echo "   was found, that is an environment problem and the bypass hides a check"
echo "   you have not actually run.)"
