#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

patterns=(
  '@ts\.function'
  '@ts\.compile'
  '@ts\.checkpoint'
  '@ts\.autodiff'
  '@tessera\.function'
  '@tessera\.compile'
  '@tessera\.checkpoint'
  'tessera\.nn'
  'ts\.nn'
)

scan_paths=(
  docs/architecture
  docs/operations
  docs/programming_guide
  docs/reference
  docs/spec
  docs/tutorials
  docs/README.md
  docs/Tessera_Kernel_Compilation_Stages_Overview.md
)

pattern="$(IFS='|'; echo "${patterns[*]}")"

if rg -n --pcre2 "$pattern" "${scan_paths[@]}" \
  --glob '!docs/archive/**' \
  --glob '!docs/old_concepts/**'; then
  echo
  echo "docs lint failed: old API spellings found in active documentation." >&2
  echo "Move old-API material to docs/archive/pre_canonical or rewrite it to canonical APIs." >&2
  exit 1
fi

echo "docs lint passed"
