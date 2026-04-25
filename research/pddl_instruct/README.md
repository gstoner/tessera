# Tessera PDDL-Instruct v1.1.1 Add-ons

This drop adds:
1) **Validator core** (`tools/validator/validator.py`) — symbolic pre/effect executor + numeric constraint checks.
2) **Reasoning pattern library** (`docs/Reasoning_Pattern_Library.md` + `presets/pddl_cot_presets.json`).
3) **Architecture bridges** (`docs/Architecture_Bridges.md`) for WMMA/WGMMA + cp.async/TMA heuristics.

## Quick demo
```bash
python3 tools/validator/validator.py \
  --trace examples/traces/flash_trace.jsonl \
  --out artifacts/report.json

cat artifacts/report.json
```

MERGE markers are present so you can splice into your larger docs.
Built: 2025-09-19T15:52:22.242155Z
