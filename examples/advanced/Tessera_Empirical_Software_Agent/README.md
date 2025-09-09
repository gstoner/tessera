<!-- ===== MERGE_START Tessera Empirical Software Agent ===== -->
# Tessera Empirical Software Agent (v0.1)

This package sketches a **Tessera-native implementation** of the system described in
*“An AI system to help scientists write expert-level empirical software” (arXiv:2509.06503)*.
It maps the paper’s **LLM + Tree Search** workflow into the Tessera Programming Model with a
reproducible runner, scorable task API, and integration points for **Tile IR/Target IR**, the
**autotuner**, and the **profiler**.

**Key pieces**
- `docs/` — concise spec (split across files with merge markers)
- `src/agents/` — Python reference runner (tree search + scoring + sandbox)
- `mlir/passes/` — C++ pass stubs to register an opt-style driver hook (`-tessera-empirical-search`)
- `examples/` — task shells you can flesh out (Kaggle playground, scRNA-seq integration, COVID-19 forecasting, integrals)
- `tests/` — spot checks / harness notes
- `src/pipelines/empirical_search_pipeline.yaml` — example pass pipeline wiring

> Drop `Tessera_Empirical_Software_Agent_v0_1/` under `tessera/tools/agents/empirical/` in your repo.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # (generate your own with tessera deps + sandbox deps)
python -m src.agents.tree_search_runner --task examples/integrals_solver --budget 512 --parallel 8
```

## What’s included vs stubbed
- ✅ Reference **interface** and **skeleton** that mirror the paper’s method
- ✅ Tessera **integration points** (autotune/profiler/IR hooks) clearly marked
- ☐ LLM provider: implement in `src/agents/llm_interface.py` (Gemini/OpenAI/self-hosted)
- ☐ Real sandbox isolation (use your existing runner policy; default subprocess+timeout)
- ☐ Domain datasets and metrics (plug into `examples/*`)

<!-- ===== MERGE_END Tessera Empirical Software Agent ===== -->
