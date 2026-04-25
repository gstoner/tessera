<!-- MERGE-START: PDDL_Instruct_in_Tessera.md -->
# PDDL‑Instruct in Tessera (v1)

This package wires **Logical Chain‑of‑Thought (CoT)** instruction‑tuned planning (PDDL‑Instruct)
into the Tessera Programming Model. It adds a lightweight `tessera_pddl` dialect, a planning
pipeline, and examples for BlocksWorld and Logistics.

## Goals
- Let users author PDDL domain/problem files and run them through a **PDDL‑Instruct** inference
  stage that emits **Logical CoT traces** plus a **candidate plan**.
- Parse / validate the plan against the domain, turn it into **Plan IR**, and (optionally)
  execute it or export to standard planners.
- Provide verifiable reasoning: precondition checks, effect applications, and invariant checks.

## Architecture
```mermaid
flowchart LR
  A[PDDL Domain/Problem] -->|lift & parse| B(tessera_pddl)
  B --> C[LLM CoT Planner (PDDL‑Instruct)]
  C --> D[Plan Trace + Candidate Plan]
  D --> E[Plan IR validate]
  E --> F{Valid?}
  F -- yes --> G[Emit Plan IR / JSON]
  F -- no  --> H[Self‑corrector: critique & repair]
  H --> C
```
**Key components**
- `dialects/tessera_pddl.td` – minimal ODS types/ops for domains, predicates, actions, states.
- `passes/pddl_instruct_pipeline.cpp` – driver hooks:
  - `-tessera-pddl-parse` (ingest .pddl files),
  - `-tessera-pddl-infer-plan` (LLM CoT prompting + repair loop),
  - `-tessera-plan-validate` (symbolic validator),
  - `-tessera-plan-export` (JSON/trace artifacts).
- `tests/FileCheck/*` – lit tests of parsing, CoT trace shape, validation.
- `examples/` – BlocksWorld/Logistics PDDL + prompt templates.

## Logical CoT Trace Schema (JSONL)
Each step is a grounded action with explicit logic:
```json
{
  "step": 3,
  "action": "(pick-up b2)",
  "applicable": true,
  "reason": [
    "pre(on-table b2) holds",
    "pre(clear b2) holds",
    "pre(handempty) holds"
  ],
  "effects": {
    "add": ["holding b2"],
    "del": ["on-table b2", "clear b2", "handempty"]
  },
  "state_hash": "sha256:..."
}
```

## Prompting Contracts
- **Plan‑Then‑Prove**: first draft plan; then verify each step with logical checks.
- **Prove‑As‑You‑Go**: interleave step emission with precondition/effect proofs.
- **Critique‑Repair**: when validation fails, produce a minimal patch to the plan.

See `docs/PDDL_CoT_Prompting_Guide.md` for exact templates.

## Pass Pipeline Alias
```
tessera-opt   -tessera-pddl-parse   -tessera-pddl-infer-plan='mode=prove-as-you-go,max_iters=3'   -tessera-plan-validate   -tessera-plan-export='emit=jsonl,trace=1' input.pddl
```

## Outputs
- `artifacts/plan.json` – grounded plan sequence.
- `artifacts/trace.jsonl` – CoT reasoning steps.
- `artifacts/validate.txt` – validator report.
- `artifacts/plan.mlir` – Plan IR module.

## Mapping to Tessera IR Stack
- **Graph IR**: a `tessera.plan.program` region with action nodes and dependencies.
- **Schedule IR**: optional timeline for plan execution on an environment or simulator.
- **Tile IR / Target IR**: not used unless plans map to simulator kernels or robot skills.

## Roadmap
- Dataset hooks: replay PDDL‑Instruct examples for regression.
- Mixed‑initiative: human edits injected into critique loop.
- Planner bridges: export to FastDownward/SymK; import heuristic costs.
<!-- MERGE-END: PDDL_Instruct_in_Tessera.md -->
