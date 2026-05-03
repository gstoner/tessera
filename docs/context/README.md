---
status: Informative
classification: Informative
authority: Agent context guide; normative claims defer to docs/README.md
last_updated: 2026-05-03
---

# Agent Context Map

This directory contains machine-readable context for agents working on Tessera
documentation, examples, and code changes. It is a navigation layer, not a new
source of truth.

## How To Use This Context

1. Start with the documentation authority tree in `docs/README.md`.
2. Use `ontology.yaml` for the allowed entity and relation vocabulary.
3. Use `knowledge_map.yaml` to find the canonical docs, implementation files,
   tests, and examples related to a concept.
4. Use generated outputs in `docs/context/generated/` for fast navigation.
5. When two documents conflict, prefer the normative root listed in
   `docs/README.md`.
6. Treat archived material as historical background only unless a canonical
   spec explicitly restores the concept.

## Source Files And Generated Outputs

The source files are:

- `ontology.yaml` — allowed entity and relation vocabulary.
- `knowledge_map.yaml` — curated project entities, references, relations, and
  agent workflows.
- `tests/COMPILER_TEST_PLAN.md` — project-level eval matrix and eval tiering.

Regenerate derived outputs with:

```bash
python3 scripts/generate_context_outputs.py
```

Check that committed generated outputs are current with:

```bash
python3 scripts/generate_context_outputs.py --check
```

Generated outputs live under `docs/context/generated/`:

- `knowledge_graph.json` — agent-readable entity/relation graph.
- `knowledge_index.md` — human-readable index grouped by entity type.
- `agent_workflows.md` — rendered workflows for docs, examples, code planning,
  and API/status checks.
- `eval_matrix.md` — rendered project eval matrix from the compiler test plan.

Generated files are derived navigation artifacts. They are not authoritative;
agents should use them to find relevant context, then verify claims against
canonical specs.

## Agent Workflows

- For documentation updates, follow `agent_workflows.write_or_update_docs` in
  `knowledge_map.yaml` before changing public API names, phase status, or
  compiler pipeline claims.
- For examples and tutorials, follow
  `agent_workflows.create_examples_or_tutorials` and verify that examples use
  canonical APIs.
- For code generation, follow `agent_workflows.plan_code_changes` to discover
  the relevant specs, source files, and tests before editing.
- For API or status claims, follow `agent_workflows.check_api_or_status_claims`
  and cite the normative reference rather than an informative guide.

## Maintenance Rules

- Add only stable concepts that help agents navigate the project.
- Keep `knowledge_map.yaml` curated; do not mirror every file in the tree.
- Every referenced repo path must exist.
- Any archive reference must be marked `historical: true` and
  `authoritative: false`.
- Add validation coverage before relying on new schema fields in agent
  workflows.
- Keep generated outputs deterministic and free of wall-clock timestamps.
