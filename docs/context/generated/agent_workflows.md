# Agent Workflows

Generated from docs/context/ontology.yaml, docs/context/knowledge_map.yaml, and tests/COMPILER_TEST_PLAN.md. This is a derived navigation artifact; canonical specs remain authoritative.

## check_api_or_status_claims

Prevent stale public claims in generated content.

1. Resolve conflicts through docs/README.md.
2. Check docs/CANONICAL_API.md for names and syntax.
3. Check docs/spec/COMPILER_REFERENCE.md for compiler status and phase labels.
4. Check tests/COMPILER_TEST_PLAN.md for validation expectations.

## create_examples_or_tutorials

Create runnable examples that match current APIs and implementation status.

1. Start from examples/getting_started or examples/compiler for style and structure.
2. Check docs/CANONICAL_API.md before introducing public API calls.
3. Prefer CPU-promised paths unless the example is explicitly hardware-marked.
4. Add or update sample tests when the example becomes part of active documentation.

## plan_code_changes

Use the map to locate specs, source files, and tests before editing code.

1. Identify the entity for the target subsystem.
2. Read its normative references before implementation.
3. Inspect implemented_in and tested_by targets before editing.
4. Update docs or examples when public behavior changes.

## write_or_update_docs

Keep documentation aligned with canonical authorities.

1. Read docs/README.md to identify the normative authority for the topic.
2. Use this knowledge map to find related guides, tests, examples, and implementation surfaces.
3. Use canonical API names from docs/CANONICAL_API.md and docs/spec/PYTHON_API_SPEC.md.
4. Avoid archive docs as authority; cite them only as historical background.
5. Run docs lint and any targeted tests for changed claims.
