#!/usr/bin/env python3
"""Generate deterministic agent context outputs from Tessera context YAML."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = ROOT / "docs" / "context"
ONTOLOGY = CONTEXT_DIR / "ontology.yaml"
KNOWLEDGE_MAP = CONTEXT_DIR / "knowledge_map.yaml"
COMPILER_TEST_PLAN = ROOT / "tests" / "COMPILER_TEST_PLAN.md"
DEFAULT_OUTPUT_DIR = CONTEXT_DIR / "generated"

GENERATED_HEADER = (
    "Generated from docs/context/ontology.yaml, docs/context/knowledge_map.yaml, "
    "and tests/COMPILER_TEST_PLAN.md. This is a derived navigation artifact; "
    "canonical specs remain authoritative."
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def is_archive_path(path: Path) -> bool:
    return "archive" in path.parts


def validate_context(
    ontology: dict[str, Any], knowledge_map: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = set(ontology["required_entity_fields"])
    allowed_types = set(ontology["entity_types"])
    allowed_relations = set(ontology["relation_types"])
    entities = knowledge_map["entities"]
    relations = knowledge_map["relations"]
    entity_ids: set[str] = set()

    for entity in entities:
        missing = required.difference(entity)
        if missing:
            raise ValueError(f"{entity.get('id', entity)} missing required fields {sorted(missing)}")
        if entity["type"] not in allowed_types:
            raise ValueError(f"{entity['id']} uses unknown entity type {entity['type']}")
        if entity["id"] in entity_ids:
            raise ValueError(f"duplicate entity id {entity['id']}")
        entity_ids.add(entity["id"])
        if not entity["references"]:
            raise ValueError(f"{entity['id']} must have at least one reference")

        for reference in entity["references"]:
            if reference.get("external"):
                continue
            raw_path = reference.get("path")
            if not raw_path:
                raise ValueError(f"{entity['id']} has a reference without a path")
            path = Path(raw_path)
            if path.is_absolute():
                raise ValueError(f"{entity['id']} reference must be relative: {path}")
            if not (ROOT / path).exists():
                raise ValueError(f"{entity['id']} references missing path: {path}")
            if is_archive_path(path) and (
                reference.get("historical") is not True
                or reference.get("authoritative") is not False
            ):
                raise ValueError(f"{entity['id']} archive reference must be marked historical/non-authoritative")

    for relation in relations:
        source = relation["source"]
        target = relation["target"]
        if source not in entity_ids:
            raise ValueError(f"relation source is not an entity id: {source}")
        if relation["relation"] not in allowed_relations:
            raise ValueError(f"{source} uses unknown relation type {relation['relation']}")
        if target in entity_ids:
            continue
        target_path = Path(target)
        if target_path.is_absolute():
            raise ValueError(f"relation target must be relative: {target}")
        if not (ROOT / target_path).exists():
            raise ValueError(f"relation target path is missing: {target}")
        if is_archive_path(target_path) and (
            relation.get("historical") is not True
            or relation.get("authoritative") is not False
        ):
            raise ValueError(f"archive relation target must be marked historical/non-authoritative: {target}")

    return sorted(entities, key=lambda item: item["id"]), sorted(
        relations, key=lambda item: (item["source"], item["relation"], item["target"])
    )


def markdown_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|")


def primary_references(entity: dict[str, Any]) -> str:
    return ", ".join(reference["path"] for reference in entity["references"][:3])


def entity_sort_key(entity: dict[str, Any]) -> tuple[int, str]:
    return int(entity.get("output_priority", 10_000)), entity["id"]


def relations_for_entity(entity_id: str, relations: list[dict[str, Any]]) -> list[str]:
    rendered: list[str] = []
    for relation in relations:
        if relation["source"] == entity_id:
            rendered.append(f"{relation['relation']} -> {relation['target']}")
        elif relation["target"] == entity_id:
            rendered.append(f"{relation['source']} -> {relation['relation']}")
    return rendered


def render_knowledge_graph(
    ontology: dict[str, Any],
    knowledge_map: dict[str, Any],
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> str:
    payload = {
        "schema_version": knowledge_map["schema_version"],
        "generated_from": [
            "docs/context/ontology.yaml",
            "docs/context/knowledge_map.yaml",
            "tests/COMPILER_TEST_PLAN.md",
        ],
        "derived_artifact": True,
        "authoritative": False,
        "entity_types": ontology["entity_types"],
        "relation_types": ontology["relation_types"],
        "entities": entities,
        "relations": relations,
        "agent_workflows": knowledge_map["agent_workflows"],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_knowledge_index(
    entities: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> str:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        by_type[entity["type"]].append(entity)

    lines = [
        "# Knowledge Index",
        "",
        GENERATED_HEADER,
        "",
        "Use this generated index for navigation, then verify claims against the canonical docs.",
        "",
    ]
    for entity_type in sorted(by_type):
        lines.extend([f"## {entity_type}", ""])
        lines.append("| ID | Name | Authority | Primary References | Relations |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entity in sorted(by_type[entity_type], key=entity_sort_key):
            entity_relations = "; ".join(relations_for_entity(entity["id"], relations)) or "none"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{markdown_escape(entity['id'])}`",
                        markdown_escape(entity["name"]),
                        markdown_escape(entity["authority"]),
                        markdown_escape(primary_references(entity)),
                        markdown_escape(entity_relations),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_agent_workflows(workflows: dict[str, Any]) -> str:
    lines = [
        "# Agent Workflows",
        "",
        GENERATED_HEADER,
        "",
    ]
    for workflow_id in sorted(workflows):
        workflow = workflows[workflow_id]
        lines.extend([f"## {workflow_id}", "", workflow["goal"], ""])
        for index, step in enumerate(workflow["steps"], start=1):
            lines.append(f"{index}. {step}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def extract_markdown_table(lines: list[str], heading: str) -> list[str]:
    for index, line in enumerate(lines):
        if line.strip() != heading:
            continue
        for table_start in range(index + 1, len(lines)):
            if lines[table_start].startswith("| "):
                table: list[str] = []
                for table_line in lines[table_start:]:
                    if not table_line.startswith("| "):
                        break
                    table.append(table_line)
                if len(table) < 3:
                    raise ValueError(f"{heading} table is too short")
                return table
        break
    raise ValueError(f"could not find table under {heading}")


def render_eval_matrix(test_plan: Path) -> str:
    lines = test_plan.read_text(encoding="utf-8").splitlines()
    eval_matrix = extract_markdown_table(lines, "## Project-Level Eval Matrix")
    eval_tiering = extract_markdown_table(lines, "### Eval Tiering")
    return (
        "# Project-Level Eval Matrix\n\n"
        f"{GENERATED_HEADER}\n\n"
        "This is an agent-friendly rendering of the project eval strategy. "
        "`tests/COMPILER_TEST_PLAN.md` remains the source of truth.\n\n"
        "## Project-Level Eval Matrix\n\n"
        + "\n".join(eval_matrix)
        + "\n\n## Eval Tiering\n\n"
        + "\n".join(eval_tiering)
        + "\n"
    )


def generate_outputs(output_dir: Path) -> dict[str, str]:
    outputs = render_outputs()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    return outputs


def render_outputs() -> dict[str, str]:
    ontology = load_yaml(ONTOLOGY)
    knowledge_map = load_yaml(KNOWLEDGE_MAP)
    entities, relations = validate_context(ontology, knowledge_map)
    return {
        "knowledge_graph.json": render_knowledge_graph(
            ontology, knowledge_map, entities, relations
        ),
        "knowledge_index.md": render_knowledge_index(entities, relations),
        "agent_workflows.md": render_agent_workflows(knowledge_map["agent_workflows"]),
        "eval_matrix.md": render_eval_matrix(COMPILER_TEST_PLAN),
    }


def check_outputs() -> int:
    expected = render_outputs()
    mismatched: list[str] = []
    for name, text in expected.items():
        path = DEFAULT_OUTPUT_DIR / name
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            mismatched.append(name)
    if mismatched:
        print("context outputs are stale:")
        for name in mismatched:
            print(f"  {name}")
        return 1
    print("context outputs are up to date")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated context outputs.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate default outputs and fail if committed files are stale.",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check_outputs()

    outputs = generate_outputs(Path(args.output_dir))
    print(f"generated {len(outputs)} context outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
