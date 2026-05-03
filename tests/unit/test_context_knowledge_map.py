from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / "docs" / "context"
ONTOLOGY = CONTEXT / "ontology.yaml"
KNOWLEDGE_MAP = CONTEXT / "knowledge_map.yaml"


def _load_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _is_archive_path(path: Path) -> bool:
    return "archive" in path.parts


def test_context_ontology_and_knowledge_map_parse():
    ontology = _load_yaml(ONTOLOGY)
    knowledge_map = _load_yaml(KNOWLEDGE_MAP)

    assert ontology["schema_version"] == 1
    assert knowledge_map["schema_version"] == 1
    assert ontology["entity_types"]
    assert ontology["relation_types"]
    assert knowledge_map["entities"]


def test_knowledge_map_entities_follow_declared_ontology():
    ontology = _load_yaml(ONTOLOGY)
    knowledge_map = _load_yaml(KNOWLEDGE_MAP)

    required_fields = set(ontology["required_entity_fields"])
    allowed_types = set(ontology["entity_types"])
    optional_fields = set(ontology.get("optional_entity_fields", {}))
    allowed_fields = required_fields | optional_fields

    seen_ids: set[str] = set()
    for entity in knowledge_map["entities"]:
        assert required_fields.issubset(entity), entity
        assert set(entity).difference(allowed_fields) == set(), entity
        assert entity["type"] in allowed_types
        assert entity["id"] not in seen_ids
        seen_ids.add(entity["id"])
        assert entity["references"], entity["id"]


def test_generated_outputs_are_marked_non_authoritative():
    generated = CONTEXT / "generated"
    expected_outputs = [
        "knowledge_graph.json",
        "knowledge_index.md",
        "agent_workflows.md",
        "eval_matrix.md",
    ]

    for name in expected_outputs:
        path = generated / name
        assert path.exists(), path

    graph = _load_yaml(generated / "knowledge_graph.json")
    assert graph["derived_artifact"] is True
    assert graph["authoritative"] is False


def test_knowledge_map_references_existing_repo_paths():
    knowledge_map = _load_yaml(KNOWLEDGE_MAP)

    for entity in knowledge_map["entities"]:
        for reference in entity["references"]:
            if reference.get("external"):
                continue

            assert "path" in reference, entity["id"]
            path = Path(reference["path"])
            assert not path.is_absolute(), reference

            full_path = ROOT / path
            assert full_path.exists(), f"{entity['id']} references missing path: {path}"

            if _is_archive_path(path):
                assert reference.get("historical") is True, reference
                assert reference.get("authoritative") is False, reference


def test_knowledge_map_relations_use_declared_types_and_valid_targets():
    ontology = _load_yaml(ONTOLOGY)
    knowledge_map = _load_yaml(KNOWLEDGE_MAP)

    allowed_relations = set(ontology["relation_types"])
    entity_ids = {entity["id"] for entity in knowledge_map["entities"]}

    for relation in knowledge_map["relations"]:
        assert relation["source"] in entity_ids, relation
        assert relation["relation"] in allowed_relations, relation

        target = relation["target"]
        if target in entity_ids:
            continue

        target_path = Path(target)
        assert not target_path.is_absolute(), relation
        assert (ROOT / target_path).exists(), relation

        if _is_archive_path(target_path):
            assert relation.get("historical") is True, relation
            assert relation.get("authoritative") is False, relation


def test_agent_workflows_cover_documentation_and_code_generation():
    knowledge_map = _load_yaml(KNOWLEDGE_MAP)
    workflows = knowledge_map["agent_workflows"]

    for workflow in (
        "write_or_update_docs",
        "create_examples_or_tutorials",
        "plan_code_changes",
        "check_api_or_status_claims",
    ):
        assert workflow in workflows
        assert workflows[workflow]["goal"]
        assert workflows[workflow]["steps"]
