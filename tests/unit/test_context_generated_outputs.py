from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "generate_context_outputs.py"
CONTEXT = ROOT / "docs" / "context"
KNOWLEDGE_MAP = CONTEXT / "knowledge_map.yaml"
GENERATED = CONTEXT / "generated"

EXPECTED_OUTPUTS = {
    "knowledge_graph.json",
    "knowledge_index.md",
    "agent_workflows.md",
    "eval_matrix.md",
}


def _run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_context_generator_writes_expected_outputs_to_temp_dir(tmp_path):
    result = _run_generator("--output-dir", str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_OUTPUTS


def test_committed_context_outputs_match_fresh_generation(tmp_path):
    result = _run_generator("--output-dir", str(tmp_path))

    assert result.returncode == 0, result.stderr
    for name in EXPECTED_OUTPUTS:
        assert (GENERATED / name).read_text(encoding="utf-8") == (
            tmp_path / name
        ).read_text(encoding="utf-8")


def test_knowledge_graph_json_includes_all_source_entities():
    source = yaml.safe_load(KNOWLEDGE_MAP.read_text(encoding="utf-8"))
    graph = json.loads((GENERATED / "knowledge_graph.json").read_text(encoding="utf-8"))

    source_ids = {entity["id"] for entity in source["entities"]}
    graph_ids = {entity["id"] for entity in graph["entities"]}

    assert graph["derived_artifact"] is True
    assert graph["authoritative"] is False
    assert source_ids.issubset(graph_ids)


def test_generated_markdown_outputs_have_key_headings():
    expected_headings = {
        "knowledge_index.md": "# Knowledge Index",
        "agent_workflows.md": "# Agent Workflows",
        "eval_matrix.md": "# Project-Level Eval Matrix",
    }

    for filename, heading in expected_headings.items():
        text = (GENERATED / filename).read_text(encoding="utf-8")
        assert heading in text


def test_context_generator_check_mode_accepts_committed_outputs():
    result = _run_generator("--check")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "up to date" in result.stdout
