from __future__ import annotations

from pathlib import Path

from tessera.compiler.primitive_coverage import (
    CONTRACT_FIELDS,
    all_primitive_coverages,
    coverage_for,
    coverage_summary,
    primitives_for_model_family,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "audit" / "execution_roadmap.md"
DASHBOARD = ROOT / "docs" / "audit" / "standalone_primitive_coverage.md"


def test_standalone_compiler_sprints_are_documented():
    text = ROADMAP.read_text(encoding="utf-8")

    assert "Standalone compiler milestone sprints (S-series)" in text
    for sprint in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"):
        assert f"[{sprint}]" in text


def test_standalone_roadmap_keeps_external_frameworks_as_references_only():
    text = ROADMAP.read_text(encoding="utf-8")

    assert "independent of PyTorch" in text
    assert "JAX, or Flax at runtime" in text
    assert "reference vocabularies" in text
    assert "supported ops" in text


def test_standalone_roadmap_covers_required_compiler_surfaces():
    text = ROADMAP.read_text(encoding="utf-8")

    expected_surfaces = [
        "Native primitive contract registry",
        "Tensor algebra, indexing, and scalar math",
        "Pytrees, module state, and model containers",
        "Explicit RNG and stochastic effects",
        "Control flow and transform composition",
        "Native sharding and distributed semantics",
        "Flax-level model primitive library",
        "Tiny standalone model conformance suite",
    ]
    for surface in expected_surfaces:
        assert surface in text


def test_standalone_roadmap_names_broad_model_families():
    text = ROADMAP.read_text(encoding="utf-8")

    expected_families = [
        "diffusion",
        "xLSTM",
        "Mamba",
        "Hyena",
        "Linformer",
        "cosFormer",
        "Griffin",
        "Megalodon",
        "JEPA",
        "Titans/Atlas",
    ]
    for family in expected_families:
        assert family in text


def test_primitive_coverage_imports_existing_ops_as_partial_entries():
    entries = all_primitive_coverages()

    matmul = entries["matmul"]
    assert matmul.existing_op
    assert matmul.status == "partial"
    assert matmul.graph_name == "tessera.matmul"
    assert matmul.contract_status["lowering_rule"] == "complete"
    assert "vjp" in matmul.missing_contracts()


def test_primitive_coverage_tracks_planned_standalone_gaps_separately():
    scan = coverage_for("scan")
    memory_write = coverage_for("memory_write")

    assert not scan.existing_op
    assert scan.status == "planned"
    assert "Mamba/SSM" in scan.model_families
    assert not memory_write.existing_op
    assert "Titans/Atlas" in memory_write.model_families


def test_primitive_coverage_contract_fields_are_complete_for_every_entry():
    for entry in all_primitive_coverages().values():
        assert set(entry.contract_status) == set(CONTRACT_FIELDS)


def test_primitive_coverage_family_queries_and_summary():
    ssm_entries = primitives_for_model_family("Mamba/SSM")
    names = {entry.name for entry in ssm_entries}
    summary = coverage_summary()

    assert "scan" in names
    assert "selective_ssm" in names
    assert summary["planned"] > 0
    assert summary["partial"] > 0


def test_primitive_coverage_renders_markdown_dashboard():
    text = render_markdown([coverage_for("scan"), coverage_for("matmul")])

    assert "# Standalone Primitive Coverage" in text
    assert "`scan`" in text
    assert "`matmul`" in text
    assert "Missing contracts" in text


def test_standalone_primitive_dashboard_documents_s1_contract():
    text = DASHBOARD.read_text(encoding="utf-8")

    assert "Standalone Primitive Coverage" in text
    assert "PyTorch" in text
    assert "JAX" in text
    assert "Flax" in text
    assert "Contract Axes" in text
    assert "Model-Family Coverage Tags" in text
