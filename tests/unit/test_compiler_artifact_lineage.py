from __future__ import annotations

import json

import pytest

from tessera.compiler.driver import CompileArtifactBundle, CompileRequest
from tessera.compiler.matmul_pipeline import LoweringArtifact


def _artifact(
    level: str,
    text: str,
    *,
    parent: LoweringArtifact | None = None,
) -> LoweringArtifact:
    return LoweringArtifact(
        level,
        text,
        producer=f"test.{level}",
        input_digest=parent.output_digest if parent is not None else None,
        representation="mlir" if level != "target" else "target_ir",
    )


def _bundle(
    graph: LoweringArtifact,
    schedule: LoweringArtifact,
    tile: LoweringArtifact,
    target: LoweringArtifact,
) -> CompileArtifactBundle:
    return CompileArtifactBundle(
        request=CompileRequest(
            source_origin="test",
            function_name="matmul",
            graph_ir=graph.text,
        ),
        graph=graph,
        schedule=schedule,
        tile=tile,
        target_ir=target,
    )


def test_complete_lineage_requires_digest_adjacency() -> None:
    graph = _artifact("graph", "graph")
    schedule = _artifact("schedule", "schedule", parent=graph)
    tile = _artifact("tile", "tile", parent=schedule)
    target = _artifact("target", "target", parent=tile)

    bundle = _bundle(graph, schedule, tile, target)

    assert bundle.lineage_complete
    assert bundle.to_metadata()["lineage_complete"] is True
    assert [row["input_digest"] for row in bundle.artifact_lineage] == [
        None,
        graph.output_digest,
        schedule.output_digest,
        tile.output_digest,
    ]


def test_substituted_tile_artifact_is_detected() -> None:
    graph = _artifact("graph", "graph")
    schedule = _artifact("schedule", "schedule", parent=graph)
    original_tile = _artifact("tile", "tile-v1", parent=schedule)
    target = _artifact("target", "target", parent=original_tile)
    substituted_tile = _artifact("tile", "tile-v2", parent=schedule)

    bundle = _bundle(graph, schedule, substituted_tile, target)

    assert not bundle.lineage_complete
    assert target.input_digest != substituted_tile.output_digest


def test_stage_presence_does_not_imply_boundary_e2e() -> None:
    bundle = _bundle(
        LoweringArtifact("graph", "graph"),
        LoweringArtifact("schedule", "schedule"),
        LoweringArtifact("tile", "tile"),
        LoweringArtifact("target", "target"),
    )

    assert all(stage["status"] == "complete" for stage in bundle.spine_stages()[:4])
    assert not bundle.lineage_complete


def test_missing_canonical_stage_does_not_imply_boundary_e2e() -> None:
    graph = _artifact("graph", "graph")
    target = _artifact("target", "target", parent=graph)
    bundle = CompileArtifactBundle(
        request=CompileRequest(
            source_origin="test",
            function_name="matmul",
            graph_ir=graph.text,
        ),
        graph=graph,
        target_ir=target,
    )

    assert not bundle.lineage_complete


def test_unknown_lineage_contract_is_not_complete() -> None:
    graph = LoweringArtifact(
        "graph",
        "graph",
        producer="test.graph",
        contract_version="tessera.compiler.lineage.v999",
    )
    schedule = _artifact("schedule", "schedule", parent=graph)
    tile = _artifact("tile", "tile", parent=schedule)
    target = _artifact("target", "target", parent=tile)

    assert not _bundle(graph, schedule, tile, target).lineage_complete


def test_lineage_survives_json_round_trip_and_rejects_tampering() -> None:
    graph = _artifact("graph", "graph")
    encoded = json.dumps(graph.to_dict(), sort_keys=True)

    restored = LoweringArtifact.from_dict(json.loads(encoded))

    assert restored == graph
    tampered = json.loads(encoded)
    tampered["text"] = "different graph"
    with pytest.raises(ValueError, match="output_digest"):
        LoweringArtifact.from_dict(tampered)
