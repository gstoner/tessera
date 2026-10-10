"""Portable reverse identity and storage gates run before CUDA allocation."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from tessera.compiler.native_attention_vjp_runtime import execute
from tessera.runtime import RuntimeArtifact

PACKET = Path("benchmarks/baselines/nvidia_public_attention_vjp_20261006")
# A source-controlled representative is written by the owning-device recorder.
@pytest.fixture
def artifact():
    packet = PACKET / "packet.json"
    row = json.loads(packet.read_text())["rows"][0]
    return RuntimeArtifact.from_json((PACKET / "artifacts" / row["artifact"]).read_text())


def test_program_roundtrip_and_pin(artifact):
    metadata = artifact.metadata
    program = NativeAttentionVJPProgram.from_json(
        metadata["program_json"], expected_digest=metadata["program_digest"]
    )
    assert program.to_json() == metadata["program_json"]
    assert program.program_digest == metadata["program_digest"]
    with pytest.raises(ValueError, match="pinned identity"):
        NativeAttentionVJPProgram.from_json(metadata["program_json"], expected_digest="0" * 64)


def test_bad_storage_precedes_device_allocation(artifact):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA session created for invalid storage")
    with patch("tessera.compiler.prepared_attention_vjp.PreparedAttentionVJP._prepare", forbidden):
        with pytest.raises(ValueError, match="names/arity"):
            execute(artifact.metadata, ())
        args = tuple(np.zeros((1,), dtype=np.float64) for _ in artifact.metadata["arg_names"])
        with pytest.raises(ValueError, match="storage"):
            execute(artifact.metadata, args)


def test_corruption_precedes_device_allocation(artifact):
    metadata = dict(artifact.metadata)
    raw = json.loads(metadata["program_json"])
    raw["program"]["active"] = [2]
    metadata["program_json"] = json.dumps(raw)
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA session created for corrupt product")
    with patch("tessera.compiler.prepared_attention_vjp.PreparedAttentionVJP._prepare", forbidden):
        with pytest.raises(ValueError, match="pinned identity"):
            execute(metadata, ())

def test_repinned_activity_must_match_native_gradient_lineage(artifact):
    from tessera.compiler.native_attention_vjp_artifact import canonical
    raw = json.loads(artifact.metadata["program_json"])
    raw["program"]["active"] = [2]
    pin = hashlib.sha256(canonical(raw["program"]).encode()).hexdigest()
    raw["program_digest"] = pin
    with pytest.raises(ValueError, match="activity"):
        NativeAttentionVJPProgram.from_json(json.dumps(raw), expected_digest=pin)


def test_fork_rejection_precedes_planner_lock(monkeypatch):
    import tessera.compiler.native_attention_vjp_runtime as adapter
    monkeypatch.setattr(adapter.os, "getpid", lambda: adapter._pid + 1)
    with pytest.raises(ValueError, match="fork"):
        adapter.execute_family(
            source=None, target=None, ordered_inputs=(), arg_names=(),
            source_arg_names=(), out_cotangents=(), wrt_names=(),
            declaration=None, source_graph_ir=None,
        )
