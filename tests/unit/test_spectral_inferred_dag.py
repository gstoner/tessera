"""SO-3: spectral physical producers consume inferred action edges."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tessera.compiler.scheduled_spectral import (
    infer_spectral_action_dag,
    lower_scheduled_spectral,
)


@pytest.mark.parametrize(
    ("op_name", "dct_type"),
    [
        ("tessera.spectral_filter", 0),
        ("tessera.dct", 2),
        ("tessera.dct", 4),
        ("tessera.spectral_conv", 0),
        ("tessera.stft", 0),
        ("tessera.istft", 0),
    ],
)
def test_spectral_families_infer_reasoned_edges_and_schedule_evidence(
    op_name, dct_type
):
    inferred, schedule = infer_spectral_action_dag(
        semantic_digest="ab" * 32,
        target="rocm",
        architecture="gfx1151",
        op_name=op_name,
        dct_type=dct_type,
        workspace_bytes=4096,
    )
    assert inferred.dependencies
    assert all(edge.reasons for edge in inferred.dependencies)
    assert schedule.edges == inferred.schedule_object.edges
    assert {role.name for role in schedule.roles} == {
        "spectral_compute",
        "spectral_queue",
    }
    assert all(action.resource_vector for action in schedule.actions)
    assert len(schedule.digest) == 64


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires tessera-opt")
def test_spectral_lowering_stamps_the_schedule_object_digest_end_to_end():
    artifact = lower_scheduled_spectral(
        target="x86",
        op_name="tessera.spectral_conv",
        input_shapes=((2, 13), (2, 7)),
    )
    encoded = json.dumps(
        artifact.schedule_object, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == artifact.schedule_digest
    assert (
        f'tessera.schedule_digest = "{artifact.schedule_digest}"'
        in artifact.schedule_ir
    )
    assert f'tessera.schedule_hash = "{artifact.schedule_digest}"' in artifact.tile_ir
    assert artifact.graph_analysis_digest


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires tessera-opt")
def test_spectral_lowering_rejects_a_stale_module_schedule_digest():
    artifact = lower_scheduled_spectral(
        target="x86",
        op_name="tessera.spectral_filter",
        input_shapes=((2, 17), (2, 17)),
    )
    stale = artifact.schedule_ir.replace(
        f'tessera.schedule_digest = "{artifact.schedule_digest}"',
        f'tessera.schedule_digest = "{"0" * 64}"',
        1,
    )
    with pytest.raises(RuntimeError, match="module Schedule Object digest"):
        run_tessera_opt(find_tessera_opt(), stale, "--tessera-schedule-to-tile")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires tessera-opt")
def test_spectral_lowering_rejects_policy_changed_behind_a_matching_digest():
    """PR #626 review, P1 — the fail-open the deleted C++ digest helper left.

    Comparing `artifact_hash` to the module digest does NOT bind the
    attributes the pass then consumes: a cached or hand-edited program can
    keep BOTH digest strings and still change `workspace_bytes` or
    `native_entry`, and the native launch would use the changed value (a
    wrong buffer contract or an incompatible entry point). The lowering now
    re-verifies every consumed policy value against the semantic payload the
    producer carries on the module.
    """
    artifact = lower_scheduled_spectral(
        target="x86",
        op_name="tessera.spectral_filter",
        input_shapes=((2, 17), (2, 17)),
    )
    match = re.search(r"workspace_bytes = (\d+) : i64", artifact.schedule_ir)
    assert match, "expected a workspace_bytes attribute to tamper with"
    tampered = artifact.schedule_ir.replace(
        match.group(0), f"workspace_bytes = {int(match.group(1)) + 4096} : i64", 1
    )
    # Both digests are untouched — only the consumed policy changed.
    assert f'tessera.schedule_digest = "{artifact.schedule_digest}"' in tampered
    with pytest.raises(RuntimeError, match="disagrees with the carried semantic"):
        run_tessera_opt(find_tessera_opt(), tampered, "--tessera-schedule-to-tile")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires tessera-opt")
def test_spectral_lowering_requires_the_semantic_payload():
    """The payload itself is mandatory: without it the consumer cannot
    re-verify what it consumes, so it fails closed rather than trusting the
    digest string."""
    artifact = lower_scheduled_spectral(
        target="x86",
        op_name="tessera.spectral_filter",
        input_shapes=((2, 17), (2, 17)),
    )
    stripped = re.sub(r', tessera\.spectral_semantic = "[^"]*"', "",
                      artifact.schedule_ir, count=1)
    assert "tessera.spectral_semantic" not in stripped
    with pytest.raises(RuntimeError, match="tessera.spectral_semantic"):
        run_tessera_opt(find_tessera_opt(), stripped, "--tessera-schedule-to-tile")
