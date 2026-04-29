from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import tessera as ts
from tessera import checkpoint, elastic, fault


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_failure_policy_decorator_attaches_ir_metadata():
    @fault.on_failure(policy="drain_then_resume", max_retries=2)
    def step(x):
        return x + 1

    cfg = step.__tessera_failure_policy__
    assert step(2) == 3
    assert cfg.policy == "drain_then_resume"
    assert "tessera.fault" in cfg.to_ir_attr()


def test_preempt_policy_decorator_attaches_grace_action():
    @fault.on_preempt(grace_s=15, action="checkpoint_then_exit")
    def main():
        return "ok"

    cfg = main.__tessera_preempt_policy__
    assert main() == "ok"
    assert cfg.grace_s == 15
    assert "checkpoint_then_exit" in cfg.to_ir_attr()


def test_fault_injection_context_tracks_active_faults():
    assert fault.active_faults() == ()
    with fault.inject(drop_device=3) as event:
        assert event.kind == "drop_device"
        assert event.target == 3
        assert fault.active_faults() == (event,)
    assert fault.active_faults() == ()


def test_elastic_config_and_context_validate_rendezvous():
    cfg = elastic.configure(
        backend="k8s",
        group="exp",
        min_ranks=2,
        max_ranks=8,
        rebalance_on_join=True,
    )
    assert cfg.backend == "k8s"
    assert cfg.to_dict()["max_ranks"] == 8
    assert "tessera.elastic" in cfg.to_ir_attr()

    with elastic.elastic(rendezvous="k8s://rdzv", min_ranks=1, max_ranks=4) as ctx:
        assert ctx.rendezvous == "k8s://rdzv"
        assert elastic.current_config() is ctx


def test_dist_namespace_exposes_elastic_helpers():
    elastic.set_current_mesh({"dp": 2, "tp": 4})

    assert ts.dist.world_size() == 8
    assert ts.dist.current_mesh() == {"dp": 2, "tp": 4}
    plan = ts.dist.reshard(
        policy="consistent_hash",
        old_mesh={"dp": 2},
        new_mesh={"dp": 4},
    )
    assert plan.moved_fraction() == pytest.approx(0.5)


def test_reshard_plan_reports_movement_and_ir_attr():
    plan = elastic.reshard(
        policy="consistent_hash",
        migrate_async=True,
        old_mesh={"dp": 8},
        new_mesh={"dp": 16},
    )

    assert plan.moved_fraction() == pytest.approx(0.5)
    assert "consistent_hash" in plan.to_ir_attr()


def test_checkpoint_save_load_atomic_manifest_and_remap(tmp_path):
    tensor = np.zeros((2, 3), dtype=np.float32)
    manifest = checkpoint.save(
        tag="step_1",
        tensors={"w": tensor},
        optimizer={"type": "adamw"},
        mesh={"dp": 2},
        root=tmp_path,
        atomic=True,
        step=1,
        numerics="deterministic",
        rng={"global": 1234},
        reduce_tree_id="ring-dp2",
        autotune_cache={"arch": "sm90", "entries": 7},
    )

    assert manifest.committed is True
    assert checkpoint.last_committed(root=tmp_path) == "step_1"

    state = checkpoint.load("step_1", root=tmp_path, remap_to={"dp": 4})
    assert state.manifest.mesh == {"dp": 4}
    assert state.tensors["w"].shape == (2, 3)
    assert state.optimizer["type"] == "adamw"

    payload = json.loads((tmp_path / "step_1" / "manifest.json").read_text())
    assert payload["committed"] is True
    assert payload["reduce_tree_id"] == "ring-dp2"


def test_checkpoint_rejects_uncommitted_manifest(tmp_path):
    checkpoint.save(tag="draft", tensors={}, root=tmp_path, atomic=False)

    with pytest.raises(checkpoint.CheckpointError, match="not committed"):
        checkpoint.load("draft", root=tmp_path)


def test_async_checkpoint_policy_validation():
    cfg = checkpoint.enable_async(max_bandwidth_gbps=4.0, flush_interval_s=5.0)

    assert cfg.enabled
    assert checkpoint.async_config().max_bandwidth_gbps == 4.0

    with pytest.raises(ValueError, match="must be > 0"):
        checkpoint.enable_async(max_bandwidth_gbps=0.0, flush_interval_s=5.0)


def test_fault_tolerance_guide_is_registered_and_covers_core_topics():
    guide = (ROOT / "docs/guides/Tessera_Fault_Tolerance_And_Elasticity_Guide.md").read_text()
    readme = (ROOT / "docs/README.md").read_text()

    for needle in [
        "Failure Handling",
        "Elastic Membership",
        "Mesh Reconfiguration and Resharding",
        "Runtime Checkpointing",
        "Orchestrator Integration",
        "Compiler Integration Contract",
    ]:
        assert needle in guide
    assert "Tessera_Fault_Tolerance_And_Elasticity_Guide.md" in readme
