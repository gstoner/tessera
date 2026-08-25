"""PR #626 review — the pipeline Schedule Object carrier is validated, and
its interleaved dependencies are keyed by the VIRTUAL STAGE.

Three claims:

* keying by `(rank, micro_batch, phase)` collided under interleaving, where
  one rank owns several virtual stages. Measured before the fix on
  num_stages=4/num_chunks=2: 64 of 128 steps shared a key and 32 of 56
  cross-stage forward steps lost their producer edge, so the emitted
  Schedule Object permitted a virtual stage to run before the stage feeding
  it;
* a producer ordered after its consumer is now a hard error for the forward
  chain instead of a silently dropped edge;
* the C++ legality pass parses the carrier rows — unique ids, resolvable
  dependencies, topological order, agreement with the declared dimensions —
  instead of only checking that the array is non-empty.
"""

from __future__ import annotations

import pytest

from tessera.compiler.pipeline_planner import Phase, PipelinePlan

INTERLEAVED = dict(num_stages=4, num_micro_batches=8, interleaved=True, num_chunks=2)


def _edges(plan):
    return {a.action_id: set(a.depends_on) for a in plan.schedule_object.actions}


@pytest.mark.parametrize("config", [
    INTERLEAVED,
    dict(num_stages=4, num_micro_batches=12, interleaved=True, num_chunks=3),
    dict(num_stages=4, num_micro_batches=8, interleaved=False),
])
def test_every_cross_stage_forward_step_keeps_its_producer_edge(config):
    plan = PipelinePlan(**config)
    steps = tuple(plan.schedule_steps())
    deps = _edges(plan)
    checked = 0
    for step in steps:
        if step.phase != Phase.FORWARD or step.stage == 0:
            continue
        producer = [
            other for other in steps
            if other.phase == Phase.FORWARD
            and other.stage == step.stage - 1
            and other.micro_batch == step.micro_batch
        ]
        if not producer:
            continue
        checked += 1
        assert plan._action_id(producer[0]) in deps[plan._action_id(step)], (
            f"{plan._action_id(step)} lost its cross-stage producer")
    assert checked, "expected cross-stage forward steps to check"


def test_interleaved_ranks_really_own_several_virtual_stages():
    """The precondition that made the old key collide — pinned so the test
    above cannot silently become vacuous."""
    plan = PipelinePlan(**INTERLEAVED)
    steps = tuple(plan.schedule_steps())
    stages_per_rank: dict[int, set[int]] = {}
    for step in steps:
        stages_per_rank.setdefault(step.rank, set()).add(step.stage)
    assert all(len(stages) > 1 for stages in stages_per_rank.values())
    rank_keys = {(s.rank, s.micro_batch, s.phase) for s in steps}
    stage_keys = {(s.stage, s.micro_batch, s.phase) for s in steps}
    assert len(rank_keys) < len(steps)      # the old key collided...
    assert len(stage_keys) == len(steps)    # ...the virtual stage does not


def test_every_dependency_precedes_its_consumer():
    for config in (INTERLEAVED, dict(num_stages=2, num_micro_batches=4)):
        plan = PipelinePlan(**config)
        steps = tuple(plan.schedule_steps())
        order = {plan._action_id(s): i for i, s in enumerate(steps)}
        for action in plan.schedule_object.actions:
            for dependency in action.depends_on:
                assert order[dependency] < order[action.action_id], (
                    f"{action.action_id} depends on later {dependency}")
