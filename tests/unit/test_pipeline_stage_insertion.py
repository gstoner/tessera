"""
Phase 4 — test_pipeline_stage_insertion.py

Tests for PipelinePlan: 1F1B schedule generation, bubble fraction, and
interleaved variant.
"""
import pytest
from tessera.compiler.pipeline_planner import PipelinePlan, Phase, ScheduleStep


class TestPipelinePlanBasic:
    def test_single_stage_no_bubble(self):
        plan = PipelinePlan(num_stages=1, num_micro_batches=4)
        assert plan.bubble_fraction == 0.0

    def test_bubble_fraction_standard(self):
        # (p-1)/m = (4-1)/8 = 0.375
        plan = PipelinePlan(num_stages=4, num_micro_batches=8)
        assert abs(plan.bubble_fraction - 3/8) < 1e-9

    def test_bubble_fraction_interleaved(self):
        # (p-1)/(m*v) = (4-1)/(8*2) = 0.1875
        plan = PipelinePlan(num_stages=4, num_micro_batches=8,
                            interleaved=True, num_chunks=2)
        assert abs(plan.bubble_fraction - 3/16) < 1e-9

    def test_warmup_steps(self):
        plan = PipelinePlan(num_stages=4, num_micro_batches=8)
        assert plan.warmup_steps == 3  # p-1

    def test_invalid_num_stages_raises(self):
        with pytest.raises(ValueError):
            PipelinePlan(num_stages=0, num_micro_batches=4)

    def test_invalid_micro_batches_raises(self):
        with pytest.raises(ValueError):
            PipelinePlan(num_stages=4, num_micro_batches=0)

    def test_interleaved_needs_num_chunks_ge_2(self):
        with pytest.raises(ValueError, match="num_chunks >= 2"):
            PipelinePlan(num_stages=4, num_micro_batches=8,
                         interleaved=True, num_chunks=1)

    def test_interleaved_needs_enough_micro_batches(self):
        with pytest.raises(ValueError):
            PipelinePlan(num_stages=4, num_micro_batches=4,
                         interleaved=True, num_chunks=2)  # need m >= 8


class TestScheduleSteps:
    def _steps(self, num_stages=4, num_micro_batches=4):
        return PipelinePlan(num_stages, num_micro_batches).schedule_steps()

    def test_returns_list(self):
        assert isinstance(self._steps(), list)

    def test_each_step_is_schedule_step(self):
        for s in self._steps():
            assert isinstance(s, ScheduleStep)

    def test_all_micro_batches_covered_forward(self):
        steps = self._steps(num_stages=4, num_micro_batches=4)
        fwd = [s for s in steps if s.phase == Phase.FORWARD]
        # Every rank × every micro-batch has exactly one forward step
        for rank in range(4):
            for mb in range(4):
                count = sum(1 for s in fwd if s.rank == rank and s.micro_batch == mb)
                assert count == 1, f"rank {rank} mb {mb}: {count} fwd steps"

    def test_all_micro_batches_covered_backward(self):
        steps = self._steps(num_stages=4, num_micro_batches=4)
        bwd = [s for s in steps if s.phase == Phase.BACKWARD]
        for rank in range(4):
            for mb in range(4):
                count = sum(1 for s in bwd if s.rank == rank and s.micro_batch == mb)
                assert count == 1, f"rank {rank} mb {mb}: {count} bwd steps"

    def test_forward_before_backward_same_mb_same_rank(self):
        steps = self._steps(num_stages=2, num_micro_batches=4)
        for rank in range(2):
            for mb in range(4):
                fwd_clk = next(s.clock for s in steps
                               if s.rank == rank and s.micro_batch == mb
                               and s.phase == Phase.FORWARD)
                bwd_clk = next(s.clock for s in steps
                               if s.rank == rank and s.micro_batch == mb
                               and s.phase == Phase.BACKWARD)
                assert fwd_clk < bwd_clk, \
                    f"rank {rank} mb {mb}: fwd@{fwd_clk} not before bwd@{bwd_clk}"

    def test_steps_sorted_by_clock(self):
        steps = self._steps()
        clocks = [s.clock for s in steps]
        assert clocks == sorted(clocks)

    def test_rank_0_first_fwd_at_clock_0(self):
        steps = self._steps(num_stages=4, num_micro_batches=4)
        first = next(s for s in steps if s.rank == 0 and s.phase == Phase.FORWARD)
        assert first.clock == 0
        assert first.micro_batch == 0


class TestInterleaved:
    def test_interleaved_steps_include_all_micro_batches(self):
        plan = PipelinePlan(num_stages=2, num_micro_batches=4,
                            interleaved=True, num_chunks=2)
        steps = plan.schedule_steps()
        fwd = [s for s in steps if s.phase == Phase.FORWARD]
        for rank in range(2):
            mbs = {s.micro_batch for s in fwd if s.rank == rank}
            assert mbs == {0, 1, 2, 3}

    def test_interleaved_bubble_less_than_standard(self):
        std = PipelinePlan(num_stages=4, num_micro_batches=8)
        itr = PipelinePlan(num_stages=4, num_micro_batches=8,
                           interleaved=True, num_chunks=2)
        assert itr.bubble_fraction < std.bubble_fraction


class TestPipelinePlanMLIR:
    def test_to_mlir_attrs_contains_num_stages(self):
        plan = PipelinePlan(num_stages=4, num_micro_batches=8)
        attr = plan.to_mlir_attrs()
        assert "num_stages = 4" in attr

    def test_to_mlir_attrs_contains_micro_batches(self):
        plan = PipelinePlan(num_stages=4, num_micro_batches=8)
        attr = plan.to_mlir_attrs()
        assert "num_micro_batches = 8" in attr

    def test_to_mlir_attrs_interleaved_false(self):
        plan = PipelinePlan(num_stages=2, num_micro_batches=4)
        attr = plan.to_mlir_attrs()
        assert "interleaved = false" in attr

    def test_to_mlir_attrs_interleaved_true(self):
        plan = PipelinePlan(num_stages=2, num_micro_batches=4,
                            interleaved=True, num_chunks=2)
        attr = plan.to_mlir_attrs()
        assert "interleaved = true" in attr

    def test_repr_includes_bubble(self):
        plan = PipelinePlan(num_stages=4, num_micro_batches=8)
        r = repr(plan)
        assert "bubble" in r
        assert "37.50%" in r


class TestAsciiRender:
    def test_render_ascii_returns_string(self):
        plan = PipelinePlan(num_stages=2, num_micro_batches=2)
        s = plan.render_ascii()
        assert isinstance(s, str)

    def test_render_ascii_has_rank_labels(self):
        plan = PipelinePlan(num_stages=2, num_micro_batches=2)
        s = plan.render_ascii()
        assert "rank 0" in s
        assert "rank 1" in s
