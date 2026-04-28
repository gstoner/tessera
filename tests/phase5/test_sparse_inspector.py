"""
Phase 5 — test_sparse_inspector.py

Tests for SolverConfig.analyze_op() — the Python-layer mirror of
SparseInspectorPass + SparsePrecondPass + SparseSolverSpecialize.
"""
import pytest
from tessera.compiler.solver_config import (
    SolverConfig, SparseAnalysisResult,
    PrecondType, SolverVariant,
)


class TestSolverConfigValidation:
    def test_default_threshold(self):
        cfg = SolverConfig()
        assert cfg.sparse_threshold == 0.05

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError):
            SolverConfig(sparse_threshold=0.0)

    def test_invalid_threshold_one(self):
        with pytest.raises(ValueError):
            SolverConfig(sparse_threshold=1.0)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError):
            SolverConfig(max_iter=0)

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError):
            SolverConfig(tolerance=-1e-8)

    def test_invalid_num_ranks(self):
        with pytest.raises(ValueError):
            SolverConfig(num_ranks=0)

    def test_repr_contains_precond(self):
        cfg = SolverConfig()
        assert "precond" in repr(cfg)

    def test_to_mlir_attrs_is_string(self):
        cfg = SolverConfig()
        attr = cfg.to_mlir_attrs()
        assert isinstance(attr, str)
        assert "tessera.solver_config" in attr

    def test_to_mlir_attrs_contains_threshold(self):
        cfg = SolverConfig(sparse_threshold=0.03)
        attr = cfg.to_mlir_attrs()
        assert "0.03" in attr

    def test_to_mlir_attrs_contains_solver(self):
        cfg = SolverConfig()
        attr = cfg.to_mlir_attrs()
        assert "gmres" in attr


class TestSparseAnalysis:
    """Test SolverConfig.analyze_op() sparsity heuristics."""

    def _cfg(self, threshold=0.05):
        return SolverConfig(sparse_threshold=threshold)

    def test_returns_sparse_analysis_result(self):
        res = self._cfg().analyze_op("matA", 0.02)
        assert isinstance(res, SparseAnalysisResult)

    def test_very_sparse_below_1pct_uses_amg(self):
        res = self._cfg().analyze_op("A", 0.005)
        assert res.is_sparse is True
        assert res.precond == PrecondType.AMG

    def test_very_sparse_uses_cg_solver(self):
        res = self._cfg().analyze_op("A", 0.005)
        assert res.solver_variant == SolverVariant.CG

    def test_moderate_sparse_below_3pct_uses_ilu(self):
        res = self._cfg().analyze_op("B", 0.02)
        assert res.precond == PrecondType.ILU
        assert res.solver_variant == SolverVariant.GMRES

    def test_light_sparse_between_3pct_threshold_uses_jacobi(self):
        res = self._cfg().analyze_op("C", 0.04)
        assert res.precond == PrecondType.JACOBI
        assert res.solver_variant == SolverVariant.GMRES

    def test_dense_above_threshold_no_precond(self):
        res = self._cfg().analyze_op("D", 0.10)
        assert res.is_sparse is False
        assert res.precond == PrecondType.NONE

    def test_exactly_at_threshold_is_not_sparse(self):
        res = self._cfg(threshold=0.05).analyze_op("E", 0.05)
        assert res.is_sparse is False

    def test_just_below_threshold_is_sparse(self):
        res = self._cfg(threshold=0.05).analyze_op("F", 0.049)
        assert res.is_sparse is True

    def test_fill_fraction_recorded(self):
        res = self._cfg().analyze_op("G", 0.025)
        assert abs(res.fill_fraction - 0.025) < 1e-9

    def test_op_name_recorded(self):
        res = self._cfg().analyze_op("myOp", 0.001)
        assert res.op_name == "myOp"

    def test_invalid_fill_negative_raises(self):
        with pytest.raises(ValueError):
            self._cfg().analyze_op("X", -0.01)

    def test_invalid_fill_above_one_raises(self):
        with pytest.raises(ValueError):
            self._cfg().analyze_op("X", 1.1)

    def test_to_ir_attr_contains_op_name(self):
        res = self._cfg().analyze_op("fc1", 0.005)
        attr = res.to_ir_attr()
        assert "fc1" in attr

    def test_to_ir_attr_sparse_hint_present(self):
        res = self._cfg().analyze_op("mat", 0.005)
        assert "tessera_solver.sparse_hint" in res.to_ir_attr()

    def test_to_ir_attr_no_sparse_hint_when_dense(self):
        res = self._cfg().analyze_op("mat", 0.5)
        assert "tessera_solver.sparse_hint" not in res.to_ir_attr()

    def test_custom_threshold_changes_classification(self):
        low_cfg  = SolverConfig(sparse_threshold=0.02)
        high_cfg = SolverConfig(sparse_threshold=0.10)
        fill = 0.05
        assert low_cfg.analyze_op("X", fill).is_sparse is False
        assert high_cfg.analyze_op("X", fill).is_sparse is True
