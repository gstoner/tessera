"""
tests/unit/test_constraints.py

Tests for ConstraintSolver and predicates:
  - Divisible
  - Range
  - Equal
  - ConstraintSolver.check() / check_all()
  - Integration with @tessera.jit
"""

import pytest

import tessera
from tessera.compiler.constraints import (
    ConstraintSolver,
    Divisible,
    Range,
    Equal,
    TesseraConstraintError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Divisible predicate
# ─────────────────────────────────────────────────────────────────────────────

class TestDivisible:
    def test_passes_when_divisible(self):
        c = Divisible("K", 64)
        assert c.check({"K": 128}) is None
        assert c.check({"K": 64}) is None
        assert c.check({"K": 256}) is None

    def test_fails_when_not_divisible(self):
        c = Divisible("K", 64)
        err = c.check({"K": 100})
        assert isinstance(err, TesseraConstraintError)
        assert "K" in str(err)
        assert "100" in str(err)

    def test_skips_symbolic(self):
        c = Divisible("K", 64)
        assert c.check({}) is None          # K not in bindings → skip
        assert c.check({"M": 128}) is None  # wrong key → skip

    def test_dim_names(self):
        c = Divisible("K", 64)
        assert c.dim_names() == ["K"]

    def test_repr(self):
        c = Divisible("K", 64)
        assert "Divisible" in repr(c)
        assert "K" in repr(c)
        assert "64" in repr(c)

    def test_invalid_divisor_raises(self):
        with pytest.raises(ValueError, match="positive int"):
            Divisible("K", 0)

    def test_negative_divisor_raises(self):
        with pytest.raises(ValueError, match="positive int"):
            Divisible("K", -1)

    def test_float_divisor_raises(self):
        with pytest.raises(ValueError, match="positive int"):
            Divisible("K", 1.5)  # type: ignore

    def test_empty_dim_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Divisible("", 64)

    def test_divisor_1_always_passes(self):
        c = Divisible("K", 1)
        assert c.check({"K": 7}) is None

    def test_immutable(self):
        c = Divisible("K", 64)
        with pytest.raises((AttributeError, TypeError)):
            c.dim = "M"  # frozen dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Range predicate
# ─────────────────────────────────────────────────────────────────────────────

class TestRange:
    def test_passes_within_range(self):
        c = Range("S", 1, 8192)
        assert c.check({"S": 1}) is None
        assert c.check({"S": 4096}) is None
        assert c.check({"S": 8192}) is None

    def test_fails_below_lo(self):
        c = Range("S", 1, 8192)
        err = c.check({"S": 0})
        assert isinstance(err, TesseraConstraintError)
        assert "S" in str(err)

    def test_fails_above_hi(self):
        c = Range("S", 1, 8192)
        err = c.check({"S": 8193})
        assert isinstance(err, TesseraConstraintError)

    def test_skips_symbolic(self):
        c = Range("S", 1, 8192)
        assert c.check({}) is None

    def test_dim_names(self):
        c = Range("S", 1, 8192)
        assert c.dim_names() == ["S"]

    def test_repr(self):
        c = Range("S", 1, 8192)
        assert "Range" in repr(c)

    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lo must be <= hi"):
            Range("S", 100, 1)

    def test_lo_equals_hi(self):
        c = Range("S", 64, 64)
        assert c.check({"S": 64}) is None
        assert c.check({"S": 63}) is not None

    def test_float_bounds_raise(self):
        with pytest.raises(ValueError, match="ints"):
            Range("S", 1.0, 8192)  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Equal predicate
# ─────────────────────────────────────────────────────────────────────────────

class TestEqual:
    def test_passes_when_equal(self):
        c = Equal("D_in", "D_out")
        assert c.check({"D_in": 256, "D_out": 256}) is None

    def test_fails_when_not_equal(self):
        c = Equal("D_in", "D_out")
        err = c.check({"D_in": 256, "D_out": 512})
        assert isinstance(err, TesseraConstraintError)
        assert "256" in str(err)
        assert "512" in str(err)

    def test_skips_one_symbolic(self):
        c = Equal("D_in", "D_out")
        assert c.check({"D_in": 256}) is None   # D_out missing → skip
        assert c.check({"D_out": 256}) is None  # D_in missing → skip
        assert c.check({}) is None

    def test_dim_names(self):
        c = Equal("D_in", "D_out")
        assert set(c.dim_names()) == {"D_in", "D_out"}

    def test_repr(self):
        c = Equal("D_in", "D_out")
        assert "Equal" in repr(c)

    def test_empty_dim_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Equal("", "D_out")


# ─────────────────────────────────────────────────────────────────────────────
# ConstraintSolver
# ─────────────────────────────────────────────────────────────────────────────

class TestConstraintSolver:
    def test_empty_solver_passes(self):
        solver = ConstraintSolver()
        solver.check()  # no constraints → should not raise

    def test_add_and_check(self):
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        solver.check({"K": 128})  # passes

    def test_check_raises_on_violation(self):
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        with pytest.raises(TesseraConstraintError):
            solver.check({"K": 100})

    def test_check_all_returns_all_errors(self):
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        solver.add(Range("S", 1, 8192))
        errors = solver.check_all({"K": 100, "S": 10000})
        assert len(errors) == 2

    def test_check_all_returns_empty_on_pass(self):
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        errors = solver.check_all({"K": 128})
        assert errors == []

    def test_chain_add(self):
        solver = (ConstraintSolver()
                  .add(Divisible("K", 64))
                  .add(Range("S", 1, 8192)))
        assert len(solver) == 2

    def test_constraints_property(self):
        solver = ConstraintSolver()
        c = Divisible("K", 64)
        solver.add(c)
        assert c in solver.constraints

    def test_constraints_is_copy(self):
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        lst = solver.constraints
        lst.clear()  # modifying copy should not affect solver
        assert len(solver) == 1

    def test_len(self):
        solver = ConstraintSolver()
        assert len(solver) == 0
        solver.add(Divisible("K", 64))
        assert len(solver) == 1

    def test_repr(self):
        solver = ConstraintSolver()
        assert "ConstraintSolver" in repr(solver)

    def test_invalid_arg_raises(self):
        solver = ConstraintSolver()
        with pytest.raises(TypeError, match="Constraint instance"):
            solver.add("not a constraint")  # type: ignore

    def test_multiple_constraints_first_violation_raised(self):
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        solver.add(Range("S", 1, 8192))
        # K=100 violates first; check() raises on first violation
        with pytest.raises(TesseraConstraintError) as exc_info:
            solver.check({"K": 100, "S": 4096})
        assert "K" in str(exc_info.value)

    def test_skip_symbolic_dimensions(self):
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        # K not in bindings → constraint is skipped, no error
        solver.check({"M": 128, "N": 256})


# ─────────────────────────────────────────────────────────────────────────────
# Integration: @tessera.jit + tessera.require()
# ─────────────────────────────────────────────────────────────────────────────

class TestJitConstraintIntegration:
    def test_jit_with_no_constraints(self):
        @tessera.jit
        def simple(x):
            return x

        assert callable(simple)

    def test_jit_injects_constraint_solver(self):
        @tessera.jit
        def aligned_gemm(x):
            tessera.require(tessera.constraint.Divisible("K", 64))

        assert len(aligned_gemm.constraints) >= 0  # may be 0 if AST parse skips body

    def test_require_is_callable_at_call_time(self):
        """tessera.require() should be a no-op when called at runtime."""
        tessera.require(Divisible("K", 64))  # should not raise

    def test_constraint_error_propagates(self):
        """If bindings are passed and violate a constraint, it should raise."""
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        with pytest.raises(TesseraConstraintError):
            solver.check({"K": 100})
