"""M7 follow-up — symbolic Cauchy-Riemann verifier tests.

Coverage:

  - Lowering: ``complex.<op>(...)`` reads pass; bare-name reads
    fail with the same source-span errors the M6 framework uses.
  - **Compile-time analyticity**: ``z²``, ``e^z``, ``mobius``,
    chained compositions pass.  ``conjugate``, ``abs``, ``z·|z|``
    fail with explicit violation reports + the offending op name.
  - ``analytic_symbolic`` decorator raises at decoration time
    (no probing required).
  - ``HolomorphicReport.format()`` produces readable output.
  - The whitelist + classification are closed sets — adding a new
    op without updating both is caught.
"""

from __future__ import annotations

import pytest

from tessera.compiler import ast_ir, complex_jit


# ---------------------------------------------------------------------------
# Whitelist contract — closed sets
# ---------------------------------------------------------------------------

def test_holomorphic_and_non_holomorphic_partition_the_whitelist() -> None:
    """Every entry in the AST whitelist must be classified as either
    holomorphic or non-holomorphic — no orphans."""
    classified = complex_jit.HOLOMORPHIC_OPS | complex_jit.NON_HOLOMORPHIC_OPS
    whitelist = set(complex_jit._COMPLEX_ATTR_TO_OP_NAME.values())
    missing = whitelist - classified
    extras = classified - whitelist
    assert not missing, f"unclassified ops in whitelist: {sorted(missing)}"
    assert not extras, f"classified ops not in whitelist: {sorted(extras)}"


def test_holomorphic_and_non_holomorphic_are_disjoint() -> None:
    """An op can't be both holomorphic and not."""
    assert not (complex_jit.HOLOMORPHIC_OPS & complex_jit.NON_HOLOMORPHIC_OPS)


# ---------------------------------------------------------------------------
# Lowering smoke
# ---------------------------------------------------------------------------

def test_lower_z_squared() -> None:
    def f(z):
        return complex.complex_mul(z, z)   # noqa: F821 — read from AST

    ir = complex_jit.lower_complex_function(f)
    assert ir.arg_names == ("z",)
    assert len(ir.ops) == 1
    assert ir.ops[0].op_name == "complex_mul"
    assert ir.ops[0].python_attr == "complex_mul"


def test_lower_chain_uses_ssa_refs() -> None:
    """A two-op chain produces an IR whose second op consumes
    the first's result."""
    def f(z):
        w = complex.complex_exp(z)            # noqa: F821
        return complex.complex_mul(w, z)      # noqa: F821

    ir = complex_jit.lower_complex_function(f)
    assert [op.op_name for op in ir.ops] == ["complex_exp", "complex_mul"]
    assert ir.ops[1].operand_refs[0] == ir.ops[0].result_name


def test_lower_rejects_non_complex_namespace() -> None:
    def f(z):
        return some_other_module.complex_mul(z, z)  # noqa: F821

    with pytest.raises(complex_jit.ComplexJitError, match="tessera.complex"):
        complex_jit.lower_complex_function(f)


def test_lower_rejects_unknown_complex_op() -> None:
    def f(z):
        return complex.does_not_exist(z)   # noqa: F821

    with pytest.raises(complex_jit.ComplexJitError):
        complex_jit.lower_complex_function(f)


def test_lower_text_starts_with_complex_ir_prefix() -> None:
    def f(z):
        return complex.complex_mul(z, z)   # noqa: F821

    ir = complex_jit.lower_complex_function(f)
    assert ir.text().startswith("complex_ir(")


# ---------------------------------------------------------------------------
# is_holomorphic — positive cases
# ---------------------------------------------------------------------------

def test_z_squared_is_holomorphic() -> None:
    def f(z):
        return complex.complex_mul(z, z)   # noqa: F821

    report = complex_jit.is_holomorphic(f)
    assert report.holomorphic
    assert bool(report) is True
    assert report.violations == ()


def test_exp_of_z_is_holomorphic() -> None:
    def f(z):
        return complex.complex_exp(z)   # noqa: F821

    assert complex_jit.is_holomorphic(f).holomorphic


def test_mobius_is_holomorphic() -> None:
    def f(z):
        return complex.mobius(z, 1.0, 2.0, 0.0, 3.0)   # noqa: F821

    assert complex_jit.is_holomorphic(f).holomorphic


def test_z_squared_times_exp_chain_is_holomorphic() -> None:
    """Composition of holomorphic ops is holomorphic."""
    def f(z):
        s = complex.complex_exp(z)            # noqa: F821
        return complex.complex_mul(s, s)      # noqa: F821

    assert complex_jit.is_holomorphic(f).holomorphic


def test_division_of_z_by_constant_is_holomorphic() -> None:
    def f(z, c):
        return complex.complex_div(z, c)   # noqa: F821

    assert complex_jit.is_holomorphic(f).holomorphic


# ---------------------------------------------------------------------------
# is_holomorphic — negative cases
# ---------------------------------------------------------------------------

def test_conjugate_is_not_holomorphic() -> None:
    def f(z):
        return complex.complex_conjugate(z)   # noqa: F821

    report = complex_jit.is_holomorphic(f)
    assert not report.holomorphic
    assert len(report.violations) == 1
    assert report.violations[0].op_name == "complex_conjugate"
    assert report.violations[0].python_attr == "complex_conjugate"


def test_abs_is_not_holomorphic() -> None:
    def f(z):
        return complex.complex_abs(z)   # noqa: F821

    report = complex_jit.is_holomorphic(f)
    assert not report.holomorphic
    assert report.violations[0].op_name == "complex_abs"


def test_one_non_holomorphic_in_a_chain_taints_the_whole_function() -> None:
    """``z · |z|`` — multiplication is holomorphic but ``abs`` isn't;
    the whole composition is non-holomorphic."""
    def f(z):
        a = complex.complex_abs(z)               # noqa: F821
        return complex.complex_mul(z, a)         # noqa: F821

    report = complex_jit.is_holomorphic(f)
    assert not report.holomorphic
    # The verifier names abs as the offender (not mul).
    op_names = [v.op_name for v in report.violations]
    assert "complex_abs" in op_names
    assert "complex_mul" not in op_names


def test_holomorphic_report_format_is_readable() -> None:
    def f(z):
        return complex.complex_conjugate(z)   # noqa: F821

    text = complex_jit.is_holomorphic(f).format()
    assert "non-holomorphic" in text
    assert "complex_conjugate" in text


# ---------------------------------------------------------------------------
# analytic_symbolic decorator
# ---------------------------------------------------------------------------

def test_analytic_symbolic_passes_holomorphic_function() -> None:
    @complex_jit.analytic_symbolic
    def f(z):
        return complex.complex_mul(z, z)   # noqa: F821

    assert getattr(f, "__tessera_analytic_symbolic__", False) is True


def test_analytic_symbolic_rejects_non_holomorphic_at_decoration() -> None:
    """No probing — the decorator inspects the IR and raises
    immediately."""
    with pytest.raises(complex_jit.NotHolomorphicError) as exc:
        @complex_jit.analytic_symbolic
        def f(z):
            return complex.complex_conjugate(z)   # noqa: F821

    assert exc.value.op_name == "complex_conjugate"
    assert exc.value.python_attr == "complex_conjugate"
    assert "non-holomorphic" in str(exc.value)


def test_analytic_symbolic_rejects_z_times_abs_with_op_name_in_message() -> None:
    with pytest.raises(complex_jit.NotHolomorphicError) as exc:
        @complex_jit.analytic_symbolic
        def g(z):
            return complex.complex_mul(z, complex.complex_abs(z))   # noqa: F821

    assert exc.value.op_name == "complex_abs"


# ---------------------------------------------------------------------------
# Structural reuse of M6's ast_ir core
# ---------------------------------------------------------------------------

def test_complex_lowering_uses_shared_ast_ir_module() -> None:
    """Decision lock: the symbolic CR verifier MUST go through the
    same ``ast_ir`` core that ``energy_jit`` + ``clifford_jit`` use.
    Otherwise we have three parallel AST→IR machinaries to keep in
    sync."""
    config = complex_jit._COMPLEX_LOWERING_CONFIG
    assert isinstance(config, ast_ir.LoweringConfig)
    assert config.namespace == "complex"
    assert config.error_class is complex_jit.ComplexJitError


# ---------------------------------------------------------------------------
# Numerical CR verifier and symbolic CR verifier agree on common cases
# ---------------------------------------------------------------------------

def test_symbolic_and_numerical_verifiers_agree_on_holomorphic_function() -> None:
    """``z²`` passes both."""
    from tessera import complex as tc

    def f_numpy(z):
        cz = tc.from_pair(z.real, z.imag)
        out = tc.complex_mul(cz, cz)
        return __import__("builtins").complex(float(out.re), float(out.im))

    def f_ir(z):
        return complex.complex_mul(z, z)   # noqa: F821

    # Numerical: probe at 4 points.
    for z0 in [1 + 0.5j, -1 + 1j, 2 + 0j, 0.3 - 0.4j]:
        passes, _ = tc.check_cauchy_riemann(f_numpy, z0)
        assert passes
    # Symbolic.
    assert complex_jit.is_holomorphic(f_ir).holomorphic


def test_symbolic_and_numerical_verifiers_agree_on_non_holomorphic_function() -> None:
    """``z̄`` fails both."""
    from tessera import complex as tc

    def f_numpy(z):
        cz = tc.from_pair(z.real, z.imag)
        out = tc.complex_conjugate(cz)
        return __import__("builtins").complex(float(out.re), float(out.im))

    def f_ir(z):
        return complex.complex_conjugate(z)   # noqa: F821

    passes, _ = tc.check_cauchy_riemann(f_numpy, 1 + 1j)
    assert not passes
    assert not complex_jit.is_holomorphic(f_ir).holomorphic
