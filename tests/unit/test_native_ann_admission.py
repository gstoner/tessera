"""Native frozen-affine admission and compositional absolute error budgets."""
from dataclasses import replace
from fractions import Fraction
import numpy as np
import platform
import pytest
from tessera.compiler.native_ann import prepare_native_ann, affine_error_bound
from tessera.compiler.evaluator import evaluate_native_ann
from tessera.compiler.scheduled_matmul import find_tessera_opt
from test_native_ann_composition import source

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason='native compiler required')


def pair():
    return prepare_native_ann(source(), allow_reassociation=True)


def test_affine_budget_composes_domain_and_frozen_weight_norms():
    if platform.machine().lower() not in ('x86_64', 'amd64'):
        pytest.skip('ANN admission owns only the x86 native JIT path')
    candidate = pair()
    small = affine_error_bound(candidate, 1.0)
    large = affine_error_bound(candidate, 2.0)
    assert isinstance(small, Fraction)
    assert 0 < small < large < Fraction(1, 1000)
    verdict = evaluate_native_ann(candidate, [], input_bound=1.0, absolute_budget=0.0)
    assert not verdict.admitted
    assert verdict.jit_digest == ''
    assert verdict.observed_error is None
    assert not verdict.promotion_eligible


@pytest.mark.parametrize('value', [True, -1, float('nan'), float('inf')])
def test_invalid_analytic_domain_refuses(value):
    with pytest.raises(ValueError, match='domain'):
        affine_error_bound(pair(), value)


def test_changed_transformed_ir_cannot_reuse_admission():
    candidate = pair()
    with pytest.raises(ValueError, match='replay'):
        affine_error_bound(replace(candidate, transformed=candidate.transformed+'\n'), 1.0)


@pytest.mark.parametrize('option', ['activation', 'shared', 'policy', 'dynamic'])
def test_non_affine_or_unfrozen_envelope_is_not_admitted(option):
    with pytest.raises(ValueError):
        prepare_native_ann(source(**{option: True}), allow_reassociation=True)


def test_missing_reassociation_permission_refuses():
    with pytest.raises(ValueError, match='permission'):
        prepare_native_ann(source())


def test_overflow_domain_fails_closed():
    with pytest.raises(ValueError, match='overflow'):
        affine_error_bound(pair(), float(np.finfo(np.float32).max))


def test_native_pair_executes_under_budget():
    from tessera import _jit_boundary as jit
    import platform
    if platform.machine().lower() not in ('x86_64', 'amd64') or jit._find_dylib() is None:
        pytest.skip('owning x86 native JIT required')
    samples = [np.zeros((3, 2), np.float32), np.ones((3, 2), np.float32),
               np.random.default_rng(71).uniform(-1, 1, (3, 2)).astype(np.float32)]
    candidate = pair()
    result = evaluate_native_ann(candidate, samples, input_bound=1.0, absolute_budget=0.001)
    assert result.admitted and len(result.jit_digest) == 64
    assert result.pair_digest == candidate.digest
    assert Fraction(result.observed_error) <= result.analytic_bound
    assert not result.promotion_eligible
    with pytest.raises(ValueError, match='domain'):
        evaluate_native_ann(candidate, [samples[1]*2], input_bound=1.0, absolute_budget=0.001)


def test_fractional_parameter_rounding_is_included_in_budget():
    text = source().replace('dense<2.0>', 'dense<0.1>').replace('dense<3.0>', 'dense<0.3>')
    candidate = prepare_native_ann(text, allow_reassociation=True)
    assert affine_error_bound(candidate, 1.0) > 0
    from tessera import _jit_boundary as jit
    import platform
    if platform.machine().lower() not in ('x86_64', 'amd64') or jit._find_dylib() is None:
        pytest.skip('owning x86 native JIT required')
    result = evaluate_native_ann(candidate, [np.full((3, 2), .7, np.float32)],
                                 input_bound=1.0, absolute_budget=0.001)
    assert result.admitted


def test_existing_arbiter_filters_over_budget_ann_and_retains_incumbent():
    from tessera import _jit_boundary as jit
    from tessera.compiler.emit import candidate as arbiter
    from tessera.compiler.native_ann import register_native_ann, ANN_AFFINE
    import platform
    if platform.machine().lower() not in ('x86_64', 'amd64') or jit._find_dylib() is None:
        pytest.skip('owning x86 native JIT required')
    registration = None
    saved = {key: list(values) for key, values in arbiter._CANDIDATES.items()}
    verifiers = dict(arbiter._OP_KIND_VERIFY)
    try:
        value = np.ones((3, 2), np.float32)
        registration = register_native_ann(pair(), [value], input_bound=1.0, absolute_budget=0.0)
        region = registration.region
        field = arbiter.live_candidates(region, ANN_AFFINE, 'x86', (value,))
        assert len(field) == 1
        assert not next(iter(field.values())).transformed
        registration.close()
        registration = register_native_ann(pair(), [value], input_bound=1.0, absolute_budget=0.001)
        region = registration.region
        winner = arbiter.arbitrate(region, ANN_AFFINE, 'x86', inputs=(value,))
        assert not winner.transformed
        # An explicitly measured eligible rewrite uses the existing selection
        # seam. Synthetic timings test selection, never count as device evidence.
        winner = arbiter.arbitrate(region, ANN_AFFINE, 'x86', inputs=(value,),
                                   measure=lambda c: 1 if c.transformed else 2)
        assert winner.transformed
        output, tag = winner.run(region, value)
        np.testing.assert_array_equal(output, np.full((3, 2), 34, np.float32))
        assert tag == 'native_cpu'
        with pytest.raises(ValueError, match='domain'):
            winner.run(region, value*2)
    finally:
        if registration is not None:
            registration.close()
        arbiter._CANDIDATES.clear()
        arbiter._CANDIDATES.update(saved)
        arbiter._OP_KIND_VERIFY.clear()
        arbiter._OP_KIND_VERIFY.update(verifiers)


def test_agreeing_wrong_native_programs_do_not_pass(monkeypatch):
    from tessera import _jit_boundary as jit
    import platform
    if platform.machine().lower() not in ('x86_64', 'amd64') or jit._find_dylib() is None:
        pytest.skip('owning x86 native JIT required')
    monkeypatch.setattr(jit, 'invoke', lambda handle, symbol, arrays, out: out.fill(0))
    result = evaluate_native_ann(pair(), [np.ones((3, 2), np.float32)],
                                 input_bound=1.0, absolute_budget=0.001)
    assert not result.admitted
    assert 'program bound' in result.reason


def test_hand_constructed_region_cannot_skip_native_probe_gate():
    from tessera.compiler.native_ann import ANNRegion
    with pytest.raises(ValueError, match='nonempty'):
        ANNRegion(pair(), 1.0, 0.001, ())
    with pytest.raises(ValueError, match='domain'):
        ANNRegion(pair(), True, 0.001, (np.ones((3, 2), np.float32).tobytes(),))


def test_directed_rounding_cannot_use_nearest_error_certificate():
    import ctypes as ct
    import platform
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        pytest.skip('x86 SysV floating-point environment required')
    candidate = pair()
    libc = ct.CDLL(None)
    previous = libc.fegetround()
    try:
        assert libc.fesetround(0x800) == 0  # FE_UPWARD on x86 SysV.
        with pytest.raises(ValueError, match='round-to-nearest'):
            affine_error_bound(candidate, 1.0)
    finally:
        assert libc.fesetround(previous) == 0
