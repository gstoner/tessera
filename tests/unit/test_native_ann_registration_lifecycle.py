"""Registration ownership is testable without a native compiler or device."""
from fractions import Fraction
import gc
import weakref

import numpy as np
import pytest
from tessera.compiler import native_ann as ann
from tessera.compiler.emit import candidate as registry


@pytest.fixture
def registration(monkeypatch):
    monkeypatch.setattr(registry, '_CANDIDATES', {})
    monkeypatch.setattr(registry, '_OP_KIND_VERIFY', {})
    monkeypatch.setattr(registry, '_REGISTRATION_HOOKS', [])
    # This suite isolates lifetime behavior. Native replay and execution are
    # covered by test_native_ann_admission on the owning native JIT host.
    monkeypatch.setattr(ann, '_affine', lambda text: ('ann', (3, 2), [], None))
    monkeypatch.setattr(ann, 'affine_error_bound', lambda *args: Fraction(1, 10000))
    pair = ann.NativeANNPair('original', 'rewrite', 'compiler')
    def create(budget=0.001):
        return ann.register_native_ann(pair, [np.ones((3, 2), np.float32)],
                                       input_bound=1.0, absolute_budget=budget)
    return create


def test_close_releases_probes_and_disables_retained_candidates(registration):
    owner = registration()
    region_ref = weakref.ref(owner.region)
    candidates = owner.candidates
    assert len(registry.candidates_for('x86', ann.ANN_AFFINE)) == 2
    owner.close()
    owner.close()
    gc.collect()
    assert region_ref() is None
    assert not registry.candidates_for('x86', ann.ANN_AFFINE)
    assert all(not candidate.available() for candidate in candidates)
    assert all(candidate.region is None for candidate in candidates)
    with pytest.raises(ValueError, match='closed'):
        _ = owner.region
    with pytest.raises(ValueError, match='closed'):
        owner.__enter__()


def test_region_churn_and_exception_exit_leave_no_candidates(registration):
    for i in range(32):
        with pytest.raises(RuntimeError, match='caller failed'):
            with registration(i / 1000) as owner:
                assert len(registry.candidates_for('x86', ann.ANN_AFFINE)) == 2
                assert owner.region.absolute_budget == i / 1000
                raise RuntimeError('caller failed')
        assert not registry.candidates_for('x86', ann.ANN_AFFINE)


def test_closing_replaced_owner_preserves_same_name_registration(registration):
    first = registration()
    second = registration()
    first.close()
    assert registry.candidates_for('x86', ann.ANN_AFFINE) == list(second.candidates)
    assert all(candidate.region is second.region for candidate in second.candidates)
    second.close()
    assert not registry.candidates_for('x86', ann.ANN_AFFINE)


def test_partial_registration_failure_retires_installed_instances(registration, monkeypatch):
    install = registry.register_candidate
    def fail_after_install(candidate):
        install(candidate)
        if candidate.transformed:
            raise RuntimeError('registration failed')
    monkeypatch.setattr(registry, 'register_candidate', fail_after_install)
    with pytest.raises(RuntimeError, match='registration failed'):
        registration()
    assert not registry.candidates_for('x86', ann.ANN_AFFINE)


@pytest.mark.parametrize('name', [
    'test_terminal_relu_native_analytic_admission',
    'test_gpu_ann_source_replay_rejects_modified_lowered_program',
])
def test_native_only_tests_skip_before_preparation_without_compiler(monkeypatch, name):
    import test_native_next_admission as native_tests
    monkeypatch.setattr(native_tests, 'find_tessera_opt', lambda: None)
    # Either honest "cannot evaluate here" reason is acceptable, and both come
    # before any preparation: a host without the native compiler, or one whose
    # toolchain cannot package the gfx1151 HSACO these tests build (the guard
    # added when they failed inside ROCDL serialization on a CUDA host).
    with pytest.raises(pytest.skip.Exception,
                       match='native compiler required|no rocm device toolchain to package for'):
        getattr(native_tests, name)()
