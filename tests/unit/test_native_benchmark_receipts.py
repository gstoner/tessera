from types import SimpleNamespace
import numpy as np
import pytest
from benchmarks.native_ann_adapter import launch_receipt, measure
from benchmarks.autodiff.benchmark_public_ssd import finite_vjp, reference


def test_receipt_observes_driver_status_and_restores_binding():
    original = lambda *args: args[0]
    native = SimpleNamespace(_launch=original)
    with launch_receipt(native) as receipt:
        assert native._launch(0) == 0
        assert native._launch(7) == 7
    assert native._launch is original
    assert receipt == dict(attempted=2, accepted=1, completed=True)


def test_receipt_failure_does_not_claim_completion():
    original = lambda: 0
    native = SimpleNamespace(_launch=original)
    with pytest.raises(RuntimeError):
        with launch_receipt(native) as receipt:
            native._launch()
            raise RuntimeError('copyback failed')
    assert native._launch is original
    assert receipt == dict(attempted=1, accepted=1, completed=False)


@pytest.mark.parametrize('repeat', [0, -1, True])
def test_adapter_rejects_invalid_repeats_before_device_access(repeat):
    with pytest.raises(ValueError, match='positive'):
        measure('nvidia', '/missing', repeat=repeat)


def test_independent_ssd_oracle_known_single_step_derivatives():
    values = [np.array([[[2.]]]), np.array([[3.]]), np.array([[[5.]]]),
              np.array([[[7.]]]), np.array([[[11.]]])]
    np.testing.assert_allclose(reference(values), [[[301.]]])
    got = finite_vjp(values, np.ones((1, 1, 1)))
    for gradient, expected in zip(got, [35, 77, 14, 43, 21], strict=True):
        np.testing.assert_allclose(gradient, expected, rtol=1e-8)
