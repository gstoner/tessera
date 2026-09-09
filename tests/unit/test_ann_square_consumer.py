"""Square tails amplify affine error and cannot inherit nonexpansive bounds."""
from fractions import Fraction
import numpy as np
import pytest
from benchmarks.record_native_ann_execution import source
from tessera.compiler.native_ann import prepare_native_ann, affine_error_bound, _exact_output
from tessera.compiler.scheduled_matmul import find_tessera_opt


def test_square_tail_carries_analytic_amplification():
    if find_tessera_opt() is None: pytest.skip('native compiler required')
    pair = prepare_native_ann(source(3,2,'square'), allow_reassociation=True)
    before, after = pair.validate()
    assert before[3] == after[3] == 'square'
    value = np.array([[-1,1],[.5,-.5],[0,1]], np.float32)
    bound = affine_error_bound(pair, 1.0)
    error = np.abs(_exact_output(before,value)-_exact_output(after,value))
    assert bound > 0 and max(error.flat) <= bound
    with pytest.raises(ValueError, match='overflow'):
        affine_error_bound(pair, 1e25)
    assert isinstance(bound, Fraction)
