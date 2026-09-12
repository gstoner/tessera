import numpy as np
import pytest
from benchmarks.nvidia.record_packed_dtype_correctness import case


@pytest.mark.parametrize('dtype', ['int4','nvfp4','fp4_e2m1','fp6_e2m3','fp6_e3m2'])
@pytest.mark.parametrize('axis', [0,1])
def test_packed_probe_exercises_nonzero_signed_values_and_layout(dtype,axis):
    contract,args,expected=case(dtype,axis)
    assert contract['offset'] > 0 and contract['packing_axis'] == axis
    assert args['RowOrigin'] == args['ColumnOrigin'] == 1
    assert expected.shape == args['output'].shape
    assert np.any(expected<0) and np.any(expected>0)
    assert len(np.unique(args['source'])) > 8
    if dtype != 'int4':
        assert len(np.unique(args['scale'])) >= 3
        assert contract['scale_offset'] > 0
