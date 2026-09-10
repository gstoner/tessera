"""Concrete surviving findings from the architecture sweep."""
import numpy as np
import pytest
from tessera.shape import Dim, DimProduct, dims_compatible, matmul_shape
from tessera.compiler.distributed_planner import DistributedPlan, LayerSpec
from tessera.compiler.native_source_state import decode_source_exception


def test_structural_product_equality_reaches_matmul():
    h, d = Dim('H'), Dim('D')
    assert matmul_shape((3, h*d), (d*h, 7)) == (3, 7)
    assert dims_compatible(DimProduct((2,h,d)), h*(2*d))
    assert dims_compatible(h+0, h)
    assert not dims_compatible(Dim('same',2), Dim('same',3))
    assert not dims_compatible(Dim('1*H'), h+0)
    assert not dims_compatible(h*d, h*h)


@pytest.mark.parametrize('stages', [(1,), (0,2), (-1,0), (False,), (0,1.0), (0,10**12)])
def test_pipeline_stage_gaps_and_invalid_indices_refuse(stages):
    plan=DistributedPlan({'pp':2}, [LayerSpec(str(i),pp_stage=stage) for i,stage in enumerate(stages)])
    with pytest.raises(ValueError,match='pipeline stages'):plan.validate()


def test_multiple_layers_may_share_contiguous_stages():
    DistributedPlan({'pp':2}, [LayerSpec(str(i),pp_stage=stage) for i,stage in enumerate((0,0,1,1))]).validate()


@pytest.mark.parametrize('child', [('Missing', ['bad']), ('ValueError', ['bad'], None, ['file',False])])
def test_malformed_exception_child_never_runs_root_constructor(child):
    calls=[]
    class CustomError(Exception):
        def __init__(self,*args):
            calls.append(args)
            super().__init__(*args)
    contract={'error_table': [('CustomError',['root'],('edge',child))]}
    with pytest.raises(RuntimeError):
        decode_source_exception(contract,[np.array([1],np.float32)],exception_types={'CustomError':CustomError})
    assert calls==[]


def test_native_exception_location_is_not_a_fabricated_python_frame():
    contract={'error_table':[('ValueError',['bad'],None,['native.py',19])]}
    error=decode_source_exception(contract,[np.array([1],np.float32)])
    assert error.__traceback__ is None
    assert error.__notes__ == ['Native source raise at native.py:19; no Python frame executed there']
