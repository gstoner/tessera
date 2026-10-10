"""Argument and semantic-bound guards remain testable without a compiler/device."""
import numpy as np
import pytest
from tessera.compiler import nvidia_tensor_lhs as lhs
from tests.device.nvidia.test_lhs_tensor_jit import rms_lhs


@pytest.mark.parametrize("axes",["M",("M","M"),("Q",),(True,),(["M"],),{"M"}])
def test_invalid_axes_do_not_reach_compiler(axes,monkeypatch):
    graph=rms_lhs._traced_autodiff_module((np.ones((3,5),np.float16),np.ones((5,7),np.float16)),{})
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid axes reached compiler")
    monkeypatch.setattr(lhs,"find_tessera_opt",forbidden)
    with pytest.raises(ValueError,match="dynamic axes"):
        lhs.package_traced_lhs(graph,dynamic_axes=axes)


def test_dynamic_row_rhs_reaches_native_compiler_after_contract_validation(monkeypatch):
    graph=rms_lhs._traced_autodiff_module((np.ones((3,5),np.float16),np.ones((5,7),np.float16)),{})
    graph.functions[0].body[1].kwargs["rhs_storage_order"]="row_major"
    def boundary(*args,**kwargs):
        raise AssertionError("admitted row RHS reached native compiler")
    monkeypatch.setattr(lhs,"find_tessera_opt",boundary)
    with pytest.raises(AssertionError,match="admitted row RHS reached native compiler"):
        lhs.package_traced_lhs(graph,dynamic_axes=("M",))


def test_unknown_rhs_storage_is_rejected_before_compiler(monkeypatch):
    graph=rms_lhs._traced_autodiff_module((np.ones((3,5),np.float16),np.ones((5,7),np.float16)),{})
    graph.functions[0].body[1].kwargs["rhs_storage_order"]="diagonal"
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid RHS layout reached compiler")
    monkeypatch.setattr(lhs,"find_tessera_opt",forbidden)
    with pytest.raises(ValueError,match="attributes differ"):
        lhs.package_traced_lhs(graph,dynamic_axes=("M",))
