"""Raw resident RHS layout projection stays semantic and metadata-only."""
from copy import deepcopy

import pytest
import tessera as ts
from tessera.compiler.nvidia_tensor_lhs import project_rhs_storage
from tests.unit.test_ordered_resident_tensor_dag import Buffer


def plain(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,output_dtype="fp32")


@pytest.mark.parametrize("dynamic",[False,True])
def test_raw_resident_projection_seals_row_major_without_mutating_graph(dynamic):
    fn=ts.jit(target="nvidia_sm120")(plain)
    roots=[Buffer((3,5)),Buffer((5,7))]
    module=fn._traced_autodiff_module(tuple(roots),{})
    saved=deepcopy(module)
    projected=project_rhs_storage(module,roots,dynamic=dynamic)
    assert projected.functions[0].body[-1].kwargs["rhs_storage_order"]=="row_major"
    assert module==saved


@pytest.mark.parametrize("policy_source",["authored","explicit"])
def test_conflicting_resident_raw_rhs_layout_is_not_normalized(policy_source):
    fn=ts.jit(target="nvidia_sm120")(plain)
    roots=[Buffer((3,5)),Buffer((5,7))]
    module=fn._traced_autodiff_module(tuple(roots),{})
    if policy_source=="authored":module.functions[0].body[-1].kwargs["rhs_storage_order"]="col_major"
    with pytest.raises(ValueError,match="row-major"):
        project_rhs_storage(module,roots,dynamic=True,
                            rhs_storage_order="col_major" if policy_source=="explicit" else None)


def test_resident_projection_revalidates_physical_pitch():
    fn=ts.jit(target="nvidia_sm120")(plain)
    roots=[Buffer((3,5)),Buffer((5,7))]
    module=fn._traced_autodiff_module(tuple(roots),{})
    roots[1].interface["strides"]=(18,2)
    with pytest.raises(ValueError,match="compact"):
        project_rhs_storage(module,roots)


@pytest.mark.parametrize("resident,layout,expected",[
    (False,"col_major","key"),(True,"row_major","key"),(True,"col_major",("key","resident_row_major"))])
def test_bounded_layout_cache_preserves_host_and_resident_physical_contracts(resident,layout,expected):
    from types import SimpleNamespace
    from tessera.compiler.bounded_nvidia_lhs import _resident_layout_key
    package=SimpleNamespace(rhs_chain=(),edge=SimpleNamespace(consumer=SimpleNamespace(
        descriptor=SimpleNamespace(provenance={"b_layout":layout}))))
    assert _resident_layout_key({"key":package},"key",resident)==expected
    assert _resident_layout_key({},"key",resident)=="key"
