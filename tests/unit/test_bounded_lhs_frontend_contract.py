"""Bounds/source certificates are checked without tools or a live GPU."""
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.bounded_nvidia_lhs import SourceCertificate


def plain(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,output_dtype="fp32")


def shape_branch(source,rhs):
    if source.shape[0]>1:
        return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs)
    return ts.ops.matmul(ts.ops.softmax(source,axis=-1),rhs)


def hidden_helper(source,rhs):
    return shape_branch(source,rhs)


@pytest.mark.parametrize("bounds",[{},{"Q":1},{"M":True},{"M":0},{"K":-1},{"N":2**31},[("M",3)]])
def test_bad_bounds_rejected_before_decoration(bounds):
    with pytest.raises(ValueError,match="shape_bounds"):
        ts.jit(target="nvidia_sm120",shape_bounds=bounds)


@pytest.mark.parametrize("target",["cpu","apple_gpu","rocm_gfx1201","nvidia_sm90",None])
def test_bounds_have_explicit_backend_contract(target):
    with pytest.raises(ValueError,match="primal nvidia_sm120"):
        ts.jit(target=target,shape_bounds={"M":3})


@pytest.mark.parametrize("option",[{"autodiff":"reverse"},{"wrt":("source",)},{"source_control_flow":True}])
def test_bounds_do_not_admit_unproved_ad_or_source_routes(option):
    with pytest.raises(ValueError,match="primal nvidia_sm120"):
        ts.jit(target="nvidia_sm120",shape_bounds={"M":3},**option)


@pytest.mark.parametrize("fn",[shape_branch,hidden_helper])
def test_shape_variant_or_hidden_helper_has_no_polymorphic_certificate(fn):
    with pytest.raises(ValueError,match="bounded LHS"):
        ts.jit(target="nvidia_sm120",shape_bounds={"M":3})(fn)


def test_false_source_cannot_hide_live_shape_branch():
    fake="""
def shape_branch(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs)
"""
    with pytest.raises(ValueError,match="live code"):
        SourceCertificate(shape_branch,fake)


def test_false_source_cannot_hide_live_helper_dependency():
    fake="""
def hidden_helper(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs)
"""
    with pytest.raises(ValueError,match="dependency"):
        SourceCertificate(hidden_helper,fake)


def test_capacity_projection_does_not_normalize_malformed_original_graph(monkeypatch):
    from copy import deepcopy
    from tessera.compiler import nvidia_tensor_lhs as lhs
    fn=ts.jit(target="nvidia_sm120")(plain)
    graph=deepcopy(fn._traced_autodiff_module((np.ones((3,5),np.float16),np.ones((5,7),np.float16)),{}))
    graph.functions[0].body[1].result_type="tensor<3x8xf32>"
    def forbidden(*args,**kwargs):raise AssertionError("malformed Graph reached compiler")
    monkeypatch.setattr(lhs,"find_tessera_opt",forbidden)
    with pytest.raises(ValueError,match="storage/shape"):
        lhs.package_traced_lhs(graph,shape_bounds={"M":8,"N":8,"K":8})


def test_capacity_overflow_remains_argument_error_without_compiler(monkeypatch):
    from tessera.compiler import nvidia_tensor_lhs as lhs
    fn=ts.jit(target="nvidia_sm120")(plain)
    graph=fn._traced_autodiff_module((np.ones((3,5),np.float16),np.ones((5,7),np.float16)),{})
    def forbidden(*args,**kwargs):raise AssertionError("invalid bound reached compiler")
    monkeypatch.setattr(lhs,"find_tessera_opt",forbidden)
    with pytest.raises(ValueError,match="bound"):
        lhs.package_traced_lhs(graph,shape_bounds={"M":2})


@pytest.mark.parametrize("order",[True,[],{},"diagonal","ROW_MAJOR"])
def test_bounded_rhs_order_rejects_invalid_configuration(order):
    import tessera as ts
    with pytest.raises(ValueError,match="rhs_storage_order"):
        ts.jit(target="nvidia_sm120",shape_bounds={"M":32},rhs_storage_order=order)


def test_rhs_order_requires_bounded_jit():
    import tessera as ts
    with pytest.raises(ValueError,match="rhs_storage_order"):
        ts.jit(target="nvidia_sm120",rhs_storage_order="row_major")


def test_rhs_request_preserves_authored_graph_storage():
    from tessera.compiler import nvidia_tensor_lhs as lhs
    fn=ts.jit(target="nvidia_sm120")(plain)
    operands=(np.ones((3,5),np.float16),np.ones((5,7),np.float16))
    graph=fn._traced_autodiff_module(operands,{})
    projected=lhs.project_rhs_storage(graph,operands,dynamic=True,rhs_storage_order="row_major")
    assert "rhs_storage_order" not in graph.functions[0].body[1].kwargs
    assert projected.functions[0].body[1].kwargs["rhs_storage_order"]=="row_major"
    with pytest.raises(ValueError,match="conflicts"):
        lhs.project_rhs_storage(projected,operands,dynamic=True,rhs_storage_order="col_major")
    assert projected.functions[0].body[1].kwargs["rhs_storage_order"]=="row_major"


def test_dispatcher_rejects_released_frontend_owner_before_tracing():
    import gc
    from tessera.compiler.bounded_nvidia_lhs import BoundedLhsDispatcher

    class Owner:
        pass

    class Certificate:
        def validate(self):
            pass

    owner = Owner()
    dispatcher = BoundedLhsDispatcher(owner, (("M", 4),), Certificate())
    del owner
    gc.collect()
    with pytest.raises(ValueError, match="frontend owner has been released"):
        dispatcher((), {})
