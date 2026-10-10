"""Public JVP retains compiler ownership and frontend request invariants."""
import copy

import numpy as np
import pytest
import tessera as ts

from tessera.autodiff import jvp, tape, TesseraAutodiffError
from tessera.compiler.constraints import Range, TesseraConstraintError
from tessera.compiler.jit import JitFn
from tests.unit.test_ordered_resident_tensor_dag import Buffer


ROW_INPUT = ts.Tensor["M", "K", "fp32"]


def rows(x: ROW_INPUT):
    return ts.ops.sum(x, axis=1)


def fake_native(self, *args, tangents):
    self._enforce_call_time_constraints(args, {})
    self.last_jvp_execution = {"execution_kind": "native_gpu", "compiler_path": "test_native"}
    return args[0], tangents[0]


@pytest.mark.parametrize("target", ["nvidia_sm120", "rocm_gfx1201", "rocm_gfx1151", "x86", "apple_gpu"])
def test_public_jvp_projects_request_without_calling_eager(target, monkeypatch):
    fn = ts.jit(target=target)(rows)
    original = copy.deepcopy(fn.graph_ir)
    primal = np.ones((3, 4), np.float32)
    tangent = np.full_like(primal, 2)
    monkeypatch.setattr(JitFn, "native_jvp", fake_native)
    result = jvp(fn, primal, tangent)
    assert result[0] is primal and result[1] is tangent
    assert fn.graph_ir == original and fn.differentiation_request is None
    witness, owner = fn._native_public_jvp_owners[(0,)]
    assert owner is not fn and owner.differentiation_request.wrt_indices == (0,)
    assert owner.differentiation_request.native_required
    assert fn.last_jvp_execution["public_transform"] == "jvp"
    assert jvp(fn, primal, tangent)[0] is primal
    assert fn._native_public_jvp_owners[(0,)][1] is owner


def test_public_jvp_preserves_resident_buffers_without_numpy_conversion(monkeypatch):
    fn = ts.jit(target="nvidia_sm120")(rows)
    primal, tangent = (Buffer((3, 4), np.float32) for _ in range(2))
    monkeypatch.setattr(JitFn, "native_jvp", fake_native)
    result = jvp(fn, primal, tangent)
    assert result[0] is primal and result[1] is tangent


def test_public_jvp_refreshes_constraints_before_backend(monkeypatch):
    fn = ts.jit(target="nvidia_sm120")(rows)
    primal = np.ones((3, 4), np.float32)
    monkeypatch.setattr(JitFn, "native_jvp", fake_native)
    jvp(fn, primal, primal)
    old = fn._native_public_jvp_owners[(0,)][1]
    fn.constraints.add(Range("M", 1, 2))
    with pytest.raises(TesseraConstraintError, match="M"):
        jvp(fn, primal, primal)
    assert fn._native_public_jvp_owners[(0,)][1] is not old
    assert fn.last_jvp_execution is None


def test_public_jvp_does_not_accept_a_reference_execution_receipt(monkeypatch):
    fn = ts.jit(target="nvidia_sm120")(rows)
    def reference(self, *args, tangents):
        self.last_jvp_execution = {"execution_kind": "reference_cpu"}
        return args[0], tangents[0]
    monkeypatch.setattr(JitFn, "native_jvp", reference)
    with pytest.raises(TesseraAutodiffError, match="native execution receipt"):
        jvp(fn, np.ones((2, 3), np.float32), np.ones((2, 3), np.float32))


def test_public_native_jvp_is_not_an_eager_higher_order_product(monkeypatch):
    fn = ts.jit(target="nvidia_sm120")(rows)
    def forbidden(*args, **kwargs):
        pytest.fail("nested eager AD entered native product")
    monkeypatch.setattr(JitFn, "native_jvp", forbidden)
    value = np.ones((2, 3), np.float32)
    with tape(), pytest.raises(TesseraAutodiffError, match="higher-order"):
        jvp(fn, value, value)


def two_roots(x, y):
    return ts.ops.add(x, y)


def test_none_tangent_is_explicit_native_activity(monkeypatch):
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("y",))(two_roots)
    before = copy.deepcopy(fn.differentiation_request)
    x, y = (np.ones((2, 3), np.float32) for _ in range(2))
    monkeypatch.setattr(JitFn, "native_jvp", fake_native)
    result = jvp(fn, (x, y), (x, None))
    assert result[1] is x
    assert fn._native_public_jvp_owners[(0,)][1].differentiation_request.mode == "forward"
    assert fn.differentiation_request == before


def test_all_inactive_native_tangents_do_not_launch(monkeypatch):
    fn = ts.jit(target="nvidia_sm120")(rows)
    def forbidden(*args, **kwargs):
        pytest.fail("inactive public JVP launched")
    monkeypatch.setattr(JitFn, "native_jvp", forbidden)
    with pytest.raises(ValueError, match="active tangent"):
        jvp(fn, np.ones((2, 3), np.float32), None)
