"""Owning bounded multi-producer Graph packages and native capacity/lifetime."""
import itertools
import json
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import _storage

pytestmark = pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")
AXES = [tuple(axis for axis, bit in zip(("M","N","K"),bits,strict=True) if bit)
        for bits in itertools.product((False,True),repeat=3) if any(bits)]


def two(source,rhs):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.rmsnorm(source,eps=1e-5),axis=-1),
                         rhs,output_dtype="fp32")


def three(source,rhs):
    value = ts.ops.layer_norm(source,eps=1e-5)
    value = ts.ops.rmsnorm(value,eps=1e-5)
    return ts.ops.matmul(ts.ops.softmax(value,axis=-1),rhs,output_dtype="fp32")


def three_fused(source,rhs,bias,residual):
    value = ts.ops.layer_norm(source,eps=1e-5)
    value = ts.ops.rmsnorm(value,eps=1e-5)
    return ts.ops.matmul(ts.ops.softmax(value,axis=-1),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


def oracle(values,producers):
    source,rhs = values[:2]
    storage = source.dtype
    value = source.astype(np.float64)
    if producers == 3:
        centered = value-value.mean(axis=-1,keepdims=True)
        value = (centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
    value = (value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
    exponential = np.exp(value-value.max(axis=-1,keepdims=True))
    value = (exponential/exponential.sum(axis=-1,keepdims=True)).astype(storage).astype(np.float64)
    result = value @ rhs.astype(np.float64)
    if len(values) == 4:
        result = np.maximum(result+values[2].astype(np.float64),0)+values[3].astype(np.float64)
        result = result.astype(np.float16)
    return result


def inputs(extents,dtype,order="C",fused=False,seed=1260):
    m,n,k = (extents[axis] for axis in ("M","N","K"))
    rng = np.random.default_rng(seed)
    storage = _storage(dtype)
    values = (rng.normal(0,.2,(m,k)).astype(storage),
              np.array(rng.normal(0,.2,(k,n)),dtype=storage,order=order))
    if fused:
        values += (rng.normal(0,.2,n).astype(np.float32),
                   rng.normal(0,.2,(m,n)).astype(np.float32))
    return values


@pytest.mark.parametrize("axes",AXES)
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("producers",[2,3])
def test_bounded_chain_native_public_replay_and_scratch(axes,dtype,producers,monkeypatch):
    assert rt._nvidia_device_name() == "sm_120"
    active = {"M":17,"N":11,"K":32}
    bounds = {axis:{"M":32,"N":24,"K":64}[axis] for axis in axes}
    capacity = {axis:bounds.get(axis,value) for axis,value in active.items()}
    function = ts.jit(target="nvidia_sm120",shape_bounds=bounds)(two if producers==2 else three)
    first_values = inputs(active,dtype)
    first = function(*first_values)
    saved = first.copy()
    program = function._nvidia_lhs_last_program
    packages = function.native_lhs_packages()
    assert len(packages) == producers+1
    assert json.loads(program.native_plan_json)["schema"].endswith(".v4")
    artifact = rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
    np.testing.assert_allclose(first,oracle(first_values,producers),rtol=.015,atol=.002)
    def forbidden(*args,**kwargs):
        raise AssertionError("warm bounded chain invoked compiler or eager execution")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(function,"_fn",forbidden)
    monkeypatch.setattr(function,"_traced_autodiff_module",forbidden)
    try:
        owner = next(iter(function._nvidia_lhs_prepared_calls.values()))
        stats = owner.scratch_stats()
        for extents in (capacity,{axis:1 if axis in axes else value for axis,value in active.items()},active):
            values = inputs(extents,dtype,seed=1261)
            actual = function(*values)
            np.testing.assert_allclose(actual,oracle(values,producers),rtol=.015,atol=.002)
            receipt = rt.launch(artifact,values)
            assert receipt["ok"] and receipt["execution_kind"] == "native_gpu"
            np.testing.assert_allclose(receipt["output"],oracle(values,producers),rtol=.015,atol=.002)
            assert function._nvidia_lhs_last_program is program
            assert function.native_lhs_packages() == packages
            assert owner.scratch_stats() == stats
            np.testing.assert_array_equal(first,saved)
        invalid = dict(capacity)
        invalid[axes[0]] += 1
        monkeypatch.setattr(owner.lib,"tessera_nvidia_matmul_context_identity",forbidden)
        with pytest.raises((ValueError,RuntimeError)):
            function(*inputs(invalid,dtype))
    finally:
        function.close_native_storage()


@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("order",["C","F"])
def test_bounded_three_chain_preserves_fused_final_store(dtype,order,monkeypatch):
    bounds = {"M":32,"N":24,"K":64}
    function = ts.jit(target="nvidia_sm120",shape_bounds=bounds)(three_fused)
    initial = inputs({"M":17,"N":11,"K":32},dtype,order,True)
    np.testing.assert_allclose(function(*initial),oracle(initial,3),rtol=.015,atol=.002)
    def forbidden(*args,**kwargs):
        raise AssertionError("warm fused chain compiled")
    monkeypatch.setattr(subprocess,"run",forbidden)
    try:
        for shape in (bounds,{"M":1,"N":1,"K":1}):
            values = inputs(shape,dtype,order,True,1262)
            np.testing.assert_allclose(function(*values),oracle(values,3),rtol=.015,atol=.002)
    finally:
        function.close_native_storage()
