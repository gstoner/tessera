"""Owning ROCm native SwiGLU math; full NVFP4 MLP composition is separate."""

import os
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_math_native as native
from tests.device.rocm.test_native_math_package_jit import target as _target_fixture
from tests.unit.test_rocm_math_widening import module


def silu(a, b):
    return ts.ops.silu_mul(a, b)


def silu_reverse(a, b):
    return ts.ops.silu_mul(b, a)


def silu_wide(a, b):
    return ts.ops.silu_mul(ts.ops.cast(a, "fp32"), ts.ops.cast(b, "fp32"))


def silu_wide_reverse(a, b):
    return ts.ops.silu_mul(ts.ops.cast(b, "fp32"), ts.ops.cast(a, "fp32"))


def oracle(values, reverse=False):
    a, b = values[::-1] if reverse else values
    a, b = a.astype(np.float64), b.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return ((a / (1.0 + np.exp(-a))) * b).astype(np.float32)


def function(storage, reverse=False):
    return (silu_reverse if reverse else silu) if storage == "f32" else (silu_wide_reverse if reverse else silu_wide)


@pytest.fixture
def target():
    return _target_fixture.__wrapped__()


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", [(3, 17), (2, 3, 257)])
def test_native_silu_jit_changed_inputs_replay_and_retained_output(target, storage, reverse, shape, monkeypatch):
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}[storage]
    rng = np.random.default_rng(1610)
    values = [rng.uniform(-7, 7, shape).astype(dtype), rng.normal(0, 0.7, shape).astype(dtype)]
    original = [a.copy() for a in values]
    expected = oracle(values, reverse)
    fn = ts.jit(target=target)(function(storage, reverse))

    def forbidden(*args, **kwargs):
        pytest.fail("native SiLU invoked eager arithmetic, legacy executor or warm compiler")

    monkeypatch.setattr(rt, "_execute_rocm_compiled_silu_mul", forbidden)
    # First tracing must also use shape/typed IR, not evaluate input arithmetic.
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbidden)
        first = fn(*values)
    np.testing.assert_allclose(first, expected, rtol=3e-6, atol=2e-7)
    assert fn.execution_kind == "native_gpu" and first.dtype == np.float32
    for a, b in zip(values, original, strict=True):
        np.testing.assert_array_equal(a, b)
    held = first.copy()
    artifact = fn.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "rocm_math_native_descriptor"
    assert all((artifact.graph_ir, artifact.schedule_ir, artifact.tile_ir, artifact.target_ir))
    assert "family=rocm_math" in artifact.schedule_ir and 'family = "binary_math"' in artifact.tile_ir
    info = artifact.launch_descriptor.provenance["native_math"]
    assert info["kind"] == "silu_mul" and info["storage"] == storage
    assert info["roles"] == ([1, 0] if reverse else [0, 1])
    native.validate_math_runtime_artifact(artifact)
    monkeypatch.setattr(native, "run_tessera_opt", forbidden)
    changed = [(a.astype(np.float32) * 0.75).astype(dtype) for a in values]
    expected = oracle(changed, reverse)
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbidden)
        again = fn(*changed)
    np.testing.assert_allclose(again, expected, rtol=3e-6, atol=2e-7)
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    buffers = dict(zip(info["bindings"][:-1], changed, strict=True))
    buffers[info["bindings"][-1]] = np.empty(shape, np.float32)
    receipt = rt.launch(restored, {"buffers": buffers, "scalars": {"N": info["elements"]}})
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    np.testing.assert_allclose(buffers[info["bindings"][-1]], expected, rtol=3e-6, atol=2e-7)
    np.testing.assert_array_equal(first, held)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(1,), (257,)])
def test_gfx1201_rank_one_widening_has_native_silu_execution(target, storage, shape):
    if target != "rocm_gfx1201":
        pytest.skip("new rank-one cast admission is gfx1201-only; sibling proof required")
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    a = np.linspace(-5, 5, np.prod(shape)).reshape(shape).astype(dtype)
    b = np.full(shape, 0.75, dtype)
    fn = ts.jit(target=target)(silu_wide)
    actual = fn(a, b)
    assert fn.execution_kind == "native_gpu" and actual.dtype == np.float32
    np.testing.assert_allclose(actual, oracle((a, b)), rtol=3e-6, atol=2e-7)
    assert fn.runtime_artifact().metadata["compiler_path"] == "rocm_math_native_descriptor"


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_native_silu_preserves_ieee_classes_and_zero_sign(target, storage):
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}[storage]
    a = np.resize(np.array([-np.inf, -100, -5, -0.0, 0.0, 1, 50, np.inf, np.nan]), (3, 17)).astype(dtype)
    b = np.ones_like(a)
    expected = oracle((a, b))
    fn = ts.jit(target=target)(function(storage))
    actual = fn(a, b)
    assert fn.execution_kind == "native_gpu"
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=3e-6, atol=2e-7)
    zeros = expected == 0
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_native_silu_reuses_shape_free_images(target, storage):
    images = []
    for shape in ((3, 17), (7, 259)):
        graph = module("div", storage, shape=shape, reverse=True)
        graph.functions[0].body[-1].op_name = "tessera.silu_mul"
        if storage == "f32":
            graph.functions[0].body = graph.functions[0].body[-1:]
            graph.functions[0].body[0].operands = ["%b", "%a"]
        recipe = native.lower_math_graph(graph, target)
        package = native.package_math_recipe(recipe, pipeline_name="tessera-lower-to-rocm")
        images.append(package.image.payload)
    assert images[0] == images[1]


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_native_silu_fresh_process_replay_without_compiler_or_eager(target, storage, tmp_path):
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}[storage]
    rng = np.random.default_rng(1710)
    values = [rng.normal(0, 0.7, (3, 17)).astype(dtype) for _ in range(2)]
    fn = ts.jit(target=target)(function(storage))
    actual = fn(*values)
    expected = oracle(values)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    artifact = fn.runtime_artifact()
    (tmp_path / "artifact.json").write_text(artifact.to_json())
    np.savez(
        tmp_path / "inputs.npz",
        a=values[0].view(np.uint16) if storage == "bf16" else values[0],
        b=values[1].view(np.uint16) if storage == "bf16" else values[1],
        expected=expected,
    )
    script = r"""
import json,sys,subprocess
from pathlib import Path
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_math_native as native
p=Path(sys.argv[1]); storage=sys.argv[2]; target=sys.argv[3]
assert rt._rocm_live_arch()==target.removeprefix("rocm_")
def forbidden(*a,**k):raise AssertionError("replay used compiler/eager/legacy arithmetic")
subprocess.run=forbidden;subprocess.check_output=forbidden;subprocess.Popen=forbidden
np.exp=forbidden;native.run_tessera_opt=forbidden;rt._execute_rocm_compiled_silu_mul=forbidden
artifact=rt.RuntimeArtifact.from_json((p/"artifact.json").read_text())
data=np.load(p/"inputs.npz")
a,b=data["a"],data["b"]
if storage=="bf16":a,b=a.view(ml_dtypes.bfloat16),b.view(ml_dtypes.bfloat16)
info=artifact.launch_descriptor.provenance["native_math"]
buffers=dict(zip(info["bindings"][:-1],(a,b),strict=True));buffers[info["bindings"][-1]]=np.empty_like(data["expected"])
receipt=rt.launch(artifact,{"buffers":buffers,"scalars":{"N":info["elements"]}})
assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
np.testing.assert_allclose(buffers[info["bindings"][-1]],data["expected"],rtol=3e-6,atol=2e-7)
print("native SiLU fresh replay passed")
"""
    env = dict(os.environ, TESSERA_OPT="/no/compiler", TESSERA_ROCM_OPT="/no/compiler")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), storage, target],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh replay passed" in result.stdout


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", [(3, 17), (2, 3, 257)])
def test_default_half_silu_uses_native_matching_store(target, storage, reverse, shape, monkeypatch):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    finfo = np.finfo(np.float16) if storage == "f16" else ml_dtypes.finfo(ml_dtypes.bfloat16)
    rng = np.random.default_rng(1810)
    values = [rng.uniform(-6, 6, shape).astype(dtype) for _ in range(2)]
    expected = oracle(values, reverse)
    fn = ts.jit(target=target)(silu_reverse if reverse else silu)

    def forbidden(*a, **k):
        pytest.fail("matching-half SiLU used eager arithmetic or legacy routing")

    monkeypatch.setattr(rt, "_execute_rocm_compiled_silu_mul", forbidden)
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbidden)
        actual = fn(*values)
    assert actual.dtype == np.dtype(dtype) and fn.execution_kind == "native_gpu"
    # One final-store ulp, plus the f32 arithmetic budget; no half intermediates.
    bound = float(finfo.eps) * np.maximum(np.abs(expected.astype(np.float64)), float(finfo.tiny))
    bound += 8 * np.finfo(np.float32).eps * np.abs(expected.astype(np.float64))
    assert np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64)) <= bound)
    artifact = fn.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "rocm_math_native_descriptor"
    info = artifact.launch_descriptor.provenance["native_math"]
    assert info["output_storage"] == info["storage"] == storage
    assert artifact.launch_descriptor.abi_id.endswith("." + storage + "_" + storage + ".v1")
    native.validate_math_runtime_artifact(artifact)
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    buffers = dict(zip(info["bindings"][:-1], values, strict=True))
    output_name = info["bindings"][-1]
    buffers[output_name] = np.empty(shape, dtype)
    receipt = rt.launch(restored, {"buffers": buffers, "scalars": {"N": info["elements"]}})
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    np.testing.assert_array_equal(buffers[output_name], actual)
