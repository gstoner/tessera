"""The compiler outlines paired producer SSA without rebuilding arithmetic."""

import base64
import json
import re
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def graph(storage="f16", wrt=1, m=7, k=32, n=9):
    return f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @product(%a: tensor<{m}x{k}x{storage}>, %b: tensor<{k}x{n}x{storage}>) -> tensor<{m}x{n}xf32>
    attributes {{tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [{wrt} : i64]}} {{
    %n = "tessera.rmsnorm"(%a) {{eps = 1.0E-5 : f64}} : (tensor<{m}x{k}x{storage}>) -> tensor<{m}x{k}x{storage}>
    %r = "tessera.relu"(%n) {{schedule = "flat_128"}} : (tensor<{m}x{k}x{storage}>) -> tensor<{m}x{k}x{storage}>
    %p = "tessera.matmul"(%r, %b) {{rhs_storage_order = "row_major", output_dtype = "fp32"}} : (tensor<{m}x{k}x{storage}>, tensor<{k}x{n}x{storage}>) -> tensor<{m}x{n}xf32>
    return %p : tensor<{m}x{n}xf32>
  }}
}}"""


def export(source, index=None):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("requires matching production compiler")
    options = "export-scaled-program=true"
    if index is not None:
        options += f" select-scaled-member={index}"
    return run_tessera_opt(tool, source, "--tessera-autodiff-forward=" + options)


@pytest.mark.compiler_route
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_native_paired_tensor_ssa_retains_shared_edge_and_two_owned_outputs(storage):
    source = graph(storage)
    ir = export(source)
    value = re.search(r'tessera.native.sm120_tensor_program_json = "([^"]+)"', ir)
    assert value is not None
    plan = json.loads(base64.b64decode(value[1], validate=True))
    assert plan["schema"] == "tessera.native.sm120_tensor_program.v7"
    assert plan["kind"] == "paired_jvp"
    assert plan["role_indices"] == [0, 1, 2]
    assert plan["outputs"] == [5, 6]
    assert [s["inputs"] for s in plan["steps"]] == [[0], [3], [4, 1], [4, 2]]
    assert plan["buffers"][4]["last_read"] == 3
    assert plan["buffers"][4]["ownership"] == "private_scratch"
    for index in (5, 6):
        assert plan["buffers"][index]["ownership"] == "returned_output"
        assert plan["buffers"][index]["last_read"] == 4
    assert "func.func private @product__jvp" in plan["paired_graph_ir"]
    assert "product__jvp" not in plan["source_graph_ir"]
    for index, member in enumerate(plan["member_graphs"]):
        assert export(source, index) == member
        schedule = run_tessera_opt(find_tessera_opt(), member, "--tessera-graph-to-schedule")
        tile = run_tessera_opt(find_tessera_opt(), schedule, "--tessera-schedule-to-tile")
        assert ("tile.mma " if index >= 2 else "tile.activation_kernel" if index == 1 else "tile.norm_kernel") in tile


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 16, 1), (7, 32, 9), (17, 48, 19), (129, 256, 33)])
def test_native_paired_tensor_owner_executes_both_outputs_and_profiles(storage, shape):
    import numpy as np
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP

    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and matching native compiler")
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(151067)
    m, k, n = shape
    a = rng.uniform(-1, 1, (m, k)).astype(dtype)
    b = rng.uniform(-0.25, 0.25, (k, n)).astype(dtype)
    db = rng.uniform(-0.125, 0.125, (k, n)).astype(dtype)
    wide = a.astype(np.float32)
    normalized = (wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + np.float32(1e-5))).astype(dtype)
    edge = np.maximum(normalized, 0).astype(np.float32)
    expected = (edge @ b.astype(np.float32), edge @ db.astype(np.float32))
    owner = PreparedTensorJVP(graph(storage, m=m, k=k, n=n))
    try:
        outputs = owner((a, b, db))
        saved = tuple(v.copy() for v in outputs)
        assert not np.shares_memory(*outputs)
        for actual, reference in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        stats = owner.scratch_stats()
        profile = owner.profile(repeats=11, captured=True)
        assert profile["timing_domain"] == "cuda_graph_batch"
        assert len(profile["stage_ms"]) == 4
        assert all(np.isfinite(x) and x >= 0 for x in [profile["program_ms"], *profile["stage_ms"]])
        for actual, reference in zip(profile["outputs"], expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        owner((a, b, db))
        assert owner.scratch_stats() == stats
        for actual, reference in zip(outputs, saved, strict=True):
            np.testing.assert_array_equal(actual, reference)
        with pytest.raises(RuntimeError, match="shape/stride/storage"):
            owner((a, b, db.astype(np.float32)))
        for actual, reference in zip(outputs, saved, strict=True):
            np.testing.assert_array_equal(actual, reference)
    finally:
        owner.close()
    with pytest.raises(ValueError, match="closed"):
        owner((a, b, db))


def _frontend_product(a, b):
    import tessera as ts

    return ts.ops.matmul(ts.ops.relu(ts.ops.rmsnorm(a, eps=1e-5)), b, output_dtype="fp32")


def _frontend_product_reversed(b, a):
    import tessera as ts

    return ts.ops.matmul(ts.ops.relu(ts.ops.rmsnorm(a, eps=1e-5)), b, output_dtype="fp32")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_public_native_rhs_jvp_uses_sealed_program_not_python_member_launches(storage, monkeypatch, permuted):
    import numpy as np
    import ml_dtypes
    import tessera as ts
    from tessera import runtime as rt

    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and production compiler")
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(151068)
    a = rng.uniform(-1, 1, (7, 32)).astype(dtype)
    b = rng.uniform(-0.25, 0.25, (32, 9)).astype(dtype)
    db = rng.uniform(-0.125, 0.125, (32, 9)).astype(dtype)
    wide = a.astype(np.float32)
    edge = np.maximum(
        (wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + np.float32(1e-5))).astype(dtype), 0
    ).astype(np.float32)
    expected = (edge @ b.astype(np.float32), edge @ db.astype(np.float32))
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("b",))(
        _frontend_product_reversed if permuted else _frontend_product
    )
    arguments = (b, a) if permuted else (a, b)

    def forbidden(*args, **kwargs):
        pytest.fail("paired program used a Python member launcher")

    monkeypatch.setattr(rt, "_submit_nvidia_sm120_native", forbidden)
    retained = []
    try:
        for _ in range(2):
            outputs = fn.native_jvp(*arguments, tangents=(db,))
            for value, reference in zip(outputs, expected, strict=True):
                np.testing.assert_allclose(value, reference, rtol=2e-5, atol=2e-5)
            retained.append(outputs)
            assert fn.last_jvp_execution["execution_kind"] == "native_gpu"
            assert fn.last_jvp_execution["native_call_binding"] == "prepared_cpp_tensor_program"
            assert fn.last_jvp_execution["frontend_authority"] == "tracer"
        assert len(fn._native_prepared_jvp_calls) == 1
        assert not np.shares_memory(retained[0][0], retained[1][0])
        assert not np.shares_memory(retained[0][1], retained[1][1])
    finally:
        for owner in fn._native_prepared_jvp_calls.values():
            owner.close()


@pytest.mark.compiler_route
def test_native_paired_tensor_does_not_mislabel_active_source_derivatives():
    with pytest.raises(RuntimeError, match="RHS-linear"):
        export(graph(wrt=0))


def test_native_paired_tensor_program_lease_and_metadata_guards():
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP

    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and production compiler")
    values = (np.ones((7, 32), np.float16), np.ones((32, 9), np.float16), np.full((32, 9), 0.125, np.float16))
    first = PreparedTensorJVP(graph())
    second = PreparedTensorJVP(graph())
    try:
        output = first(values)
        saved = tuple(v.copy() for v in output)
        second(values)
        with pytest.raises(RuntimeError, match="lease is stale"):
            first.profile(repeats=3)
        first(values)
        first.profile(repeats=3)
        first.plan["buffers"][4]["last_read"] = 2
        with pytest.raises(ValueError, match="metadata changed"):
            first(values)
        for actual, reference in zip(output, saved, strict=True):
            np.testing.assert_array_equal(actual, reference)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("producer", ["rmsnorm", "layer_norm", "softmax"])
@pytest.mark.parametrize("schedule", ["serial", "cooperative_128"])
def test_paired_owner_preserves_each_registered_coefficient_and_row_mapping(storage, producer, schedule):
    import numpy as np
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP

    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and production compiler")
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    source = graph(storage).replace('"tessera.rmsnorm"', f'"tessera.{producer}"')
    attrs = (
        f'axis = -1 : i64, schedule = "{schedule}"'
        if producer == "softmax"
        else f'eps = 1.0E-5 : f64, schedule = "{schedule}"'
    )
    source = source.replace("eps = 1.0E-5 : f64", attrs)
    rng = np.random.default_rng(151071)
    a = rng.uniform(-1, 1, (7, 32)).astype(dtype)
    b = rng.uniform(-0.25, 0.25, (32, 9)).astype(dtype)
    db = rng.uniform(-0.125, 0.125, (32, 9)).astype(dtype)
    wide = a.astype(np.float32)
    if producer == "softmax":
        exp = np.exp(wide - wide.max(axis=-1, keepdims=True))
        coefficient = exp / exp.sum(axis=-1, keepdims=True)
    else:
        centered = wide - wide.mean(axis=-1, keepdims=True) if producer == "layer_norm" else wide
        coefficient = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + np.float32(1e-5))
    edge = np.maximum(coefficient.astype(dtype), 0).astype(np.float32)
    expected = (edge @ b.astype(np.float32), edge @ db.astype(np.float32))
    owner = PreparedTensorJVP(source)
    try:
        outputs = owner((a, b, db))
        for actual, reference in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
    finally:
        owner.close()
