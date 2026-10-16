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
    from tessera.compiler.portable_nvidia_tensor_jvp import validate_plan
    assert validate_plan(json.dumps(plan)) == plan
    malformed = json.loads(json.dumps(plan))
    malformed["buffers"][4]["last_read"] = 2
    with pytest.raises(ValueError, match="lifetime"):
        validate_plan(json.dumps(malformed))
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
        artifact = fn.native_jvp_runtime_artifact()
        assert artifact.artifact_hash == fn.last_jvp_execution["artifact_hash"]
        assert artifact.metadata["arg_names"] == ["primal_0", "primal_1", "tangent_0" if permuted else "tangent_1"]
        restored = rt.RuntimeArtifact.from_json(artifact.to_json())
        roots = (*arguments, db)
        receipt = rt.launch(restored, dict(zip(artifact.metadata["arg_names"], roots, strict=True)))
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        assert receipt["compiler_path"] == "nvidia_sm120_tensor_jvp_compiled"
        assert len(receipt["component_receipts"]) == 4
        for actual, reference in zip(receipt["output"], expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        artifact.metadata["arg_names"][0] = "changed_by_inspection"
        assert fn.native_jvp_runtime_artifact().metadata["arg_names"][0] == "primal_0"
        # Public replay must borrow the current owner, not invalidate its lease.
        resumed = fn.native_jvp(*arguments, tangents=(db,))
        for actual, reference in zip(resumed, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        assert len(fn._native_prepared_jvp_calls) == 1
        assert not np.shares_memory(retained[0][0], retained[1][0])
        assert not np.shares_memory(retained[0][1], retained[1][1])
        from tessera.compiler.jit import TesseraJitError
        with pytest.raises(TesseraJitError, match="shapes must match"):
            fn.native_jvp(*arguments, tangents=(db[:, :-1],))
        with pytest.raises(TesseraJitError, match="no persisted"):
            fn.native_jvp_runtime_artifact()
    finally:
        for owner in fn._native_prepared_jvp_calls.values():
            owner.close()


@pytest.mark.compiler_route
def test_native_paired_tensor_does_not_mislabel_active_source_derivatives():
    # This storage-precision/flat-ReLU graph remains outside the new static
    # explicit-FP32 source-active contract; require its specific diagnostic.
    with pytest.raises(RuntimeError, match="explicit-fp32 RMSNorm/ReLU JVP"):
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

@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_paired_tensor_portable_fresh_process_without_compiler(storage, tmp_path):
    import os
    import subprocess
    import sys
    import numpy as np
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP

    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and production compiler")
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    rng = np.random.default_rng(151070)
    values = (rng.uniform(-1, 1, (7, 32)).astype(dtype),
              rng.uniform(-0.25, 0.25, (32, 9)).astype(dtype),
              rng.uniform(-0.125, 0.125, (32, 9)).astype(dtype))
    owner = PreparedTensorJVP(graph(storage))
    try:
        expected = owner(values)
        data = owner.manifest()
        from tessera.compiler.nvidia_tensor_jvp_runtime import runtime_artifact
        public_artifact = runtime_artifact(owner).to_json()
    finally:
        owner.close()
    (tmp_path / "runtime.json").write_text(public_artifact)
    (tmp_path / "program.json").write_text(json.dumps(data))
    np.savez(tmp_path / "inputs.npz", **{str(i): v.view(np.uint16) for i, v in enumerate(values)})
    script = """
import json, sys
from pathlib import Path
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP
import tessera.compiler.native_sm120_tensor_program as exporter
import tessera.compiler.nvidia_native as native

def forbidden(*args, **kwargs):
    raise AssertionError("portable replay attempted compiler or Python member launch")
exporter.run_tessera_opt = forbidden
exporter.export_native_sm120_tensor_jvp = forbidden
native.package_scheduled_matmul = forbidden
native.package_scheduled_kernel = forbidden
rt._submit_nvidia_sm120_native = forbidden
folder = Path(sys.argv[1])
dtype = np.dtype(np.float16 if sys.argv[2] == "f16" else ml_dtypes.bfloat16)
with np.load(folder / "inputs.npz") as loaded:
    values = tuple(loaded[str(i)].view(dtype) for i in range(3))
owner = PreparedTensorJVP.from_manifest(json.loads((folder / "program.json").read_text()))
try:
    outputs = owner(values)
    artifact = rt.RuntimeArtifact.from_json((folder / "runtime.json").read_text())
    receipt = rt.launch(artifact, values)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    assert receipt["compiler_path"] == "nvidia_sm120_tensor_jvp_compiled"
    assert len(receipt["component_receipts"]) == 4
    for actual, expected in zip(receipt["output"], outputs, strict=True):
        np.testing.assert_array_equal(actual, expected)
    retained = tuple(v.copy() for v in outputs)
    owner(values)
    for value, saved in zip(outputs, retained, strict=True):
        np.testing.assert_array_equal(value, saved)
    assert not np.shares_memory(*outputs)
    np.savez(folder / "outputs.npz", primal=outputs[0], tangent=outputs[1])
finally:
    owner.close()
"""
    env = {**os.environ, "TESSERA_OPT": "/unavailable/compiler", "TESSERA_ROCM_OPT": "/unavailable/compiler"}
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), storage],
                            env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(tmp_path / "outputs.npz") as replay:
        for name, value in zip(("primal", "tangent"), expected, strict=True):
            np.testing.assert_array_equal(replay[name], value)


def test_paired_tensor_portable_rejects_resealed_lifetime_and_wrong_schema():
    import copy
    from tessera import runtime as rt
    from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP
    from tessera.compiler.portable_nvidia_tensor_jvp import digest
    from tessera.compiler.native_sm120_tensor_program import validate_native_tensor_plan
    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and production compiler")
    owner = PreparedTensorJVP(graph())
    try:
        data = owner.manifest()
    finally:
        owner.close()
    with pytest.raises(ValueError, match="schema"):
        validate_native_tensor_plan(data["plan_json"])
    for edit in ("lifetime", "role", "shape", "output", "member"):
        bad = copy.deepcopy(data)
        plan = json.loads(bad["plan_json"])
        if edit == "lifetime":
            plan["buffers"][4]["last_read"] = 2
        elif edit == "role":
            plan["role_indices"] = [0, 1, 1]
        elif edit == "shape":
            plan["buffers"][2]["shape"] = [32, 8]
        elif edit == "output":
            plan["outputs"] = [5, 5]
        else:
            bad["artifacts"][0]["graph_ir"] += "// changed"
        bad["plan_json"] = json.dumps(plan)
        bad["contract_digest"] = digest({k: v for k, v in bad.items() if k != "contract_digest"})
        with pytest.raises(ValueError):
            PreparedTensorJVP.from_manifest(bad)

def test_declared_paired_executor_is_registered():
    from tessera import runtime as rt
    assert "nvidia_tensor_rhs_jvp_program" in rt._executor_table()


def test_paired_public_artifact_rejects_parent_changes_before_provider(monkeypatch):
    import copy
    from dataclasses import replace
    from tessera import runtime as rt
    from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP
    from tessera.compiler.nvidia_tensor_jvp_runtime import runtime_artifact
    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and production compiler")
    owner = PreparedTensorJVP(graph())
    try:
        artifact = runtime_artifact(owner)
    finally:
        owner.close()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid public contract reached native provider")
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    for edit in ("graph", "names", "target"):
        metadata = copy.deepcopy(artifact.metadata)
        parent = artifact.graph_ir
        if edit == "graph":
            parent += "// changed"
        elif edit == "names":
            metadata["arg_names"] = ["primal_0", "primal_1", "tangent_0"]
        else:
            metadata["target"] = "rocm_gfx1201"
        bad = replace(artifact, graph_ir=parent, metadata=metadata)
        result = rt.launch(bad, ())
        assert result["ok"] is False, result
    result = rt.launch(artifact, (), stream=object())
    assert result["ok"] is False and "owns" in result["reason"]

def test_paired_public_certificate_cache_revalidates_changed_payload(monkeypatch):
    import copy
    from dataclasses import replace
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP
    from tessera.compiler import nvidia_tensor_jvp_runtime as binding
    from tessera.compiler.portable_nvidia_tensor_jvp import digest
    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning RTX5070 and production compiler")
    owner = PreparedTensorJVP(graph())
    values = (np.ones((7, 32), np.float16), np.ones((32, 9), np.float16), np.ones((32, 9), np.float16))
    calls = []
    original = binding.restore
    def counted(data):
        calls.append(1)
        return original(data)
    monkeypatch.setattr(binding, "restore", counted)
    with binding._lock:
        binding._certificates.clear()
    try:
        artifact = binding.runtime_artifact(owner)
        for _ in range(2):
            result = rt.launch(artifact, values)
            assert result["ok"], result
        assert len(calls) == 1
        for reseal in (False, True):
            metadata = copy.deepcopy(artifact.metadata)
            program = metadata["native_program"]
            plan = json.loads(program["plan_json"])
            plan["buffers"][4]["last_read"] = 2
            program["plan_json"] = json.dumps(plan)
            if reseal:
                program["contract_digest"] = digest({k: v for k, v in program.items() if k != "contract_digest"})
            result = rt.launch(replace(artifact, metadata=metadata), values)
            assert not result["ok"], result
        assert len(calls) == 3
        # Rejected certificates cannot evict or poison the valid warm owner.
        assert rt.launch(artifact, values)["ok"]
        assert len(calls) == 3
    finally:
        owner.close()
