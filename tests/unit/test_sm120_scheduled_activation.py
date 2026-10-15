"""W1.1 standalone ReLU must retain its own checked Schedule/Tile store."""
from __future__ import annotations

import json

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


@pytest.fixture
def tool():
    compiler = find_tessera_opt()
    if compiler is None:
        pytest.skip("requires the production MLIR compiler")
    return compiler


def _graph(storage="f16", arch="sm_120"):
    return f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "{arch}"}} {{
  func.func @relu_member(%x: tensor<7x32x{storage}>) -> tensor<7x32x{storage}> {{
    %y = "tessera.relu"(%x) : (tensor<7x32x{storage}>) -> tensor<7x32x{storage}>
    return %y : tensor<7x32x{storage}>
  }}
}}'''


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_relu_has_explicit_native_schedule_and_storage_boundary(tool, storage):
    schedule = run_tessera_opt(tool, _graph(storage), "--tessera-graph-to-schedule")
    assert "schedule.activation" in schedule
    assert 'kind = "relu"' in schedule
    assert 'nan_mode = "propagate"' in schedule
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    assert "tile.activation_kernel" in tile
    assert f'storage = "{storage}"' in tile
    assert "!llvm.ptr, !llvm.ptr, i64, i64" in tile
    assert "tessera.relu" not in tile
    assert "schedule." not in tile
    assert "norm_epsilon" not in tile


@pytest.mark.parametrize("old,new", [
    ('kind = "relu"', 'kind = "silu"'),
    ('nan_mode = "propagate"', 'nan_mode = "ignore"'),
    ('workgroup_size = 128 : i64', 'workgroup_size = 64 : i64'),
    ('schedule = "serial"', 'schedule = "cooperative_128"'),
    ('zero_mode = "positive"', 'zero_mode = "preserve"'),
])
def test_relu_schedule_rejects_policy_mutation(tool, old, new):
    schedule = run_tessera_opt(tool, _graph(), "--tessera-graph-to-schedule")
    assert old in schedule
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool, schedule.replace(old, new), "--tessera-schedule-to-tile")


def test_relu_schedule_does_not_promote_f32_or_sibling_architecture(tool):
    for graph in (_graph("f32"), _graph(arch="gfx1201")):
        with pytest.raises(RuntimeError):
            run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")


@pytest.mark.compiler_route
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_norm_relu_matmul_native_export_retains_two_private_stores(tool, storage):
    from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
    source = f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @norm_relu(%x: tensor<7x32x{storage}>, %w: tensor<32x9x{storage}>) -> tensor<7x9xf32> {{
    %norm = "tessera.rmsnorm"(%x) {{eps = 1.0E-5 : f64}} : (tensor<7x32x{storage}>) -> tensor<7x32x{storage}>
    %edge = "tessera.relu"(%norm) : (tensor<7x32x{storage}>) -> tensor<7x32x{storage}>
    %out = "tessera.matmul"(%edge, %w) {{rhs_storage_order = "row_major"}} : (tensor<7x32x{storage}>, tensor<32x9x{storage}>) -> tensor<7x9xf32>
    return %out : tensor<7x9xf32>
  }}
}}'''
    native = export_native_sm120_tensor_graph(source, tool=tool)
    plan = native.manifest
    assert [step["operation"] for step in plan["steps"]] == [
        "tessera.rmsnorm", "tessera.relu", "tessera.matmul"]
    assert [step["inputs"] for step in plan["steps"]] == [[0], [2], [3, 1]]
    for buffer_id, write, last_read in ((2, 0, 1), (3, 1, 2)):
        buffer = plan["buffers"][buffer_id]
        assert buffer["ownership"] == "private_scratch"
        assert buffer["storage"] == storage
        assert buffer["bytes"] == 7 * 32 * 2
        assert (buffer["first_write"], buffer["last_read"]) == (write, last_read)
    norm, activation, matmul = native.scheduled_members()
    assert norm.kind == "rmsnorm"
    assert activation.family == "activation" and activation.kind == "relu"
    assert activation.epsilon == 0.0
    assert "tile.view" in matmul.tile_ir
    assert "tile.fragment_pack" in matmul.tile_ir


@pytest.fixture
def sm120_device(tool):
    from tessera import runtime as rt
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires exact SM120 device execution")
    return rt


def _frontend_program(dtype, shape):
    import numpy as np
    import tessera as ts
    from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
    owner = ts.from_text("""def norm_relu(a, b):
    normalized = tessera.ops.rmsnorm(a, eps=1e-5)
    edge = tessera.ops.relu(normalized)
    return tessera.ops.matmul(edge, b, output_dtype="fp32")
""", target="nvidia_sm120")
    m, k, n = shape
    rng = np.random.default_rng(47017)
    a = rng.uniform(-1, 1, (m, k)).astype(dtype)
    b = rng.uniform(-0.25, 0.25, (k, n)).astype(dtype)
    module, _ = owner._trace_frontend_capture((a, b), {})
    return package_traced_lhs(module), (a, b), owner


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_native_relu_preserves_nonfinite_and_signed_zero_storage(sm120_device, storage):
    import numpy as np
    import ml_dtypes
    from tessera.compiler import nvidia_native
    from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    program, _, _ = _frontend_program(dtype, (7, 32, 9))
    native = export_native_sm120_tensor_graph(program.graph_ir)
    artifact = native.scheduled_members()[1]
    package = nvidia_native.package_scheduled_kernel(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    # Both signs of zero, subnormal/normal boundaries, largest finite values,
    # infinities and distinct quiet-NaN payloads exercise storage semantics.
    patterns = ([0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF,
                 0x0400, 0x8400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
                 0x7E11, 0xFE55] if storage == "f16" else
                [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F,
                 0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80,
                 0x7FC1, 0xFFE5])
    source = np.resize(np.array(patterns, dtype=np.uint16), (7, 32)).view(dtype)
    output = np.empty_like(source)
    expected = np.maximum(0, source)
    sm120_device._submit_nvidia_sm120_native(package.image, package.descriptor,
        {artifact.input_name: source, artifact.output_name: output},
        {"Rows": 7, "Columns": 32}, None)
    np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(7, 32, 9), (17, 64, 19), (129, 256, 33)])
def test_frontend_norm_relu_matmul_prepared_numerics_and_retained_outputs(sm120_device, storage, shape):
    import numpy as np
    import ml_dtypes
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    program, (a, b), public = _frontend_program(dtype, shape)
    assert [step["operation"] for step in json.loads(program.native_plan_json)["steps"]] == [
        "tessera.rmsnorm", "tessera.relu", "tessera.matmul"]
    original = (a.copy(), b.copy())
    owner = PreparedLhsCall(program)
    try:
        outputs = []
        for factor in (1.0, 0.5):
            values = (a, (b.astype(np.float32) * factor).astype(dtype))
            wide = values[0].astype(np.float32)
            normalized = (wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + np.float32(1e-5))).astype(dtype)
            edge = np.maximum(0, normalized).astype(np.float32)
            expected = edge @ values[1].astype(np.float32)
            output, receipt = owner(values)
            assert receipt["execution_kind"] == "native_gpu"
            assert len(receipt["component_receipts"]) == 3
            np.testing.assert_allclose(output, expected, atol=0.02 if storage == "bf16" else 0.003, rtol=0.02 if storage == "bf16" else 0.003)
            public_output = public(*values)
            assert public.execution_kind == "native_gpu"
            artifact = public.runtime_artifact()
            assert artifact.metadata["native_program"]["schema"] == "tessera.nvidia.lhs_tensor_program.v3"
            np.testing.assert_allclose(public_output, expected,
                atol=0.02 if storage == "bf16" else 0.003,
                rtol=0.02 if storage == "bf16" else 0.003)
            outputs.append((output, output.copy()))
        for output, retained in outputs:
            np.testing.assert_array_equal(output, retained)
        for actual, before in zip((a, b), original, strict=True):
            np.testing.assert_array_equal(actual, before)
    finally:
        owner.close()


@pytest.mark.compiler_route
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_relu_only_producer_selects_native_tensor_export(tool, storage):
    from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
    source = f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @relu_matmul(%x: tensor<7x32x{storage}>, %w: tensor<32x9x{storage}>) -> tensor<7x9xf32> {{
    %edge = "tessera.relu"(%x) : (tensor<7x32x{storage}>) -> tensor<7x32x{storage}>
    %out = "tessera.matmul"(%edge, %w) {{rhs_storage_order = "row_major"}} : (tensor<7x32x{storage}>, tensor<32x9x{storage}>) -> tensor<7x9xf32>
    return %out : tensor<7x9xf32>
  }}
}}'''
    native = export_native_sm120_tensor_graph(source, tool=tool)
    assert [step["operation"] for step in native.manifest["steps"]] == ["tessera.relu", "tessera.matmul"]
    activation, matmul = native.scheduled_members()
    assert activation.kind == "relu"
    assert "tile.view" in matmul.tile_ir


@pytest.mark.compiler_route
def test_post_matmul_relu_keeps_its_own_export_route(tool):
    source = '''module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @post_relu(%x: tensor<7x32xf16>, %w: tensor<32x9xf16>) -> tensor<7x9xf32> {
    %product = "tessera.matmul"(%x, %w) : (tensor<7x32xf16>, tensor<32x9xf16>) -> tensor<7x9xf32>
    %out = "tessera.relu"(%product) : (tensor<7x9xf32>) -> tensor<7x9xf32>
    return %out : tensor<7x9xf32>
  }
}'''
    # Its existing scaled-program exporter may refuse this separate envelope.
    # The upstream tensor-producer exporter must not intercept it.
    try:
        result = run_tessera_opt(tool, source, "--tessera-autodiff-forward=export-scaled-primal=true")
    except RuntimeError as exc:
        assert "SM120 tensor" not in str(exc)
    else:
        assert "tessera.native.sm120_tensor_program" not in result


@pytest.mark.parametrize("field,value", [
    ("zero_mode", "preserve"), ("nan_mode", "ignore"),
    ("schedule", "cooperative_128"), ("storage", "bf16"),
    ("schedule_digest", "not-a-digest"),
    ("storage", {"malformed": True}), ("storage", None),
])
def test_composed_relu_rejects_forged_numerical_descriptor(tool, field, value):
    import numpy as np
    from dataclasses import replace
    from tessera.compiler import nvidia_native
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    if not nvidia_native.tools_available():
        pytest.skip("requires matching native NVIDIA assembly tools")
    program, _, _ = _frontend_program(np.float16, (7, 32, 9))
    producer = program.edge.producer
    forged = replace(producer, descriptor=replace(producer.descriptor,
        provenance={**producer.descriptor.provenance, field: value}))
    broken = replace(program, edge=replace(program.edge, producer=forged),
        producer_chain=(*program.producer_chain[:-1], forged))
    with pytest.raises(ValueError, match="numerical ABI mismatch"):
        broken.validate()
    # Validation must reject before native owner allocation or attachment.
    with pytest.raises(ValueError, match="numerical ABI mismatch"):
        PreparedLhsCall(broken)


def test_relu_producer_execution_registration_keeps_epilogue_proof_separate():
    from tessera.compiler import backend_manifest as bm, conformance_matrix as cm
    from tessera.compiler.execution_matrix import lookup

    fixture = "tests/unit/test_sm120_scheduled_activation.py"
    row = next(item for item in bm.manifest_for("relu") if item.target == "nvidia_sm120")
    assert "scheduled_activation" in row.feature_flags
    assert row.execute_compare_fixture == fixture
    assert bm._NUMERICAL_FIXTURES[("matmul_relu", "nvidia_sm120")] == (
        "tests/device/nvidia/test_matmul_relu.py")
    execution = lookup("nvidia_sm120", "canonical_nvidia_lhs_program")
    assert execution.executable and execution.executor_id == "nvidia_lhs_program"
    assert execution.numerical_fixture == fixture and execution.evidence_target == "nvidia_sm120"
    assert cm._numerical_check_present("relu", "nvidia_sm120")
    for sibling in ("nvidia_sm80", "nvidia_sm90", "nvidia_sm100"):
        assert not cm._numerical_check_present("relu", sibling)
