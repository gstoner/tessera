"""Native row-batch contract: logical shape remains in Graph, flattened in Schedule."""
import copy
import pytest
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type, _infer_result_types
from tessera.compiler import nvidia_native, scheduled_matmul


def batch_module(batch=3, rows=7, n=5, k=31):
    sk = (k + 15) // 16
    types = [tensor_ir_type((batch, rows, k), "nvfp4"), tensor_ir_type((k, n), "nvfp4"),
             tensor_ir_type((batch, rows, sk), "uint8"), tensor_ir_type((sk, n), "uint8")]
    attrs = {"physical_contract": "nvidia_sm120_nvfp4_blockscale_v1", "batching": "shared_rhs_rows",
             "scale_layout": {"granularity": "block", "block": [1, 16], "format": "ue4m3"},
             "numeric_policy": {"accum": "fp32", "execution_mode": "exact_per_block"}}
    out = _infer_result_types("tessera.scaled_matmul", types, attrs)[0]
    names = ("a", "b", "sa", "sb")
    return GraphIRModule(functions=[GraphIRFunction(
        name="shared_rhs_batch", args=[IRArg(name, ty) for name, ty in zip(names, types)],
        result_types=[out], body=[IROp(result="d", op_name="tessera.scaled_matmul",
            operands=["%" + name for name in names], operand_types=list(map(str, types)),
            result_type=str(out), kwargs=attrs)], return_values=["%d"])])


@pytest.mark.parametrize("batch,rows,n,k", [(1, 1, 1, 1), (3, 7, 5, 31), (2, 17, 19, 129)])
def test_shared_rhs_batch_frontend_contract_preserves_logical_axes(batch, rows, n, k):
    module = batch_module(batch, rows, n, k)
    assert module.functions[0].result_types[0].shape == tuple(map(str, (batch, rows, n)))
    assert nvidia_native.supports_nvfp4_matmul(module)
    assert scheduled_matmul._graph_contract(module, "nvidia_sm120")[6:9] == (batch * rows, n, k)


@pytest.mark.parametrize("edit", ["scale_batch", "scale_k", "output_batch", "rhs_batch", "missing_policy"])
def test_shared_rhs_batch_admission_checks_logical_roles(edit):
    module = batch_module()
    fn = module.functions[0]
    if edit == "scale_batch": fn.args[2].ir_type = tensor_ir_type((2, 7, 2), "uint8")
    elif edit == "scale_k": fn.args[2].ir_type = tensor_ir_type((3, 7, 3), "uint8")
    elif edit == "output_batch": fn.result_types[0] = tensor_ir_type((2, 7, 5), "fp32")
    elif edit == "rhs_batch": fn.args[1].ir_type = tensor_ir_type((3, 31, 5), "nvfp4")
    else: fn.body[0].kwargs.pop("batching")
    assert not nvidia_native.supports_nvfp4_matmul(module)


@pytest.mark.parametrize("batch,rows,n,k", [(3, 7, 5, 31), (2, 17, 19, 129)])
def test_shared_rhs_batch_native_schedule_owns_zero_copy_row_projection(batch, rows, n, k):
    if scheduled_matmul.find_tessera_opt() is None:
        pytest.skip("requires matching native compiler")
    module = batch_module(batch, rows, n, k)
    original = copy.deepcopy(module)
    artifact = scheduled_matmul.lower_scheduled_matmul(module, target="nvidia_sm120")
    assert (artifact.m, artifact.n, artifact.k) == (batch * rows, n, k)
    assert f"tensor<{batch}x{rows}x{k}x!tessera.nvfp4>" in artifact.graph_ir
    assert "shared_rhs_rows" in artifact.schedule_ir
    assert "tile.matmul_kernel" in artifact.tile_ir
    assert "tessera.scale_vector_size = 16" in artifact.tile_ir
    assert "tessera.storage_pack" in artifact.tile_ir
    assert "tessera.scaled_matmul" not in artifact.tile_ir
    assert module == original


def test_shared_schedule_cannot_rebind_equal_flat_rows_to_different_logical_batches(monkeypatch):
    if scheduled_matmul.find_tessera_opt() is None:
        pytest.skip("requires matching native compiler")
    artifact = scheduled_matmul.lower_scheduled_matmul(batch_module(3, 7), target="nvidia_sm120")
    monkeypatch.setattr(nvidia_native, "_compile_tile_ir", lambda *args: pytest.fail("stale logical batch reached native packaging"))
    with pytest.raises(ValueError, match="different logical Graph"):
        nvidia_native.package_nvfp4_matmul(batch_module(7, 3), pipeline_name="native_batch", scheduled_artifact=artifact)


@pytest.mark.parametrize("batch,rows,k,valid", [
    (1, 1, 9223372036854775807, True),
    (9223372036854775807, 2, 31, False),
    (1, 0, 31, False),
    (0, 1, 31, False),
    (1, 1, 0, False),
])
def test_native_batch_verifier_handles_static_dimension_boundaries(batch, rows, k, valid):
    """Graph verification must not overflow before native launch limits apply."""
    import subprocess

    compiler = scheduled_matmul.find_tessera_opt()
    if compiler is None:
        pytest.skip("requires matching native compiler")
    scale_k = k // 16 + (k % 16 != 0)
    a = f"tensor<{batch}x{rows}x{k}x!tessera.nvfp4>"
    b = f"tensor<{k}x5x!tessera.nvfp4>"
    sa = f"tensor<{batch}x{rows}x{scale_k}xui8>"
    sb = f"tensor<{scale_k}x5xui8>"
    d = f"tensor<{batch}x{rows}x5xf32>"
    source = f"""module {{
      func.func @batch(%a: {a}, %b: {b}, %sa: {sa}, %sb: {sb}) -> {d} {{
        %d = tessera.scaled_matmul %a, %b scales (%sa, %sb) {{
          batching = "shared_rhs_rows",
          physical_contract = "nvidia_sm120_nvfp4_blockscale_v1",
          scale_layout = {{granularity = "block", block = [1, 16], format = "ue4m3"}},
          numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}}
        }} : ({a}, {b}, {sa}, {sb}) -> {d}
        return %d : {d}
      }}
    }}"""
    process = subprocess.run([str(compiler)], input=source, capture_output=True, text=True)
    if valid:
        assert process.returncode == 0, process.stderr
    else:
        assert process.returncode != 0
        assert "NVIDIA NVFP4 contract requires" in process.stderr


def frontend_batch_module(batch=3, rows=7, n=5, k=31, batching="shared_rhs_rows"):
    """Use public Python syntax and typed annotations, without hand-built ops."""
    from tessera.compiler.graph_ir import GraphIRBuilder
    scale_k = k // 16 + (k % 16 != 0)
    independent_rhs = batching == "independent_rhs"
    b_annotation = f"tensor<{batch}x{k}x{n}x!tessera.nvfp4>" if independent_rhs else f"tensor<{k}x{n}x!tessera.nvfp4>"
    sb_annotation = f"tensor<{batch}x{scale_k}x{n}xui8>" if independent_rhs else f"tensor<{scale_k}x{n}xui8>"
    source = f'''def shared_rhs_batch(
    a: "tensor<{batch}x{rows}x{k}x!tessera.nvfp4>",
    b: "{b_annotation}",
    sa: "tensor<{batch}x{rows}x{scale_k}xui8>",
    sb: "{sb_annotation}"):
    return ts.ops.scaled_matmul(a, b, sa, sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={{"accum": "fp32", "execution_mode": "exact_per_block"}},
        scale_layout={{"granularity": "block", "block": [1, 16], "format": "ue4m3"}},
        batching="{batching}", transposeA=False, transposeB=False)
'''
    def shared_rhs_batch(a, b, sa, sb):
        raise AssertionError("typed frontend must consume source without eager execution")
    shared_rhs_batch.__annotations__ = {
        "a": f"tensor<{batch}x{rows}x{k}x!tessera.nvfp4>",
        "b": b_annotation,
        "sa": f"tensor<{batch}x{rows}x{scale_k}xui8>",
        "sb": sb_annotation,
    }
    builder = GraphIRBuilder()
    builder.lower(shared_rhs_batch, source_text=source,
                  source_origin="W1.1 Python shared-RHS batch")
    module = builder.module()
    assert not builder.diagnostics, builder.diagnostics
    return module


@pytest.mark.parametrize("batch,rows,n,k", [(3, 7, 5, 31), (2, 17, 19, 129)])
def test_python_frontend_preserves_shared_batch_operands_and_policies(batch, rows, n, k):
    module = frontend_batch_module(batch, rows, n, k)
    fn = module.functions[0]
    assert len(fn.body) == 1
    op = fn.body[0]
    assert op.op_name == "tessera.scaled_matmul"
    assert op.operands == ["%a", "%b", "%sa", "%sb"]
    assert op.kwargs["batching"] == "shared_rhs_rows"
    assert op.kwargs["scale_layout"]["block"] == [1, 16]
    assert str(fn.result_types[0]) == f"tensor<{batch}x{rows}x{n}xf32>"
    assert nvidia_native.supports_nvfp4_matmul(module)


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_mlir_unsigned_annotations_use_canonical_storage_names(bits):
    from tessera.compiler.graph_ir import _parse_mlir_tensor_type

    value = _parse_mlir_tensor_type(f"tensor<2x3xui{bits}>")
    assert value.dtype == f"uint{bits}"
    assert str(value) == f"tensor<2x3xui{bits}>"


@pytest.mark.parametrize("batch,rows,n,k", [(3, 7, 5, 31), (2, 17, 19, 129)])
def test_independent_rhs_frontend_preserves_batched_scale_roles(batch, rows, n, k):
    module = frontend_batch_module(batch, rows, n, k, "independent_rhs")
    fn = module.functions[0]
    assert fn.args[1].ir_type.shape == tuple(map(str, (batch, k, n)))
    assert fn.args[3].ir_type.shape == tuple(map(str, (batch, (k + 15) // 16, n)))
    assert nvidia_native.supports_nvfp4_matmul(module)
    assert scheduled_matmul._graph_contract(module, "nvidia_sm120")[6:9] == (batch * rows, n, k)


@pytest.mark.parametrize("edit", ["rhs_batch", "scale_batch", "scale_k", "result_batch"])
def test_independent_rhs_admission_rejects_mismatched_logical_axes(edit):
    module = frontend_batch_module(batching="independent_rhs")
    fn = module.functions[0]
    if edit == "rhs_batch": fn.args[1].ir_type = tensor_ir_type((2, 31, 5), "nvfp4")
    elif edit == "scale_batch": fn.args[3].ir_type = tensor_ir_type((2, 2, 5), "uint8")
    elif edit == "scale_k": fn.args[3].ir_type = tensor_ir_type((3, 3, 5), "uint8")
    else: fn.result_types[0] = tensor_ir_type((2, 7, 5), "fp32")
    assert not nvidia_native.supports_nvfp4_matmul(module)
    with pytest.raises(ValueError):
        scheduled_matmul._graph_contract(module, "nvidia_sm120")


@pytest.mark.parametrize("field,extra", [
    ("numeric_policy", {"storage": "bf16"}),
    ("numeric_policy", {"rounding": "toward_zero"}),
    ("scale_layout", {"axis": "N"}),
])
def test_direct_native_nvfp4_preserves_exact_named_policy(field, extra):
    import subprocess

    compiler = scheduled_matmul.find_tessera_opt()
    if compiler is None:
        pytest.skip("requires matching native compiler")
    module = batch_module()
    contract = scheduled_matmul._graph_contract(module, "nvidia_sm120")
    module.module_attrs["tessera.target"] = f"\"{contract[0]}\""
    module.module_attrs["tessera.arch"] = f"\"{contract[1]}\""
    canonical = module.to_mlir(verify=False, canonical=True, target="nvidia_sm120")
    control = subprocess.run([str(compiler), "--tessera-graph-to-schedule"], input=canonical,
                             capture_output=True, text=True)
    assert control.returncode == 0, control.stderr
    module.functions[0].body[0].kwargs[field].update(extra)
    authored = module.to_mlir(verify=False, canonical=True, target="nvidia_sm120")
    process = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
                             input=authored, capture_output=True, text=True)
    assert process.returncode != 0, "native import silently dropped requested policy"
    assert "NVIDIA NVFP4 contract requires" in process.stderr
