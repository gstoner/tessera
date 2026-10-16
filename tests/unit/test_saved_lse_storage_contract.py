"""Native semantic/Schedule/Tile storage contract for saved-LSE forward."""
import json
import re
import subprocess

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route


def module(storage="f16", output=None, lse="f32", key=None, bias=None,
           bias_storage="f32", target="nvidia_sm120", arch="sm_120", public=True):
    output = output or storage
    tensors = [
        f"tensor<1x2x3x4x{storage}>", f"tensor<1x1x5x4x{key or storage}>",
        f"tensor<1x1x5x3x{storage}>"]
    names = ["q", "k", "v"]
    if bias:
        tensors.append(f"tensor<{bias}x{bias_storage}>")
        names.append("bias")
    results = [f"tensor<1x2x3x3x{output}>", f"tensor<1x2x3x{lse}>"]
    arguments = ", ".join(f"%{name}: {typ}" for name, typ in zip(names, tensors, strict=True))
    operands = ", ".join("%"+name for name in names)
    attributes = 'lse_checkpoint = "saved", head_dim = 4 : i64, causal = true' if public else (
        "scale = 0.5 : f32, causal = true")
    if public:
        attributes += ", operandSegmentSizes = array<i32: 1, 1, 1, " + str(int(bool(bias))) + ">"
    op = "tessera.flash_attn" if public else "tessera_attn.checkpoint_forward"
    return f'''module attributes {{tessera.target = "{target}", tessera.arch = "{arch}"}} {{
 func.func @saved({arguments}) -> ({", ".join(results)})
 attributes {{tessera.argument_bindings = {json.dumps(names)}, tessera.result_bindings = ["o", "lse"]}} {{
  %o, %lse = "{op}"({operands}) {{{attributes}}} : ({", ".join(tensors)}) -> ({", ".join(results)})
  return %o, %lse : {", ".join(results)}
 }}
}}'''


@pytest.fixture
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("production tessera-opt required")
    return tool


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("bias", [None, "1x2x1x1", "1x2x3x5"])
def test_authored_saved_graph_preserves_primary_storage_and_f32_lse(compiler, storage, bias):
    schedule = run_tessera_opt(compiler, module(storage=storage, bias=bias), "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    assert f"storage = \"{storage}\"" in tile
    assert 'lse_checkpoint = "saved"' in tile and 'accum = "f32"' in tile
    contract = re.search(r"tessera.native_contract = \{([^\n]*)\}", tile)
    assert contract is not None
    if storage != "f32":
        assert 'row_lse_storage = "f32"' in contract[1]
        assert f'output_storage = "{storage}"' in contract[1]
        assert f"_out_{storage}_" in tile
    else:
        assert 'row_lse_storage' not in contract[1]


@pytest.mark.parametrize("storage,output", [
    ("f16", "f16"), ("f16", "f32"), ("bf16", "bf16"), ("bf16", "f32"), ("f32", "f32")])
@pytest.mark.parametrize("bias", [None, "1x2x1x1", "1x2x3x5"])
def test_registered_checkpoint_forward_has_independent_output_and_auxiliary_types(
    compiler, storage, output, bias,
):
    schedule = run_tessera_opt(compiler, module(storage=storage, output=output, bias=bias, public=False),
                              "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    assert f'storage = "{storage}"' in tile and 'lse_checkpoint = "saved"' in tile
    if output != "f32":
        assert f'output_storage = "{output}"' in tile


@pytest.mark.parametrize("edits,reason", [
    ({"lse": "f16"}, "row LSE storage"),
    ({"key": "f32"}, "Q/K/V storage"),
    ({"output": "bf16"}, "output storage"),
    ({"bias": "1x2x1x1", "bias_storage": "f16"}, "auxiliary and backward"),
    ({"target": "rocm", "arch": "gfx1201"}, "SM120 native contract"),
    ({"target": "apple_gpu", "arch": "apple7"}, "SM120 native contract"),
    ({"target": "x86", "arch": "zen5-avx512"}, "SM120 native contract"),
    ({"storage": "f64", "output": "f64"}, "f16/bf16/f32"),
])
def test_checkpoint_storage_mismatch_is_rejected_in_native_verification(compiler, edits, reason):
    process = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
                             input=module(public=False, **edits), capture_output=True, text=True)
    assert process.returncode == 1, process.stdout + process.stderr
    assert reason in process.stderr


@pytest.fixture
def typed_package(compiler, monkeypatch):
    from tessera.compiler import nvidia_native as native
    from tessera.compiler import scheduled_checkpoint as checkpoint
    graph = module(storage="f16", bias="1x2x1x1", public=False)
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    scheduled = checkpoint._decode_checkpoint(graph, schedule, tile, backward=False)
    monkeypatch.setattr(native, "_compile_tile_ir",
        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    return native.package_scheduled_checkpoint(scheduled, pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("field,value", [
    ("storage", "f32"), ("output", "bf16"), ("row_lse", "f16[B,Hq,Sq]"),
    ("accum", "f16"), ("bias", False), ("checkpoint_role", "backward"),
    ("checkpoint_contract", "0"*64), ("schedule_digest", "0"*64),
    ("scale", 0.0), ("shape", [1, 2, 1, 0, 5, 4, 3]),
])
def test_typed_descriptor_policy_tampering_is_rejected(typed_package, field, value):
    from dataclasses import replace
    from tessera.compiler.typed_attention_checkpoint import validate_typed_saved_lse_descriptor
    descriptor = typed_package.descriptor
    validate_typed_saved_lse_descriptor(descriptor)
    altered = replace(descriptor, provenance={**descriptor.provenance, field: value})
    with pytest.raises(ValueError):
        validate_typed_saved_lse_descriptor(altered)


@pytest.mark.parametrize("index,changes", [
    (0, {"dtype": "fp32"}), (2, {"alignment": 1}),
    (3, {"dtype": "fp16"}), (4, {"dtype": "fp32"}),
    (5, {"dtype": "fp16"}), (5, {"rank": 4}),
])
def test_typed_descriptor_buffer_roles_are_checked(typed_package, index, changes):
    from dataclasses import replace
    from tessera.compiler.typed_attention_checkpoint import validate_typed_saved_lse_descriptor
    descriptor = typed_package.descriptor
    buffers = list(descriptor.buffers)
    buffers[index] = replace(buffers[index], **changes)
    with pytest.raises(ValueError):
        validate_typed_saved_lse_descriptor(replace(descriptor, buffers=tuple(buffers)))
