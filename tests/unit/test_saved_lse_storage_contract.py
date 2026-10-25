"""Native semantic/Schedule/Tile storage contract for saved-LSE forward."""
import json
import re
import subprocess

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route


def module(storage="f16", output=None, lse="f32", key=None, bias=None,
           bias_storage="f32", target="nvidia_sm120", arch="sm_120", public=True, kv=None):
    output = output or storage
    tensors = [
        f"tensor<1x2x3x4x{storage}>", f"tensor<1x1x5x4x{key or kv or storage}>",
        f"tensor<1x1x5x3x{kv or storage}>"]
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
    ({"key": "f32"}, "K/V storage must agree"),
    ({"kv": "bf16", "output": "bf16"}, "output storage must be f32 or match Q"),
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

@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("bias", [None, "1x2x1x1", "1x2x3x5"])
def test_forward_only_wrapper_preserves_native_storage_identity(compiler, monkeypatch, storage, bias):
    from tessera.compiler import nvidia_native as native, scheduled_checkpoint as checkpoint
    graph = module(storage=storage, bias=bias, public=False)
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    forward = checkpoint._decode_checkpoint(graph, schedule, tile, backward=False)
    monkeypatch.setattr(checkpoint, "lower_generated_checkpoint", lambda source: forward)
    monkeypatch.setattr(native, "_compile_tile_ir",
        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    wrapper = native.package_generated_attention_forward_checkpoint(
        "native producer", pipeline_name="tessera-nvidia-pipeline-sm120")
    assert wrapper.contract_digest == wrapper.forward.descriptor.provenance["checkpoint_contract"]
    expected = native._checkpoint_identity(forward.dims, forward.scale, forward.causal,
        bias=forward.bias, bias_shape=forward.bias_shape, shape_bounds=forward.shape_bounds,
        storage=forward.storage, output_storage=forward.output_storage)
    assert wrapper.contract_digest == expected
    from tessera.compiler.resident_attention import checkpoint_shapes
    dims, shapes = checkpoint_shapes(wrapper)
    assert tuple(dims) == (1, 2, 1, 3, 5, 4, 3)
    assert tuple(shapes) == (
        (1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3),
        (1, 2, 3, 3), (1, 2, 3),
    )


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("bias", [None, "1x2x1x1", "1x2x3x5"])
def test_scheduled_ad_pair_rejects_half_forward_before_target_compilation(
    compiler, monkeypatch, storage, bias,
):
    from tessera.compiler import nvidia_native as native, scheduled_checkpoint as checkpoint
    graph = module(storage=storage, bias=bias, public=False)
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    forward = checkpoint._decode_checkpoint(graph, schedule, tile, backward=False)
    names = ("do", "q", "k", "v", "o", "bias", "lse", "dq", "dk", "dv") if bias else (
        "do", "q", "k", "v", "o", "lse", "dq", "dk", "dv")
    backward = checkpoint.lower_scheduled_checkpoint(names, forward.dims, forward.scale, True,
        backward=True, bias=bool(bias), bias_shape=tuple(map(int, bias.split("x"))) if bias else None)
    monkeypatch.setattr(native, "_compile_tile_ir",
        lambda *args: pytest.fail("compiled a mismatched half/f32 AD pair"))
    with pytest.raises(ValueError, match="paired checkpoint AD requires its f32 storage"):
        native.package_scheduled_checkpoint_pair(forward, backward, pipeline_name="test")


# SM120-SAVED-LSE-MIXED-STORAGE-20261025: K/V share one storage that may
# differ from Q on the forward; O is f32 or Q's storage; accumulation is f32.
MIXED = [("f16", "bf16"), ("bf16", "f16"), ("f32", "f16"), ("f32", "bf16"),
         ("f16", "f32"), ("bf16", "f32")]


@pytest.mark.parametrize("storage,kv", MIXED)
@pytest.mark.parametrize("bias", [None, "1x2x1x1", "1x2x3x5"])
@pytest.mark.parametrize("public", [True, False])
def test_mixed_forward_storage_reaches_tile_with_distinct_kv_storage(compiler, storage, kv, bias, public):
    tile = run_tessera_opt(compiler, run_tessera_opt(compiler,
        module(storage=storage, kv=kv, bias=bias, public=public), "--tessera-graph-to-schedule"),
        "--tessera-schedule-to-tile")
    contract = re.search(r"tessera.native_contract = \{([^\n]*)\}", tile)
    assert contract is not None
    assert f'kv_storage = "{kv}"' in contract[1] and f'storage = "{storage}"' in contract[1]
    assert 'accum = "f32"' in contract[1] and 'row_lse_storage = "f32"' in contract[1]
    entry = re.findall(r"llvm.func @([\w]+)\(", tile)
    suffix = "_" if storage == "f32" else f"_out_{storage}_"
    assert len(entry) == 1 and entry[0].startswith(
        f"tessera_tile_attention_lse_{storage}_kv_{kv}{suffix}")
    kernel = [line for line in tile.splitlines() if "tile.attention_kernel" in line]
    assert len(kernel) == 1 and f'kv_storage = "{kv}"' in kernel[0]


@pytest.mark.parametrize("storage,kv", MIXED)
def test_mixed_forward_with_f32_output_and_homogeneous_controls(compiler, storage, kv):
    tile = run_tessera_opt(compiler, run_tessera_opt(compiler,
        module(storage=storage, kv=kv, output="f32", public=False), "--tessera-graph-to-schedule"),
        "--tessera-schedule-to-tile")
    assert f"_kv_{kv}_" in tile and "_out_" not in re.findall(r"llvm.func @([\w]+)\(", tile)[0]
    homogeneous = run_tessera_opt(compiler, run_tessera_opt(compiler,
        module(storage=storage, public=False), "--tessera-graph-to-schedule"),
        "--tessera-schedule-to-tile")
    assert "kv_storage" not in homogeneous and "_kv_" not in homogeneous


@pytest.mark.parametrize("target,arch", [("rocm", "gfx1201"), ("x86", "zen5-avx512")])
def test_mixed_forward_storage_requires_the_sm120_contract(compiler, target, arch):
    process = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
        input=module(storage="f32", kv="f16", output="f32", public=False, target=target, arch=arch),
        capture_output=True, text=True)
    assert process.returncode == 1 and "SM120 native contract" in process.stderr


@pytest.mark.parametrize("storage,kv", MIXED)
def test_mixed_forward_descriptor_and_identity(compiler, monkeypatch, storage, kv):
    from tessera.compiler import nvidia_native as native, scheduled_checkpoint as checkpoint
    from tessera.compiler.typed_attention_checkpoint import validate_typed_saved_lse_descriptor
    graph = module(storage=storage, kv=kv, bias="1x2x1x1", public=False)
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    scheduled = checkpoint._decode_checkpoint(graph, schedule, tile, backward=False)
    assert scheduled.kv_storage == kv and scheduled.storage == storage
    monkeypatch.setattr(native, "_compile_tile_ir",
        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    package = native.package_scheduled_checkpoint(scheduled, pipeline_name="tessera-nvidia-pipeline-sm120")
    descriptor = package.descriptor
    dtype = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}
    buffers = sorted(descriptor.buffers, key=lambda value: value.ordinal)
    assert [b.dtype for b in buffers[:3]] == [dtype[storage], dtype[kv], dtype[kv]]
    assert descriptor.abi_id in native.SM120_ATTN_LSE_TYPED_ABIS
    assert validate_typed_saved_lse_descriptor(descriptor) == (storage, storage)
    homogeneous = native._checkpoint_identity(scheduled.dims, scheduled.scale, scheduled.causal,
        bias=True, storage=storage, output_storage=storage)
    assert descriptor.provenance["checkpoint_contract"] != homogeneous
    from dataclasses import replace
    for provenance in ({**descriptor.provenance, "kv_storage": storage},
                       {key: value for key, value in descriptor.provenance.items() if key != "kv_storage"}):
        with pytest.raises(ValueError):
            validate_typed_saved_lse_descriptor(replace(descriptor, provenance=provenance))
    changed = list(descriptor.buffers)
    changed[2] = replace(changed[2], dtype=dtype[storage])
    with pytest.raises(ValueError):
        validate_typed_saved_lse_descriptor(replace(descriptor, buffers=tuple(changed)))


def test_homogeneous_identity_is_unchanged_by_the_kv_field():
    from tessera.compiler.nvidia_native import _checkpoint_identity
    dims = (1, 2, 1, 3, 5, 4, 3)
    for storage in ("f16", "bf16", "f32"):
        assert _checkpoint_identity(dims, 0.5, True, storage=storage, output_storage=storage) == \
            _checkpoint_identity(dims, 0.5, True, storage=storage, output_storage=storage,
                                 kv_storage=storage)


def test_mixed_forward_is_refused_by_the_paired_ad_contract(compiler):
    from tessera.compiler import nvidia_native as native, scheduled_checkpoint as checkpoint
    graph = module(storage="f16", kv="bf16", public=False)
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    forward = checkpoint._decode_checkpoint(graph, schedule, tile, backward=False)
    backward = checkpoint.lower_scheduled_checkpoint(
        ("do", "q", "k", "v", "o", "lse", "dq", "dk", "dv"), forward.dims, forward.scale, True,
        backward=True)
    with pytest.raises(ValueError):
        native.package_scheduled_checkpoint_pair(forward, backward, pipeline_name="test")
