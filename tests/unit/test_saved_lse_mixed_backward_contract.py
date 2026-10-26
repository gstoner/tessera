"""SM120-SAVED-LSE-MIXED-BACKWARD-20261026: mixed Q vs K/V saved-LSE reverse."""
import re
import subprocess

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route


def backward_module(q="f16", kv="bf16", output=None, target="nvidia_sm120", arch="sm_120",
                    key=None, gradient_k=None):
    output = output or q
    types = {
        "do": f"tensor<1x2x3x3x{output}>", "q": f"tensor<1x2x3x4x{q}>",
        "k": f"tensor<1x1x5x4x{key or kv}>", "v": f"tensor<1x1x5x3x{kv}>",
        "o": f"tensor<1x2x3x3x{output}>", "lse": "tensor<1x2x3xf32>",
    }
    results = [f"tensor<1x2x3x4x{q}>", f"tensor<1x1x5x4x{gradient_k or kv}>", f"tensor<1x1x5x3x{kv}>"]
    names = list(types)
    arguments = ", ".join(f"%{n}: {t}" for n, t in types.items())
    operand_types = ", ".join(types.values())
    return f'''module attributes {{tessera.target = "{target}", tessera.arch = "{arch}"}} {{
 func.func @bwd({arguments}) -> ({", ".join(results)})
 attributes {{tessera.argument_bindings = {names!r}, tessera.result_bindings = ["dq", "dk", "dv"]}} {{
  %dq, %dk, %dv = "tessera_attn.checkpoint_backward"({", ".join("%" + n for n in names)}) {{scale = 0.5 : f32, causal = true}} : ({operand_types}) -> ({", ".join(results)})
  return %dq, %dk, %dv : {", ".join(results)}
 }}
}}'''.replace("'", '"')


@pytest.fixture
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("production tessera-opt required")
    return tool


MIXED = [("f16", "bf16"), ("bf16", "f16"), ("f32", "f16"), ("f32", "bf16"), ("f16", "f32"), ("bf16", "f32")]


@pytest.mark.parametrize("q,kv", MIXED)
@pytest.mark.parametrize("output", [None, "f32"])
def test_mixed_backward_checkpoint_types_gradients_per_role(compiler, q, kv, output):
    tile = run_tessera_opt(compiler, run_tessera_opt(compiler,
        backward_module(q=q, kv=kv, output=output), "--tessera-graph-to-schedule"),
        "--tessera-schedule-to-tile")
    contract = re.search(r"tessera.native_contract = \{([^\n]*)\}", tile)
    assert contract is not None
    assert f'kv_storage = "{kv}"' in contract[1] and f'gradient_storage = "{q}"' in contract[1]
    assert 'row_correction = "probability_dot_v_f32_v1"' in contract[1]
    entry = re.findall(r"llvm.func @([\w]+)\(", tile)
    assert len(entry) == 1 and f"{q}_kv_{kv}_out_{output or q}_" in entry[0]
    kernel = [line for line in tile.splitlines() if "tile.attention_backward_kernel" in line]
    assert len(kernel) == 1 and f'kv_storage = "{kv}"' in kernel[0]


@pytest.mark.parametrize("q", ["f16", "bf16", "f32"])
def test_homogeneous_backward_names_no_kv_storage(compiler, q):
    tile = run_tessera_opt(compiler, run_tessera_opt(compiler,
        backward_module(q=q, kv=q), "--tessera-graph-to-schedule"), "--tessera-schedule-to-tile")
    assert "kv_storage" not in tile and "_kv_" not in tile


@pytest.mark.parametrize("edits,reason", [
    ({"kv": "f8E4M3FN"}, "FP8 only for forward K/V"),
    ({"key": "f16", "kv": "bf16"}, "K/V storage must agree"),
    ({"gradient_k": "f16", "kv": "bf16"}, "gradient/output/f32 auxiliary role"),
    ({"target": "rocm", "arch": "gfx1201"}, "SM120 native contract"),
])
def test_mixed_backward_refusals(compiler, edits, reason):
    process = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
                             input=backward_module(**edits), capture_output=True, text=True)
    assert process.returncode == 1, process.stdout + process.stderr
    assert reason in process.stderr


def test_mixed_typed_backward_descriptor_and_pair(compiler, monkeypatch):
    from tessera.compiler import nvidia_native as native, scheduled_checkpoint as checkpoint
    from tessera.compiler.typed_attention_backward import validate_typed_backward_descriptor
    graph = backward_module(q="f32", kv="f16", output="f32")
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    scheduled = checkpoint._decode_checkpoint(graph, schedule, tile, backward=True)
    assert scheduled.kv_storage == "f16" and scheduled.storage == "f32"
    monkeypatch.setattr(native, "_compile_tile_ir",
        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    descriptor = native.package_scheduled_checkpoint(
        scheduled, pipeline_name="tessera-nvidia-pipeline-sm120").descriptor
    assert descriptor.abi_id == native.SM120_ATTN_BWD_LSE_TYPED_ABI
    buffers = {b.name: b.dtype for b in descriptor.buffers}
    assert buffers["q"] == "fp32" and buffers["k"] == buffers["v"] == "fp16"
    assert buffers["dq"] == "fp32" and buffers["dk"] == buffers["dv"] == "fp16"
    assert validate_typed_backward_descriptor(descriptor) == ("f32", "f32")
    from dataclasses import replace
    for provenance in ({**descriptor.provenance, "kv_storage": "f32"},
                       {k: v for k, v in descriptor.provenance.items() if k != "kv_storage"}):
        with pytest.raises(ValueError):
            validate_typed_backward_descriptor(replace(descriptor, provenance=provenance))


def test_jvp_and_projection_owners_still_refuse_mixed_storage(compiler):
    """Only the paired saved-LSE reverse opts into mixed K/V; forward-mode AD
    over a mixed flash_attn must not be admitted."""
    text = '''module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
 func.func @f(%q: tensor<1x2x3x4xf16>, %k: tensor<1x1x5x4xbf16>, %v: tensor<1x1x5x3xbf16>) -> tensor<1x2x3x3xf16> {
  %o = "tessera.flash_attn"(%q, %k, %v) {head_dim = 4 : i64, causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (tensor<1x2x3x4xf16>, tensor<1x1x5x4xbf16>, tensor<1x1x5x3xbf16>) -> tensor<1x2x3x3xf16>
  return %o : tensor<1x2x3x3xf16>
 }
}'''
    process = subprocess.run([str(compiler), "--tessera-autodiff-forward"], input=text,
                             capture_output=True, text=True)
    assert process.returncode != 0 or "tessera_attn.checkpoint_jvp" not in process.stdout
