"""Compiler-owned optional bias gradient; runtime/automatic AD admission is separate."""
import re
import pytest
from tessera.compiler.scheduled_checkpoint import _graph_text
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")


def bias_gradient_graph(with_bias=True, result_type="tensor<1x2x3x4xf32>"):
    names = ("do", "q", "k", "v", "output", "bias", "lse", "dq", "dk", "dv")
    if not with_bias:
        names = tuple(n for n in names if n != "bias")
    text = _graph_text(names, (1, 2, 1, 3, 4, 4, 3), .5, True, True, with_bias)
    output_types = "tensor<1x2x3x4xf32>, tensor<1x1x4x4xf32>, tensor<1x1x4x3xf32>"
    text = text.replace("-> (" + output_types + ")", "-> (" + output_types + ", " + result_type + ")")
    text = text.replace('["dq", "dk", "dv"]', '["dq", "dk", "dv", "dbias"]')
    text = text.replace("%r0, %r1, %r2", "%r0, %r1, %r2, %r3")
    text = text.replace("return %r0, %r1, %r2, %r3 : " + output_types,
                        "return %r0, %r1, %r2, %r3 : " + output_types + ", " + result_type)
    return text


def test_bias_gradient_survives_schedule_replay_and_native_lowering():
    tool = find_tessera_opt()
    graph = bias_gradient_graph()
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    assert 'results = ["dq", "dk", "dv", "dbias"]' in tile
    assert "bias_gradient = true" in tile
    signature = re.search(r"llvm.func @\w+\((.*?)\)", tile)[1]
    assert signature.count("!llvm.ptr") == 11
    assert signature.count("i64") == 7
    from tessera.compiler.nvidia_native import _compile_tile_ir
    target, ptx, *_ = _compile_tile_ir(tile, re.search(r"llvm.func @(\w+)", tile)[1])
    assert "tile.attention_backward_kernel" not in target
    assert ".entry" in ptx
    assert re.search(r"st\.global\.(?:f32|b32)", ptx)


@pytest.mark.parametrize("with_bias,result_type", [
    (False, "tensor<1x2x3x4xf32>"),
    (True, "tensor<1x2x3x5xf32>"),
    (True, "tensor<1x2x3x4xf16>"),
])
def test_bias_gradient_rejects_missing_bias_or_wrong_result(with_bias, result_type):
    with pytest.raises(RuntimeError, match="bias gradient"):
        run_tessera_opt(find_tessera_opt(), bias_gradient_graph(with_bias, result_type),
                       "--tessera-graph-to-schedule")

def test_extended_checkpoint_projects_checked_four_output_abi(monkeypatch):
    from dataclasses import replace
    from tessera.compiler import nvidia_native as native
    from tessera.compiler.scheduled_checkpoint import lower_scheduled_checkpoint
    names = ("do", "q", "k", "v", "output", "bias", "lse", "dq", "dk", "dv", "dbias")
    artifact = lower_scheduled_checkpoint(names, (1,2,1,3,4,4,3), .5, True,
        backward=True, bias=True, bias_gradient=True)
    monkeypatch.setattr(native, "_compile_tile_ir",
        lambda *args: (args[0], "// PTX", {}, "compiler", "toolchain", (), "cold"))
    package = native.package_scheduled_checkpoint(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert package.descriptor.abi_id == native.SM120_ATTN_BWD_LSE_BIAS_GRAD_F32_ABI
    assert "_bias_gradient_" in package.descriptor.entry_symbol
    assert tuple(x.name for x in package.descriptor.buffers) == names
    assert [x.direction for x in package.descriptor.buffers[-4:]] == ["output"] * 4
    assert package.descriptor.scalars[0].ordinal == 11
    with pytest.raises(ValueError, match="metadata"):
        replace(artifact, bias_gradient=False).validate()


def test_bias_gradient_checked_package_executes_on_sm120():
    from tests._support.nvidia import nvidia_cuda_host_ready
    if not nvidia_cuda_host_ready():
        pytest.skip("requires owning SM120 device")
    from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import run_case
    result = run_case((1,2,1,3,5,4,3), True, samples=3, reps=20)
    assert max(result["max_abs_errors"]) < 4e-5

@pytest.mark.parametrize("wrt,active", [(("v","bias","q","k"),[2,3,0,1]), (("q",),[0])])
def test_jit_bias_gradient_keeps_private_saved_generation_on_sm120(wrt,active):
    from tests._support.nvidia import nvidia_cuda_host_ready
    if not nvidia_cuda_host_ready():
        pytest.skip("requires owning SM120 device")
    from benchmarks.nvidia.benchmark_jit_attention_bias_vjp import run_case
    row=run_case((1,2,1,3,5,4,3),True,wrt,samples=3)
    assert row["active"] == active
    assert row["max_abs_gradient_error"] < 4e-5
