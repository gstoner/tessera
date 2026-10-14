"""Public bounds binding and native paired O/LSE projection."""
import copy
import json
import re

import numpy as np
import pytest
import tessera as ts

from tessera.compiler.native_gpu_storage import _decode_image
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_attention_jvp_schema_matrix import graph


def saved(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")


def source(biased=False, bounds="9, 11"):
    return graph(False, True, biased).replace(
        'tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: ' + bounds + ">")


def test_public_jvp_keeps_static_trace_and_requests_native_projection(monkeypatch):
    from tessera.compiler import native_attention_program
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("v", "q"))(saved)
    original = copy.deepcopy(fn.graph_ir)
    recorded = []
    monkeypatch.setattr(native_attention_program, "compile_attention_program",
                        lambda text, *a, **k: recorded.append((text, a)) or "program")
    values = [np.zeros(shape, np.float32) for shape in
              ((1, 2, 3, 8), (1, 1, 5, 8), (1, 1, 5, 8))]
    assert fn.compile_native_attention_jvp(
        *values, compiler="unused", llvm_bin="unused", sequence_bounds=(9, 11)) == "program"
    text, args = recorded[0]
    assert 'tessera.attention_sequence_bounds = array<i64: 9, 11>' in text
    assert "tensor<1x2x3x8xf32>" in text and "tensor<1x2x?x8xf32>" not in text
    assert args[0] == (2, 0)
    assert fn.graph_ir == original


@pytest.mark.parametrize("bounds", [(0, 11), (9,), (True, 11), (9, 11.0), "9,11"])
def test_public_bounds_rejected_before_tracing(bounds, monkeypatch):
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q",))(saved)
    monkeypatch.setattr(fn, "_specialized_autodiff_module",
                        lambda *a, **k: pytest.fail("invalid capacity reached tracing"))
    with pytest.raises(ts.TesseraJitError, match="positive Sq/Sk"):
        fn.compile_native_attention_jvp(compiler="unused", llvm_bin="unused",
                                        sequence_bounds=bounds)


@pytest.mark.compiler_route
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("wrt", [(0, 1, 2), (2,)])
def test_native_projection_exports_schema_five_and_dynamic_lse(biased, wrt):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching native compiler required")
    text = source(biased).replace('tessera.autodiff = "forward"',
        'tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [' +
        ", ".join(map(str, wrt)) + "]")
    product = run_tessera_opt(tool, text, "--tessera-autodiff-forward=export-attention-jvp")
    fields = re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"', product)
    contract = json.loads(_decode_image(fields[0]))
    assert contract["schema"] == 5 and contract["saved_lse"] is True
    assert contract["dims"] == [1, 2, 1, -1, -1, 8, 8]
    assert contract["shape_bounds"] == [1, 2, 1, 9, 11, 8, 8]
    assert contract["shape_policy"] == "bounded_sequences_v1"
    assert "tessera.attention_sequence_bounds" not in product
    tile = run_tessera_opt(tool, product,
        "--pass-pipeline=builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile)")
    fields = re.findall(r'tessera.native_tensor_contract = "((?:\\.|[^"\\])*)"', tile)
    manifest = json.loads(_decode_image(fields[0]))
    assert manifest["schema"] == 2
    assert [row["name"] for row in manifest["arguments"]][-4:] == [
        "dlse", "scratch", "query_size", "key_size"]


@pytest.mark.compiler_route
@pytest.mark.parametrize("bounds", ["9", "0, 11", "3, 11", "9, 5"])
def test_native_forward_projection_rejects_invalid_capacity(bounds):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching native compiler required")
    with pytest.raises(RuntimeError, match="projection|capacity"):
        run_tessera_opt(tool, source(bounds=bounds),
                       "--tessera-autodiff-forward=export-attention-jvp")


@pytest.mark.compiler_route
@pytest.mark.parametrize("sq,sk,expected", [
    (1, 4, [1, 2, 1, -1]),
    (3, 1, [1, 2, -1, 1]),
    (1, 1, [1, 2, 1, 1]),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_trace_at_one_preserves_physical_bias_extent(sq, sk, expected, reverse):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching native compiler required")
    text = source(True)
    text = (text.replace("1x2x4x6", f"1x2x{sq}x{sk}")
                .replace("1x2x4x8", f"1x2x{sq}x8")
                .replace("1x1x6x8", f"1x1x{sk}x8")
                .replace("1x2x4xf32", f"1x2x{sq}xf32"))
    if reverse:
        from tessera.compiler.scheduled_checkpoint import lower_generated_checkpoint
        # Paired checkpoint reverse consumes O and retains LSE as a residual.
        plain = graph(False, False, True).replace(
            'tessera.arch = "sm_120"',
            'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 9, 11>')
        plain = (plain.replace('tessera.autodiff = "forward"', 'tessera.autodiff = "reverse"')
                      .replace("1x2x4x6", f"1x2x{sq}x{sk}")
                      .replace("1x2x4x8", f"1x2x{sq}x8")
                      .replace("1x1x6x8", f"1x1x{sk}x8"))
        artifact = lower_generated_checkpoint(plain, backward=True, prune_inactive=True)
        assert artifact.bias_shape == tuple(-(1 << 63) if d == -1 else d for d in expected)
    else:
        product = run_tessera_opt(tool, text, "--tessera-autodiff-forward=export-attention-jvp")
        fields = re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"', product)
        contract = json.loads(_decode_image(fields[0]))
        assert contract["schema"] == 5
        assert contract["bias_shape"] == expected
