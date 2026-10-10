"""Symbolic saved-generation JVP export; device execution has a separate gate."""
import json
import re
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tessera.compiler.native_gpu_storage import _decode_image
from test_native_attention_ad_products import source

pytestmark = pytest.mark.compiler_route


def graph(query=True, key=True, wrt=(0, 1, 2)):
    text = source("forward", ", tessera.autodiff.wrt_indices = [" +
                  ", ".join(map(str, wrt)) + "]")
    if query:
        text = text.replace("1x2x4x8xf32", "1x2x?x8xf32")
    if key:
        text = text.replace("1x1x6x8xf32", "1x1x?x8xf32")
    capacities = (1, 2, 1, 9 if query else 4, 11 if key else 6, 8, 8)
    return text.replace("module {", 'module attributes {tessera.target = "nvidia_sm120", '
        'tessera.arch = "sm_120", tessera.attention_shape_bounds = array<i64: ' +
        ", ".join(map(str, capacities)) + ">} {", 1)


def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching native compiler required")
    return tool


@pytest.mark.parametrize("query,key", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("wrt", [(0,), (1,), (2,), (2, 0, 1)])
def test_bounded_jvp_export_keeps_symbolic_shapes_and_inactive_regions(query, key, wrt):
    tool = compiler()
    product = run_tessera_opt(tool, graph(query, key, wrt),
                             "--tessera-autodiff-forward=export-attention-jvp")
    fields = re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"', product)
    assert len(fields) == 1
    contract = json.loads(_decode_image(fields[0]).decode())
    assert contract["schema"] == 3
    assert contract["dims"] == [1, 2, 1, -1 if query else 4, -1 if key else 6, 8, 8]
    assert contract["shape_bounds"] == [1, 2, 1, 9 if query else 4, 11 if key else 6, 8, 8]
    assert contract["shape_policy"] == "bounded_sequences_v1"
    assert contract["active"] == [i in wrt for i in range(3)]
    assert "tessera_attn.checkpoint_jvp" in product
    # Reparse the registered Graph product to verify paired output/LSE lineage.
    run_tessera_opt(tool, product, "--canonicalize")
    dynamic_inactive = ((query and 0 not in wrt) or
                        (key and (1 not in wrt or 2 not in wrt)))
    if dynamic_inactive:
        assert "tensor.generate" in product and "tensor.dim" in product


@pytest.mark.parametrize("old,new,diagnostic", [
    ("1, 2, 1, 9, 11, 8, 8", "1, 2", "seven i64 capacities"),
    ("1, 2, 1, 9, 11, 8, 8", "1, 3, 1, 9, 11, 8, 8", "preserve fixed dimensions"),
    ("1, 2, 1, 9, 11, 8, 8", "1, 2, 1, 0, 11, 8, 8", "preserve fixed dimensions"),
    ("1, 2, 1, 9, 11, 8, 8", "1, 2, 1, 9223372036854775807, 11, 8, 8", "byte-address ABI"),
])
def test_bounded_jvp_export_rejects_invalid_capacity(old, new, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        run_tessera_opt(compiler(), graph().replace(old, new),
                       "--tessera-autodiff-forward=export-attention-jvp")

@pytest.mark.parametrize("case,activity,bias_shape", [
    ("biasvqk_bias_5_0_1x4x1x1", [False, False, False, True], [1, 4, 1, 1]),
    ("biasvqk_bias_v_k_q_5_1_2x4x3x5", [True, True, True, True], [2, 4, -1, -1]),
    ("biasvqk_k_bias_5_0_2x4x3x5", [False, True, False, True], [2, 4, -1, -1]),
])
def test_bounded_bias_jvp_preserves_physical_roles_and_symbolic_bias(case, activity, bias_shape):
    from test_native_attention_bias_jvp import source as bias_source
    text = bias_source(case)
    for old, new in [
        ("2x4x3x4xf32", "2x4x?x4xf32"),
        ("2x4x3x3xf32", "2x4x?x3xf32"),
        ("2x2x5x4xf32", "2x2x?x4xf32"),
        ("2x2x5x3xf32", "2x2x?x3xf32"),
        ("2x4x3x5xf32", "2x4x?x?xf32"),
    ]:
        text = text.replace(old, new)
    text = text.replace("module attributes {",
        "module attributes {tessera.attention_shape_bounds = array<i64: 2, 4, 2, 9, 11, 4, 3>, ", 1)
    tool = compiler()
    product = run_tessera_opt(tool, text, "--tessera-autodiff-forward=export-attention-jvp")
    fields = re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"', product)
    assert len(fields) == 1
    contract = json.loads(_decode_image(fields[0]).decode())
    assert contract["schema"] == 4
    assert contract["dims"] == [2, 4, 2, -1, -1, 4, 3]
    assert contract["shape_bounds"] == [2, 4, 2, 9, 11, 4, 3]
    assert contract["bias_shape"] == bias_shape
    assert contract["active"] == activity
    run_tessera_opt(tool, product, "--canonicalize")
