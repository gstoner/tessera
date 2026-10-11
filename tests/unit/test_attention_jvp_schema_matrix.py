"""Regression matrix for output selection and bounded attention AD schemas."""
import json
import re

import pytest

from tessera.compiler.native_gpu_storage import _decode_image
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route


def graph(bounded, saved_lse, biased):
    seq = "?" if bounded else "4"
    key = "?" if bounded else "6"
    q = f"tensor<1x2x{seq}x8xf32>"
    k = f"tensor<1x1x{key}x8xf32>"
    lse = f"tensor<1x2x{seq}xf32>"
    bias = f"tensor<1x2x{seq}x{key}xf32>"
    extra_arg = f", %bias: {bias}" if biased else ""
    operands = "%q, %k, %v" + (", %bias" if biased else "")
    types = f"{q}, {k}, {k}" + (f", {bias}" if biased else "")
    results = f"({q}, {lse})" if saved_lse else q
    names = "%o, %lse" if saved_lse else "%o"
    policy = ', lse_checkpoint = "saved"' if saved_lse else ""
    capacities = ", tessera.attention_shape_bounds = array<i64: 1, 2, 1, 9, 11, 8, 8>" if bounded else ""
    return f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"{capacities}}} {{
      func.func @attention(%q: {q}, %k: {k}, %v: {k}{extra_arg}) -> {results}
      attributes {{tessera.autodiff = "forward"}} {{
        {names} = "tessera.flash_attn"({operands}) {{head_dim = 8 : i64, causal = true,
          dropout_p = 0.0 : f64, operandSegmentSizes = array<i32: 1, 1, 1, {int(biased)}>{policy}}}
          : ({types}) -> {results}
        return {names} : {q + ", " + lse if saved_lse else q}
      }}
    }}'''


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("saved_lse", [False, True])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("value_only", [False, True])
def test_schema_matrix_keeps_shape_policy_and_output_abi(bounded, saved_lse, biased, value_only):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching native compiler required")
    source = graph(bounded, saved_lse, biased)
    if value_only:
        source = source.replace('tessera.autodiff = "forward"',
                                'tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [2]')
    product = run_tessera_opt(tool, source,
                             "--tessera-autodiff-forward=export-attention-jvp")
    fields = re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"', product)
    assert len(fields) == 1
    contract = json.loads(_decode_image(fields[0]).decode())
    expected = (5 if bounded else 6) if saved_lse else (3 if bounded else 1) + int(biased)
    assert contract["schema"] == expected
    assert contract.get("saved_lse", False) is saved_lse
    assert bool(contract.get("shape_bounds")) is bounded
    assert bool(contract.get("bias_shape")) is biased
    if bounded:
        assert contract["shape_policy"] == "bounded_sequences_v1"
    tile = run_tessera_opt(tool, product,
                          "--pass-pipeline=builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile)")
    fields = re.findall(r'tessera\.native_tensor_contract = "((?:\\.|[^"\\])*)"', tile)
    assert len(fields) == 1
    manifest = json.loads(_decode_image(fields[0]).decode())
    names = [a["name"] for a in manifest["arguments"]]
    assert ("dlse" in names) is saved_lse
    if bounded:
        assert names[-3:] == ["scratch", "query_size", "key_size"]
    else:
        assert names[-1:] == ["scratch"]
    outputs = [a["name"] for a in manifest["arguments"] if a.get("writable")]
    assert outputs == (["tangent", "dlse"] if saved_lse else ["tangent"])



@pytest.mark.parametrize("bounded,saved_lse,biased,schema", [
    (False, False, False, 1), (False, False, True, 2),
    (True, False, False, 3), (True, False, True, 4),
    (True, True, False, 5), (True, True, True, 5),
    (False, True, False, 6), (False, True, True, 6),
])
def test_adapter_accepts_matching_schema_and_rejects_collisions(monkeypatch, bounded, saved_lse, biased, schema):
    from tessera.compiler import native_attention_jvp as adapter
    from tessera.compiler import scheduled_matmul
    from tessera.compiler.attention_shape_contract import DYNAMIC_DIM
    dims = (1, 2, 1, DYNAMIC_DIM if bounded else 4, DYNAMIC_DIM if bounded else 6, 8, 8)
    bounds = (1, 2, 1, 9, 11, 8, 8) if bounded else ()
    bias = (1, 2, dims[3], dims[4]) if biased else ()
    portable = lambda shape: [-1 if d == DYNAMIC_DIM else d for d in shape]
    contract = dict(schema=schema, dims=portable(dims), scale=0.5, causal=True,
                    bias_shape=portable(bias), active=[True] * (3 + int(biased)))
    if bounded:
        contract.update(shape_bounds=list(bounds), shape_policy="bounded_sequences_v1")
    if saved_lse:
        contract["saved_lse"] = True

    def export(*args):
        literal = json.dumps(json.dumps(contract))
        return 'module attributes {tessera.autodiff.attention_jvp_contract = ' + literal + '} {}'

    monkeypatch.setattr(scheduled_matmul, "run_tessera_opt", export)
    monkeypatch.setattr(adapter, "_lower_graph", lambda *a, **k: "verified tile")
    monkeypatch.setattr(adapter, "build_native_gpu_storage", lambda *a, **k: "package")
    kwargs = dict(compiler="compiler", llvm_bin="llvm", bias_shape=bias, shape_bounds=bounds)
    assert adapter.materialize_generated("graph", dims, 0.5, True, **kwargs) == "package"
    # All other numeric variants must be rejected, including branch-local
    # saved-LSE schema 3 and the other static/bounded saved-LSE variant.
    for wrong in (1, 2, 3, 4, 5, 6, True):
        if type(wrong) is int and wrong == schema:
            continue
        contract["schema"] = wrong
        with pytest.raises(ValueError, match="policy differs"):
            adapter.materialize_generated("graph", dims, 0.5, True, **kwargs)
    contract["schema"] = schema
    contract["shape_policy"] = "static" if bounded else "bounded_sequences_v1"
    with pytest.raises(ValueError, match="policy differs"):
        adapter.materialize_generated("graph", dims, 0.5, True, **kwargs)
