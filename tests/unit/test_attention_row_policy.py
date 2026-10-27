"""SM120-TYPED-REVERSE-ROW-PREPASS-20261026: automatic row-correction policy."""
import pytest

from tessera.compiler import native_attention_vjp_runtime as runtime


def graph(q, k, v, bias=False):
    types = [f"tensor<1x2x3x4x{q}>", f"tensor<1x1x5x4x{k}>", f"tensor<1x1x5x3x{v}>"]
    if bias:
        types.append("tensor<1x2x1x5xf32>")
    # Canonical traced Graph IR form, including the trailing source location.
    return ("module {\n  func.func @attention() {\n"
            f"    %v0 = tessera.flash_attn %a0, %a1, %a2 {{causal = true, head_dim = 4 : i64, "
            f"operandSegmentSizes = array<i32: 1, 1, 1, {int(bias)}>}} : ({', '.join(types)}) "
            f'-> tensor<1x2x3x3x{q}> loc("f.py":3:12)\n  }}\n}}')


@pytest.mark.parametrize("q,kv,expected", [
    ("f16", "f16", "prepass"), ("bf16", "bf16", "prepass"),
    ("f16", "bf16", "prepass"), ("f32", "f16", "prepass"), ("f32", "bf16", "prepass"),
    ("bf16", "f32", "prepass"), ("f32", "f32", "inline")])
@pytest.mark.parametrize("bias", [False, True])
def test_auto_policy_selects_prepass_for_any_half_role(monkeypatch, q, kv, expected, bias):
    monkeypatch.delenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION", raising=False)
    assert runtime._operand_storages(graph(q, kv, kv, bias)) == (q, kv, kv)
    assert runtime._row_policy(graph(q, kv, kv, bias)) == expected


def test_distinct_k_and_v_storage_keeps_inline(monkeypatch):
    monkeypatch.delenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION", raising=False)
    assert runtime._row_policy(graph("f16", "f16", "bf16")) == "inline"


@pytest.mark.parametrize("requested", ["inline", "prepass"])
def test_explicit_policy_overrides_selection(monkeypatch, requested):
    monkeypatch.setenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION", requested)
    assert runtime._row_policy(graph("f32", "f32", "f32")) == requested


def test_unknown_policy_and_unreadable_graph_fail_closed(monkeypatch):
    monkeypatch.setenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION", "fast")
    with pytest.raises(ValueError, match="auto, inline or prepass"):
        runtime._row_policy(graph("f16", "f16", "f16"))
    monkeypatch.delenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION")
    with pytest.raises(ValueError, match="operand storage"):
        runtime._row_policy("module {}")
