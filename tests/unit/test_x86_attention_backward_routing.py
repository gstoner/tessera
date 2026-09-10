from types import SimpleNamespace

from tessera.compiler import x86_native
from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module


def test_native_selector_routes_attention_backward_through_scheduled_artifact(monkeypatch):
    module = SimpleNamespace(functions=[SimpleNamespace(
        body=[SimpleNamespace(op_name="tessera.flash_attn_bwd")]
    )])
    monkeypatch.setattr(
        "tessera.compiler.scheduled_attention_backward.supports_scheduled_attention_backward",
        lambda value, *, target: value is module and target == "x86",
    )
    for name in ("supports_softmax", "supports_reduction", "supports_promoted_matmul",
                 "supports_attention", "supports_promoted_elementwise"):
        monkeypatch.setattr(x86_native, name, lambda value: False)
    monkeypatch.setattr(
        "tessera.compiler.x86_breadth.supports_promoted_graph_breadth",
        lambda value: False,
    )
    artifact, package = object(), object()
    monkeypatch.setattr(
        "tessera.compiler.scheduled_attention_backward.lower_scheduled_attention_backward",
        lambda value, *, target: artifact,
    )
    monkeypatch.setattr(
        x86_native, "package_scheduled_attention_backward",
        lambda value, *, pipeline_name: package,
    )
    assert x86_native.supports_native_package(module)
    assert x86_native.native_package_kind(module) == "attention_backward"
    assert x86_native.package_native(module, pipeline_name="x86") is package


def test_real_saved_lse_backward_contract_is_automatically_admitted():
    module = _module(1, 4, 2, 17, 19, 16, dtype="fp32", lse_checkpoint="auto")
    module.functions[0].body[0].kwargs["window"] = 3
    module.functions[0].body[0].kwargs["softcap"] = 4.0
    assert x86_native.requests_attention_backward(module)
    assert x86_native.supports_native_package(module)
    assert x86_native.native_package_kind(module) == "attention_backward"
