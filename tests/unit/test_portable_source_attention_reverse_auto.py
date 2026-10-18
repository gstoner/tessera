"""Compiler-free auto-policy replay; fixture PTX is never executed."""
from copy import deepcopy
import subprocess

import pytest

from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.portable_source_attention_reverse import restore
from tests.unit.test_native_source_attention_reverse_auto import auto_source
from tests.unit.test_portable_source_attention_reverse import portable_program, reseal

pytestmark = pytest.mark.compiler_route
AUTO = "tessera.native.source_attention_reverse_auto_request"


@pytest.fixture(scope="module", params=[
    ("f16", False), ("bf16", False), ("f16", True), ("bf16", True)])
def auto_portable(compiler_toolchain, request):
    storage, wide = request.param
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    dims = (17, 19, 257, 13) if wide else (3, 5, 4, 3)
    data = portable_program(export_native_source_attention_reverse(auto_source(storage, dims=dims), tool=tool))
    return data, wide


def test_auto_selected_program_restores_without_compiler_or_provider(auto_portable, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    data, wide = auto_portable
    def forbidden(*args, **kwargs):
        pytest.fail("auto replay invoked compiler/provider")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", forbidden)
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    _, plan, members, _ = restore(data)
    assert len(members) == (5 if wide else 4)
    assert plan["schema"].endswith(".v2" if wide else ".v1")


@pytest.mark.parametrize("value", ["false", "true", '"auto"'])
def test_rehashed_invalid_auto_type_is_rejected(auto_portable, value, monkeypatch):
    from tessera import runtime as rt
    data, _ = auto_portable
    data = deepcopy(data)
    data["plan"]["source_graph_ir"] = data["plan"]["source_graph_ir"].replace(AUTO + ",", AUTO + " = " + value + ",")
    def forbidden(*args, **kwargs):
        pytest.fail("invalid auto replay reached provider")
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    with pytest.raises(ValueError):
        restore(reseal(data))


@pytest.mark.parametrize("placement", ["absent", "function", "string"])
def test_only_module_auto_request_selects_rows(auto_portable, placement):
    data, wide = auto_portable
    data = deepcopy(data)
    text = data["plan"]["source_graph_ir"].replace(AUTO + ", ", "")
    if placement == "function":
        index = text.index("attributes {", text.index("func.func @")) + len("attributes {")
        text = text[:index] + AUTO + ", " + text[index:]
    elif placement == "string":
        text = text.replace("module attributes {", 'module attributes {debug_note = ", ' + AUTO + ', ", ', 1)
    data["plan"]["source_graph_ir"] = text
    if wide:
        with pytest.raises(ValueError):
            restore(reseal(data))
    else:
        _, plan, members, _ = restore(reseal(data))
        assert plan["schema"].endswith(".v1") and len(members) == 4
