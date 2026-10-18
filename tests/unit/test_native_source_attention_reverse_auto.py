"""Native automatic source-adjoint strategy, independent of Python shape dispatch."""
import pytest
from tests.unit.test_native_source_attention_reverse_outline import source, outline
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def auto_source(storage="f16", variant="normal", dims=(3, 5, 4, 3)):
    sq, sk, d, dv = dims
    text = source(storage, variant)
    replacements = {
        f"tensor<1x2x3x4x{storage}>": f"tensor<1x2x{sq}x{d}x{storage}>",
        f"tensor<1x1x5x4x{storage}>": f"tensor<1x1x{sk}x{d}x{storage}>",
        f"tensor<1x1x5x3x{storage}>": f"tensor<1x1x{sk}x{dv}x{storage}>",
        f"tensor<1x2x3x3x{storage}>": f"tensor<1x2x{sq}x{dv}x{storage}>",
        "tensor<1x2x3xf32>": f"tensor<1x2x{sq}xf32>",
        "head_dim = 4 : i64": f"head_dim = {d} : i64",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("module attributes {",
        "module attributes {tessera.native.source_attention_reverse_auto_request = unit, ", 1)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
@pytest.mark.parametrize("dims,version,count", [((3, 5, 4, 3), 1, 4), ((17, 19, 257, 13), 2, 5)])
def test_native_auto_uses_actual_query_role(compiler_toolchain, storage, variant, dims, version, count):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    _, plan = outline(tool, auto_source(storage, variant, dims))
    assert plan["schema"] == f"tessera.native.source_attention_reverse.v{version}"
    assert len(plan["steps"]) == len(plan["member_graphs"]) == count


@pytest.mark.parametrize("fault", ["false", "true", "string", "missing_reverse"])
def test_native_auto_request_cannot_be_silently_ignored(compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = auto_source()
    if fault == "missing_reverse":
        text = text.replace("tessera.native.source_attention_reverse_request = unit, ", "")
    else:
        value = '"auto"' if fault == "string" else fault
        text = text.replace("source_attention_reverse_auto_request = unit",
                            "source_attention_reverse_auto_request = " + value)
    result = invoke(tool, text, "--tessera-autodiff-paired")
    assert result.returncode != 0, "invalid native automatic request was silently ignored"



@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("dims,version", [
    ((15, 5, 35, 7), 1), ((16, 5, 35, 7), 2),
    ((16, 5, 31, 7), 1), ((16, 5, 32, 7), 2),
    ((16, 3, 35, 7), 1), ((16, 4, 35, 7), 2),
    ((16, 5, 35, 3), 1), ((16, 5, 35, 4), 2),
    ((16, 4, 32, 4), 2)])
def test_auto_policy_boundaries_are_native(compiler_toolchain, storage, dims, version):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    _, plan = outline(tool, auto_source(storage, dims=dims))
    assert plan["schema"] == f"tessera.native.source_attention_reverse.v{version}"


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_explicit_rows_remains_authoritative_with_auto(compiler_toolchain, storage):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = auto_source(storage).replace("module attributes {",
        "module attributes {tessera.native.source_attention_reverse_rows_request = unit, ", 1)
    _, plan = outline(tool, text)
    assert plan["schema"] == "tessera.native.source_attention_reverse.v2"
