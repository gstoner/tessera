"""Native quantizer/product SSA partition and private-buffer lifetime proof."""
import base64
import json
import re

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_k16_quant_graph import source as quant_source

pytestmark = pytest.mark.compiler_route
OPTION = "--tessera-autodiff-forward=export-scaled-primal=true"


def source(m=3, n=17, k=32, output="f32"):
    text = quant_source(m, k).replace("tessera.native.block_quant", "tessera.native.quantized_product")
    q = f"tensor<{m}x{k}xf8E4M3FN>"
    sa = f"tensor<{m}x{k//16}xf32>"
    b = f"tensor<{n}x{k}xf8E4M3FN>"
    sb = f"tensor<{k//16}x{n}xf32>"
    out = f"tensor<{m}x{n}x{output}>"
    text = text.replace(f"%x: tensor<{m}x{k}xbf16>) -> ({q}, {sa})",
                        f"%x: tensor<{m}x{k}xbf16>, %b: {b}, %sb: {sb}) -> {out}")
    text = text.replace(f"return %codes, %scales : {q}, {sa}", f'''%result = tessera.scaled_matmul %codes, %b scales(%scales, %sb) {{
      transposeB = true,
      numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
      scale_layout = {{granularity = "block", block = [1, 16], format = "fp32"}}
    }} : ({q}, {b}, {sa}, {sb}) -> {out}
    return %result : {out}''')
    return text


def manifest(text):
    result = run_tessera_opt(find_tessera_opt(), text, OPTION)
    encoded = re.search(r'tessera.native.quantized_product_json = "([^"\n]+)"', result)[1]
    return json.loads(base64.b64decode(encoded))


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("m,n,k,out", [(3,17,32,"f32"), (19,33,64,"bf16")])
def test_native_partition_preserves_members_and_private_lifetimes(m,n,k,out):
    text = source(m,n,k,out)
    plan = manifest(text)
    assert plan["schema"] == "tessera.native.quantized_product.v1"
    assert [s["inputs"] for s in plan["steps"]] == [[0], [3,1,4,2]]
    assert [s["outputs"] for s in plan["steps"]] == [[3,4], [5]]
    assert plan["output"] == 5
    assert [b["ownership"] for b in plan["buffers"]] == ["readonly_input"]*3 + ["private_scratch"]*2 + ["returned_output"]
    assert [(b["first_write"],b["last_read"]) for b in plan["buffers"]] == [(-1,0),(-1,1),(-1,1),(0,1),(0,1),(1,2)]
    assert [b["bytes"] for b in plan["buffers"]] == [m*k*2,n*k,k//16*n*4,m*k,m*k//16*4,m*n*(2 if out=="bf16" else 4)]
    for index, member in enumerate(plan["member_graphs"]):
        projected = run_tessera_opt(find_tessera_opt(),text,OPTION+f" select-scaled-member={index}")
        assert projected == member
        schedule = run_tessera_opt(find_tessera_opt(), member, "--tessera-graph-to-schedule")
        assert "schedule.artifact" in schedule
        tile = run_tessera_opt(find_tessera_opt(), schedule, "--tessera-schedule-to-tile")
        assert "tile." in tile
    assert "tessera.native.quantized_product_json" not in text


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("old,new", [
    ('kind = "max"', 'kind = "sum"'),
    ("dense<448.0>", "dense<449.0>"),
    ('execution_mode = "exact_per_block"', 'execution_mode = "approximate"'),
    ('granularity = "block"', 'granularity = "per_tensor"'),
    ("scales(%scales, %sb)", "scales(%scales, %scales)"),
    ('"gfx1201"', '"gfx1151"'),
    ("transposeB = true,", "transposeB = true, transposeA = true,"),
    ('accum = "fp32",', 'accum = "fp32", unexpected = true,'),
])
def test_composed_contract_never_rewrites_conflicting_semantics(old,new):
    with pytest.raises(RuntimeError):
        manifest(source().replace(old,new))
