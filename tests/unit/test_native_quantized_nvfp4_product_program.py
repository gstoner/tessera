"""Original quantizer/weight-expansion/product SSA, without host intermediates."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_quantized_product_program import source, manifest, OPTION
from tests.unit.test_rocm_graph_nvfp4_expand import POLICY

pytestmark = pytest.mark.compiler_route


def composed_source(m=3, n=17, k=32):
    text = source(m, n, k)
    old = f"%b: tensor<{n}x{k}xf8E4M3FN>, %sb: tensor<{k//16}x{n}xf32>"
    roots = f"%packed: tensor<{n}x{k//2}xi8>, %weight_scales: tensor<{n}x{k//16}xf8E4M3FN>, %globals: tensor<1xf64>"
    expansion = ('%b, %sb = "tessera.nvfp4_expand"(%packed, %weight_scales, %globals) '
                 + '{row_offsets = [0 : i64, ' + str(n) + ' : i64], numeric_policy = {'
                 + POLICY + '}} : ('
                 + f'tensor<{n}x{k//2}xi8>, tensor<{n}x{k//16}xf8E4M3FN>, tensor<1xf64>) '
                 + f'-> (tensor<{n}x{k}xf8E4M3FN>, tensor<{k//16}x{n}xf32>)\n')
    assert old in text
    return text.replace(old, roots).replace('%result = tessera.scaled_matmul', expansion + '%result = tessera.scaled_matmul')


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("m,n,k", [(1,1,16), (3,17,32), (19,33,64)])
def test_native_three_member_program_preserves_four_private_edges(m,n,k):
    text = composed_source(m,n,k)
    plan = manifest(text)
    assert plan["schema"] == "tessera.native.quantized_product.v2"
    assert [s["inputs"] for s in plan["steps"]] == [[0], [1,2,3], [4,6,5,7]]
    assert [s["outputs"] for s in plan["steps"]] == [[4,5], [6,7], [8]]
    assert plan["output"] == 8
    assert [b["ownership"] for b in plan["buffers"]] == ["readonly_input"]*4 + ["private_scratch"]*4 + ["returned_output"]
    assert [(b["first_write"], b["last_read"]) for b in plan["buffers"]] == [(-1,0),(-1,1),(-1,1),(-1,1),(0,2),(0,2),(1,2),(1,2),(2,3)]
    tool = find_tessera_opt()
    for index, carrier in enumerate(("tile.row_region_kernel", "tile.nvfp4_expand_kernel", "tile.scaled_matmul_kernel")):
        member = run_tessera_opt(tool,text,OPTION+f" select-scaled-member={index}")
        assert member == plan["member_graphs"][index]
        schedule = run_tessera_opt(tool,member,"--tessera-graph-to-schedule")
        tile = run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
        assert carrier in tile
    assert 'tessera.native.quantized_product_json' not in text


@pytest.mark.usefixtures("production_compiler")
def test_native_original_root_order_is_not_assumed():
    text = composed_source()
    start = text.index("(",text.index("func.func"))+1
    end = text.index(") ->",start)
    args=text[start:end].split(", ")
    assert len(args)==4
    text=text[:start]+", ".join(args[i] for i in (3,1,0,2))+text[end:]
    plan=manifest(text)
    assert plan["steps"][0]["inputs"]==[2]
    assert plan["steps"][1]["inputs"]==[1,3,0]
    assert [b["last_read"] for b in plan["buffers"][:4]]==[1,1,0,1]


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("old,new",[
    ('kind = "max"','kind = "sum"'),
    ('f64_product_to_f32_rne','truncate'),
    ('execution_mode = "exact_per_block"','execution_mode = "approximate"'),
    ('[0 : i64, 17 : i64]','[0 : i64, 16 : i64]'),
    ('scales(%scales, %sb)','scales(%scales, %scales)'),
])
def test_native_three_member_program_rejects_conflicting_semantics(old,new):
    text=composed_source()
    assert old in text
    with pytest.raises(RuntimeError):
        manifest(text.replace(old,new))
