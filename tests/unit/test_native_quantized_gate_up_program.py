"""Native quantizer/expansion/BF16-store/activation partition contracts."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
from tests.unit.test_native_quantized_nvfp4_product_program import composed_source
from tests.unit.test_native_quantized_product_program import manifest,OPTION

pytestmark=pytest.mark.compiler_route


def source(m=3,width=17,k=48,reverse=False):
    n=2*width
    text=composed_source(m,n,k).replace("tensor<1xf64>","tensor<2xf64>")
    text=text.replace(f"[0 : i64, {n} : i64]",f"[0 : i64, {width} : i64, {n} : i64]")
    acc=f"tensor<{m}x{n}xf32>";parent=f"tensor<{m}x{n}xbf16>"
    out=f"tensor<{m}x{width}xbf16>";wide=f"tensor<{m}x{width}xf32>"
    text=text.replace(") -> "+acc,") -> "+out,1)
    lhs,rhs=("%right_f32","%left_f32") if reverse else ("%left_f32","%right_f32")
    tail=(f'%parent = tessera.cast %result {{dtype = "bf16"}} : ({acc}) -> {parent}\n'
          +f'%left = tessera.slice %parent {{start_indices = [0, 0], slice_sizes = [{m}, {width}]}} : ({parent}) -> {out}\n'
          +f'%right = tessera.slice %parent {{start_indices = [0, {width}], slice_sizes = [{m}, {width}]}} : ({parent}) -> {out}\n'
          +f'%left_f32 = tessera.cast %left {{dtype = "fp32"}} : ({out}) -> {wide}\n'
          +f'%right_f32 = tessera.cast %right {{dtype = "fp32"}} : ({out}) -> {wide}\n'
          +f'%gated = tessera.silu_mul {lhs}, {rhs} : ({wide}, {wide}) -> {wide}\n'
          +f'%activation = tessera.cast %gated {{dtype = "bf16"}} : ({wide}) -> {out}\n'
          +f'return %activation : {out}')
    old=f"return %result : {acc}"
    assert old in text
    return text.replace(old,tail)


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("m,width,k",[(1,1,16),(3,17,48)])
def test_native_four_members_keep_explicit_store_and_aliasing_views_in_members(m,width,k):
    text=source(m,width,k)
    plan=manifest(text)
    assert plan["schema"]=="tessera.native.quantized_product.v3"
    assert [s["inputs"] for s in plan["steps"]]==[[0],[1,2,3],[4,6,5,7],[8]]
    assert [s["outputs"] for s in plan["steps"]]==[[4,5],[6,7],[8],[9]]
    assert plan["output"]==9 and len(plan["buffers"])==10
    assert [b["ownership"] for b in plan["buffers"]]==["readonly_input"]*4+["private_scratch"]*5+["returned_output"]
    assert [(b["first_write"],b["last_read"]) for b in plan["buffers"]]==[(-1,0),(-1,1),(-1,1),(-1,1),(0,2),(0,2),(1,2),(1,2),(2,3),(3,4)]
    assert plan["buffers"][8]["storage"]=="bf16"
    assert plan["buffers"][8]["shape"]==[m,2*width]
    assert plan["buffers"][9]["shape"]==[m,width]
    assert "tessera.native.fp8_store_cast" in plan["member_graphs"][2]
    assert "tessera.native.gate_up_view" in plan["member_graphs"][3]
    for i,member in enumerate(plan["member_graphs"]):
        projected=run_tessera_opt(find_tessera_opt(),text,OPTION+f" select-scaled-member={i}")
        assert projected==member
        assert "tessera.native.quantized_product_projection" not in projected
        schedule=run_tessera_opt(find_tessera_opt(),projected,"--tessera-graph-to-schedule")
        tile=run_tessera_opt(find_tessera_opt(),schedule,"--tessera-schedule-to-tile")
        assert "tile." in tile
        if i==2:
            assert "tessera.cast" not in schedule and "bf16" in tile


@pytest.mark.usefixtures("production_compiler")
def test_native_activation_preserves_which_view_receives_silu():
    left,right=manifest(source()),manifest(source(reverse=True))
    assert left["buffers"]==right["buffers"]
    assert left["member_graphs"][3]!=right["member_graphs"][3]


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("old,new",[
    ('kind = "max"','kind = "sum"'),
    ('execution_mode = "exact_per_block"','execution_mode = "approximate"'),
    ('f64_product_to_f32_rne','truncate'),
])
def test_native_four_member_program_does_not_normalize_conflicting_math(old,new):
    text=source()
    assert old in text
    with pytest.raises(RuntimeError):manifest(text.replace(old,new))


@pytest.mark.usefixtures("production_compiler")
def test_native_four_member_frame_uses_original_argument_order():
    text=source()
    start=text.index("(",text.index("func.func"))+1;end=text.index(") ->",start)
    args=text[start:end].split(", ");assert len(args)==4
    text=text[:start]+", ".join(args[i] for i in (3,1,0,2))+text[end:]
    plan=manifest(text)
    assert plan["steps"][0]["inputs"]==[2]
    assert plan["steps"][1]["inputs"]==[1,3,0]
    assert [b["last_read"] for b in plan["buffers"][:4]]==[1,1,0,1]


@pytest.mark.usefixtures("production_compiler")
def test_native_four_member_admission_preserves_and_rejects_unknown_function_math():
    text=source().replace(') -> tensor<3x17xbf16> {',
        ') -> tensor<3x17xbf16> attributes {tessera.unproved_math = "approximate"} {',1)
    assert "tessera.unproved_math" in text
    run_tessera_opt(find_tessera_opt(),text,"--verify-each")
    with pytest.raises(RuntimeError):manifest(text)
