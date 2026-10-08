"""Exact Graph casts precede native f32 math; no implicit dtype rewriting."""
from copy import deepcopy
import pytest
from tessera.compiler import rocm_math_native as native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt


def module(kind="sqrt",storage="f16",shape=(3,17),reverse=False):
    dims=tuple(map(str,shape))
    source=IRType("tensor<"+"x".join(dims)+"x"+storage+">",dims,{"f16":"fp16","bf16":"bf16","f32":"fp32"}[storage])
    result=IRType("tensor<"+"x".join(dims)+"xf32>",dims,"fp32")
    names=["a","b"] if kind in {"add","div"} else ["a"]
    casts=[IROp(result="wide_"+name,op_name="tessera.cast",operands=["%"+name],
                 operand_types=[str(source)],result_type=str(result),kwargs={"dtype":"fp32"}) for name in names]
    ordered=list(reversed(names)) if reverse else names
    consumer=IROp(result="out",op_name="tessera."+kind,operands=["%wide_"+name for name in ordered],
                   operand_types=[str(result)]*len(names),result_type=str(result),
                   kwargs={"axis":-1} if kind.startswith("cum") else {})
    return GraphIRModule([GraphIRFunction(name="widen_math",args=[IRArg(name,source) for name in names],
        result_types=[result],body=casts+[consumer],return_values=["%out"])])


@pytest.mark.parametrize("target",["rocm_gfx1151","rocm_gfx1201"])
@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("kind",["sqrt","exp","add","div","cumsum","cummax"])
def test_exact_widening_graph_replays_without_mutating_source(target,storage,kind):
    if find_tessera_opt() is None:pytest.skip("requires matching compiler")
    graph=module(kind,storage,reverse=kind in {"add","div"})
    before=deepcopy(graph)
    recipe=native.lower_math_graph(graph,target)
    assert graph==before
    assert "tessera.cast" in recipe.graph_ir and "tessera.cast" in recipe.schedule_ir
    assert "tessera.cast" not in recipe.tile_ir
    info=native._info(recipe.tile_ir)
    assert info["storage"]==storage and info["output_storage"]=="f32"
    assert info["roles"]==((1,0) if kind in {"add","div"} else (0,))
    assert storage+"_f32" in native.math_abi(info)


@pytest.mark.parametrize("bad",["cast_policy","extra_cast","wrong_source_shape","missing_cast","mixed_storage"])
def test_native_widening_refuses_semantic_changes(bad):
    if find_tessera_opt() is None:pytest.skip("requires matching compiler")
    graph=module("div")
    fn=graph.functions[0]
    if bad=="cast_policy":fn.body[0].kwargs["numeric_policy"]={"rounding":"rtz"}
    if bad=="extra_cast":fn.body.insert(0,deepcopy(fn.body[0]));fn.body[0].result="unused"
    if bad=="wrong_source_shape":
        fn.args[0].ir_type=IRType("tensor<3x18xf16>",("3","18"),"fp16")
    if bad=="missing_cast":
        fn.body[-1].operands[0]="%a";fn.body=fn.body[1:]
    if bad=="mixed_storage":
        fn.args[1].ir_type=IRType("tensor<3x17xbf16>",("3","17"),"bf16")
        fn.body[1].operand_types=["tensor<3x17xbf16>"]
    with pytest.raises((ValueError,RuntimeError)):
        native.lower_math_graph(graph,"rocm_gfx1151")


@pytest.mark.parametrize("storage",["f16","bf16"])
def test_tracer_binds_positional_cast_dtype_as_attribute(storage):
    import numpy as np
    import tessera as ts
    from tessera.compiler.trace import trace
    dtype=np.float16 if storage=="f16" else pytest.importorskip("ml_dtypes").bfloat16
    def call(a):return ts.ops.sqrt(ts.ops.cast(a,"fp32"))
    graph=trace(call,np.ones((3,17),dtype))
    # Positional dtype is an attribute, not an additional tensor edge.
    assert [op.op_name for op in graph.body]==["tessera.cast","tessera.sqrt"]
    cast=graph.body[0]
    assert len(cast.operands)==1 and cast.kwargs["dtype"]=="fp32"
    assert graph.output_specs[0][1]=="f32"


@pytest.mark.parametrize("target",["x86","apple_gpu","nvidia"])
@pytest.mark.parametrize("kind",["sqrt","cumsum"])
def test_narrow_tile_storage_requires_rocm_ownership(target,kind):
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    tool=find_tessera_opt()
    if tool is None:pytest.skip("requires matching compiler")
    tile=native.lower_math_graph(module(kind),"rocm_gfx1151").tile_ir
    changed=tile.replace('tessera.target = "rocm"',f'tessera.target = "{target}"')
    assert changed!=tile
    with pytest.raises(RuntimeError,match="storage"):
        run_tessera_opt(tool,changed,"--canonicalize")


@pytest.mark.parametrize("kind",["sqrt","cumsum"])
def test_narrow_tile_rejects_changed_output_storage(kind):
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    tool=find_tessera_opt()
    if tool is None:pytest.skip("requires matching compiler")
    tile=native.lower_math_graph(module(kind),"rocm_gfx1151").tile_ir
    # Alter only the Tile operation; retain its native ownership record.
    lines=tile.splitlines()
    for i,line in enumerate(lines):
        if "tile.elementwise_kernel" in line or "tile.scan_kernel" in line:
            lines[i]=line.replace('output_storage = "f32"','output_storage = "f16"')
    changed="\n".join(lines)
    with pytest.raises(RuntimeError,match="storage"):
        run_tessera_opt(tool,changed,"--canonicalize")
