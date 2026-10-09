"""Native compiler policy and ancestry checks; no GPU execution claims."""
from copy import deepcopy
import pytest
from tessera.compiler.graph_ir import tensor_ir_type
from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
from tessera.compiler.scheduled_matmul import run_tessera_opt,find_tessera_opt
from tests.unit.test_scheduled_kernel_consumers import _module

@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("dtype",["fp16","bf16","fp32"])
@pytest.mark.parametrize("columns",[1,35,255,256,513,4097])
@pytest.mark.parametrize("override",[None,"serial","cooperative_128"])
def test_native_softmax_row_policy_and_explicit_override(dtype,columns,override):
    graph=_module(family="softmax",target="nvidia_sm120")
    fn=graph.functions[0];typ=tensor_ir_type((3,columns),dtype)
    fn.args[0].ir_type=typ;fn.result_types=[typ]
    fn.body[0].operand_types=[str(typ)];fn.body[0].result_type=str(typ)
    fn.body[0].inferred_type=typ
    before=deepcopy(graph)
    artifact=lower_scheduled_kernel(graph,target="nvidia_sm120",schedule=override)
    expected=override or ("cooperative_128" if columns>=256 else "serial")
    assert artifact.schedule==expected
    # ODS elides the default serial attribute in assembly.
    assert ('schedule = "cooperative_128"' in artifact.tile_ir)==(expected=="cooperative_128")
    assert artifact.function_name=="tessera_tile_softmax_"+artifact.storage+(
        "_cooperative_128" if expected=="cooperative_128" else "")
    assert graph==before
    assert run_tessera_opt(find_tessera_opt(),artifact.schedule_ir,"--tessera-schedule-to-tile")==artifact.tile_ir

@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("target",["x86","rocm_gfx1151","rocm_gfx1201","apple_gpu"])
def test_sibling_softmax_policy_is_not_transferred(target):
    graph=_module(family="softmax",target=target)
    fn=graph.functions[0];typ=tensor_ir_type((3,513),"fp32")
    fn.args[0].ir_type=typ;fn.result_types=[typ]
    fn.body[0].operand_types=[str(typ)];fn.body[0].result_type=str(typ);fn.body[0].inferred_type=typ
    artifact=lower_scheduled_kernel(graph,target=target)
    assert artifact.schedule=="serial"
    assert "cooperative_128" not in artifact.tile_ir
