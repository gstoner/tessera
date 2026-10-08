"""The legacy SM120 entry delegates tensor storage to native Schedule/Tile."""
from dataclasses import replace
import numpy as np
from itertools import permutations
import pytest
from tessera.compiler import scheduled_matmul, nvidia_native
from tessera import runtime as rt
from tests.unit.test_scheduled_matmul_consumers import _module, _dynamic_module, requires_nvidia_target_ir
from tests._support.nvidia import nvidia_cuda_host_ready

pytestmark = requires_nvidia_target_ir

def legacy_artifact(shape=(17, 19, 23), dtype="fp16", *, module=None, via_tiling=False):
    if module is None:
        module = _module(target="nvidia_sm120", shape=shape, dtype=dtype)
    canonical = scheduled_matmul.lower_scheduled_matmul(module, target="nvidia_sm120")
    # Feed the original frontend name into the legacy pass. The native pass
    # owns namespace normalization, not a prepared Python kernel symbol.
    original_name = module.functions[0].name
    import re
    graph_name = re.search(r"func.func @([^ (]+)", canonical.graph_ir)[1]
    source = canonical.graph_ir.replace("@"+graph_name+"(", "@"+original_name+"(", 1)
    if via_tiling:
        source = scheduled_matmul.run_tessera_opt(
            scheduled_matmul.find_tessera_opt(), source, "--tessera-tiling")
        assert source == canonical.schedule_ir
        assert "schedule.matmul" in source
        assert "tessera.canonical_k_step" not in source
        assert "tessera.k_reduction_accumulate" not in source
    tile = scheduled_matmul.run_tessera_opt(
        scheduled_matmul.find_tessera_opt(), source,
        "--tessera-tile-ir-lowering=sm=120")
    # The delegated native Schedule projection must be byte-for-byte the
    # canonical replay product, including its checked storage and lineage.
    assert tile == canonical.tile_ir
    projected = replace(canonical, tile_ir=tile)
    projected.validate()
    return projected

@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", [(16,16,8), (48,67,17), (64,256,64)])
def test_legacy_sm120_producer_reuses_typed_schedule_projection(shape,dtype):
    artifact = legacy_artifact(shape,dtype)
    assert "tile.view" in artifact.tile_ir
    assert "tile.fragment_pack" in artifact.tile_ir
    assert "tile.fragment_zero" in artifact.tile_ir
    assert "tile.fragment_unpack" in artifact.tile_ir
    assert "tile.async_copy" not in artifact.tile_ir
    assert "tessera.matmul" not in artifact.tile_ir



def test_legacy_sm120_delegation_rejects_other_architecture():
    canonical = scheduled_matmul.lower_scheduled_matmul(
        _module(target="nvidia_sm120"), target="nvidia_sm120")
    mismatched = canonical.graph_ir.replace('"nvidia_sm120"', '"rocm_gfx1201"')
    with pytest.raises(RuntimeError, match="explicit.*nvidia_sm120"):
        scheduled_matmul.run_tessera_opt(
            scheduled_matmul.find_tessera_opt(), mismatched,
            "--tessera-tile-ir-lowering=sm=120")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("output_dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("via_tiling", [False, True])
def test_legacy_entry_preserves_fused_and_dynamic_schedule_contract(dtype,dynamic,output_dtype,via_tiling):
    factory = _dynamic_module if dynamic else _module
    module = factory(target="nvidia_sm120", dtype=dtype, output_dtype=output_dtype,
                     bias=True, residual=True, activation="relu")
    projected = legacy_artifact(module=module,via_tiling=via_tiling)
    assert projected.dynamic_m is dynamic
    assert projected.output_dtype == output_dtype
    assert projected.bias_name == "bias"
    assert projected.residual_name == "residual"
    assert projected.activation == "relu"



@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_sm120_tiling_keeps_native_k_accumulator_lineage(dtype):
    projected = legacy_artifact((64,256,64),dtype,via_tiling=True)
    assert "scf.for" in projected.tile_ir
    assert "!tile.fragment" in projected.tile_ir
    assert "tile.fragment_zero" in projected.tile_ir
    assert "tile.fragment_unpack" in projected.tile_ir
    assert "tensor.extract_slice" not in projected.tile_ir
    assert "tile.async_copy" not in projected.tile_ir

def test_sm120_generic_tiling_knobs_do_not_silently_change_native_profile():
    canonical = scheduled_matmul.lower_scheduled_matmul(
        _module(target="nvidia_sm120"),target="nvidia_sm120")
    with pytest.raises(RuntimeError,match="native Schedule owns"):
        scheduled_matmul.run_tessera_opt(scheduled_matmul.find_tessera_opt(),
            canonical.graph_ir,"--tessera-tiling=tile-k=32")


def canonical_loop_artifact(module, *, mutate=None):
    canonical=scheduled_matmul.lower_scheduled_matmul(module,target="nvidia_sm120")
    # Force the target-neutral tiler, then restore the explicit owning target.
    generic=canonical.graph_ir.replace('"nvidia_sm120"','"generic_tensor_replay"')
    tiled=scheduled_matmul.run_tessera_opt(scheduled_matmul.find_tessera_opt(),
        generic,"--tessera-tiling=tile-m=16 tile-n=16 tile-k=16")
    tiled=tiled.replace('"generic_tensor_replay"','"nvidia_sm120"')
    assert "tessera.canonical_k_step" in tiled
    assert "tessera.k_reduction_accumulate" in tiled
    if mutate is not None:
        tiled=mutate(tiled)
    native=scheduled_matmul.run_tessera_opt(scheduled_matmul.find_tessera_opt(),
        tiled,"--tessera-tile-ir-lowering=sm=120")
    assert native==canonical.tile_ir
    artifact=replace(canonical,tile_ir=native)
    artifact.validate()
    return artifact

@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("shape",[(16,32,16),(17,35,19),(48,67,17)])
@pytest.mark.parametrize("fused",[False,True])
def test_sm120_canonical_tensor_loop_replays_into_native_storage(dtype,shape,fused):
    artifact=canonical_loop_artifact(_module(target="nvidia_sm120",dtype=dtype,
        shape=shape,bias=fused,residual=fused,activation="relu" if fused else "none"))
    assert "tile.view" in artifact.tile_ir
    assert "tile.fragment_pack" in artifact.tile_ir
    assert "tensor.extract_slice" not in artifact.tile_ir
    assert "tile.async_copy" not in artifact.tile_ir

def test_sm120_canonical_tensor_loop_checks_full_accumulator_replay():
    def nonzero(source):
        import re
        source,n=re.subn(r"dense<0[.]000000e[+]00> : tensor<32x32xf32>",
                         "dense<1.000000e+00> : tensor<32x32xf32>",source,count=1)
        assert n==1
        return source
    with pytest.raises(RuntimeError,match="complete semantic tiling replay"):
        canonical_loop_artifact(_module(target="nvidia_sm120",shape=(32,32,32)),mutate=nonzero)



@pytest.mark.parametrize("tamper",["loop_bound","extra_arithmetic"])
def test_sm120_canonical_tensor_loop_checks_bounds_and_return_lineage(tamper):
    def mutate(source):
        if tamper=="loop_bound":
            assert "arith.constant 32 : index" in source
            return source.replace("arith.constant 32 : index","arith.constant 16 : index",1)
        import re
        source,n=re.subn(r"return (%[a-zA-Z0-9_]+) : (tensor<32x32xf32>)",
            lambda m: "%changed = tessera.add "+m[1]+", "+m[1]+" : ("+m[2]+", "+m[2]+") -> "+m[2]+"\n    return %changed : "+m[2],
            source,count=1)
        assert n==1
        return source
    with pytest.raises(RuntimeError,match="complete semantic tiling replay"):
        canonical_loop_artifact(_module(target="nvidia_sm120",shape=(32,32,32)),mutate=mutate)


@pytest.mark.parametrize("wrong",["target","architecture"])
def test_sm120_canonical_tensor_loop_checks_owner_before_recovery(wrong):
    def mutate(source):
        if wrong=="target":
            return source.replace('"nvidia_sm120"','"rocm_gfx1201"')
        return source.replace('"sm_120"','"sm_90"')
    with pytest.raises(RuntimeError,match="explicit.*nvidia_sm120"):
        canonical_loop_artifact(_module(target="nvidia_sm120",shape=(32,32,32)),mutate=mutate)


@pytest.mark.parametrize("ambiguous",["duplicate_bias","multiple_activation"])
def test_sm120_canonical_tensor_loop_rejects_ambiguous_epilogue_before_building_graph(ambiguous):
    def mutate(source):
        if ambiguous=="duplicate_bias":
            return source.replace(") -> tensor<32x32xf32> {",
                ", %extra_bias0: tensor<32xf32>, %extra_bias1: tensor<32xf32>) -> tensor<32x32xf32> {",1)
        import re
        return re.sub(r"return (%[a-zA-Z0-9_]+) : tensor<32x32xf32>",
            lambda m: "%extra_act0 = tessera.relu "+m[1]+" : (tensor<32x32xf32>) -> tensor<32x32xf32>\n"
            "%extra_act1 = tessera.relu "+m[1]+" : (tensor<32x32xf32>) -> tensor<32x32xf32>\n"
            "return "+m[1]+" : tensor<32x32xf32>",source,count=1)
    message="epilogue operand roles" if ambiguous=="duplicate_bias" else "multiple activation sites"
    with pytest.raises(RuntimeError,match=message):
        canonical_loop_artifact(_module(target="nvidia_sm120",shape=(32,32,32)),mutate=mutate)

@pytest.mark.parametrize("fused", [False, True])
def test_registered_sm120_pipeline_preserves_complete_canonical_tensor_function(fused):
    canonical = scheduled_matmul.lower_scheduled_matmul(
        _module(target="nvidia_sm120", shape=(17,35,19), bias=fused,
                residual=fused, activation="relu" if fused else "none"),
        target="nvidia_sm120")
    tool = scheduled_matmul.find_tessera_opt()
    tensor = scheduled_matmul.run_tessera_opt(
        tool, canonical.graph_ir.replace('"nvidia_sm120"', '"generic_tensor_replay"'),
        "--tessera-tiling=tile-m=16 tile-n=16 tile-k=16"
    ).replace('"generic_tensor_replay"', '"nvidia_sm120"')
    full = scheduled_matmul.run_tessera_opt(tool, tensor, "--tessera-nvidia-pipeline-sm120")
    # Module metadata may differ; the executable function and checked ABI must
    # be exactly the native Schedule projection of the complete logical product.
    def body(source):
        return source[source.index("  llvm.func"):source.rindex("\n}")].strip()
    # Normalize harmless CSE/constant placement through the same MLIR pass.
    assert body(scheduled_matmul.run_tessera_opt(tool, full, "--canonicalize")) == body(
        scheduled_matmul.run_tessera_opt(tool, canonical.tile_ir, "--canonicalize"))

def test_graph_schedule_does_not_package_an_inner_canonical_k_step():
    canonical = scheduled_matmul.lower_scheduled_matmul(
        _module(target="nvidia_sm120", shape=(17,35,19)), target="nvidia_sm120")
    tool = scheduled_matmul.find_tessera_opt()
    tensor = scheduled_matmul.run_tessera_opt(
        tool, canonical.graph_ir.replace('"nvidia_sm120"', '"generic_tensor_replay"'),
        "--tessera-tiling=tile-m=16 tile-n=16 tile-k=16"
    ).replace('"generic_tensor_replay"', '"nvidia_sm120"')
    with pytest.raises(RuntimeError, match="complete logical contraction"):
        scheduled_matmul.run_tessera_opt(tool, tensor, "--tessera-graph-to-schedule")
    recovered = scheduled_matmul.run_tessera_opt(
        tool, tensor, "--tessera-tile-ir-lowering=sm=120 canonical-recovery-only=true")
    assert "tessera.canonical_k_step" not in recovered
    assert "schedule.matmul" not in recovered
    assert "tile.mma" not in recovered
    assert "tensor<17x35xf16>" in recovered
    assert "tensor<35x19xf16>" in recovered


@pytest.mark.parametrize("order", [(0,1), (1,0), *permutations(range(4))])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_canonical_recovery_preserves_frontend_operand_roles(order,dtype):
    fused=len(order)==4
    module=_module(target="nvidia_sm120",shape=(17,35,19),dtype=dtype,
        bias=fused,residual=fused,activation="relu" if fused else "none")
    function=module.functions[0]
    function.args=[function.args[i] for i in order]
    artifact=canonical_loop_artifact(module)
    assert "tile.view" in artifact.tile_ir
    assert "tile.fragment_pack" in artifact.tile_ir




def test_canonical_input_lineage_does_not_authorize_shifted_padding():
    def mutate(source):
        import re
        source,count=re.subn(
            r"(tensor.insert_slice %arg0 into %\w+)\[0, 0\]",
            r"\1[1, 0]",source,count=1)
        assert count==1
        return source
    with pytest.raises(RuntimeError,match="complete semantic tiling replay"):
        canonical_loop_artifact(
            _module(target="nvidia_sm120",shape=(17,35,19)),mutate=mutate)
