"""Host-independent scaled Graph typing and resident semantic guards."""
import pytest
from tessera.compiler.graph_ir import _infer_result_types,tensor_ir_type
from tessera.compiler.rocm_nvfp4_program import packed_consumer_attrs,_full_graph,package_traced_resident


def test_packed_scaled_frontend_result_is_bf16_mn():
    types=[tensor_ir_type((257,1024),"uint8"),tensor_ir_type((80,512),"uint8"),
           tensor_ir_type((257,),"fp32"),tensor_ir_type((33,80),"uint8")]
    assert str(_infer_result_types("tessera.scaled_matmul",types,packed_consumer_attrs(1024))[0])=="tensor<257x80xbf16>"
    types[3]=tensor_ir_type((80,33),"uint8")
    with pytest.raises(ValueError,match="operand shape/storage"):
        _infer_result_types("tessera.scaled_matmul",types,packed_consumer_attrs(1024))


def test_declared_generic_scaled_shape_and_nvfp4_nk_shape_are_distinct():
    generic=[tensor_ir_type((16,32),"fp16"),tensor_ir_type((32,48),"fp16"),
             tensor_ir_type((16,),"fp32"),tensor_ir_type((48,),"fp32")]
    assert str(_infer_result_types("tessera.scaled_matmul",generic,{})[0])=="tensor<16x48xf32>"
    packed=[tensor_ir_type((16,32),"uint8"),tensor_ir_type((48,16),"uint8"),
            generic[2],generic[3]]
    assert str(_infer_result_types("tessera.scaled_matmul",packed,
        {"physical_contract":"nvidia_sm120_nvfp4_blockscale_v1"})[0])=="tensor<16x48xf32>"


@pytest.mark.parametrize("change",["policy","edge","result","target","bindings","argument_layout","function_contract"])
def test_frontend_rejects_semantic_mutations_before_compiler(change,monkeypatch):
    from tessera.compiler import rocm_nvfp4_program as program
    module=_full_graph(128,32,256,[0,16,32],(0,1,2,3,4))
    fn=module.functions[0]
    if change=="policy":
        fn.body[2].kwargs["numeric_policy"]["execution_mode"]="exact_per_block"
    elif change=="edge":
        fn.body[2].operands[1]="%packed"
    elif change=="result":
        fn.return_values=["%packed"]
    elif change=="target":
        module.module_attrs["tessera.arch"]='"gfx1151"'
    elif change=="argument_layout":
        fn.args[0].layout="col_major"
    elif change=="function_contract":
        fn.fn_attrs["tessera.aliasing"]='"in_place"'
    else:
        fn.body[0].operands[2]=fn.body[0].operands[0]
    monkeypatch.setattr(program,"find_tessera_opt",lambda:pytest.fail("invalid Graph reached compiler"))
    with pytest.raises(ValueError):
        package_traced_resident(module)


def test_scaled_profile_does_not_promote_general_ad_or_batching():
    from tessera.compiler.primitive_coverage import coverage_for
    entry=coverage_for("scaled_matmul")
    for axis in ("vjp","jvp","transpose_rule","sharding_rule"):
        assert entry.contract_status[axis]=="planned"
    assert entry.contract_status["batching_rule"] == "partial"
    assert entry.contract_status["lowering_rule"] == "complete"
    assert entry.contract_status["math_semantics"]=="partial"


@pytest.mark.parametrize("attributes", [
    {"transposeA": True}, {"transposeB": True},
    {"transposeA": "false"}, {"transposeB": 0},
    {"batching": "shared_rhs_rows"},
])
def test_public_scaled_reference_checks_declared_transform_attributes(attributes):
    from tessera import ops
    import numpy as np

    with pytest.raises(ValueError, match="transpose"):
        ops.scaled_matmul(*(np.zeros((1, 1), dtype=np.uint8) for _ in range(4)),
                         **packed_consumer_attrs(256), **attributes)
