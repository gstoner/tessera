"""Canonical native quantizer packaging and checked two-output ABI."""
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest
from tessera.compiler.rocm_block_quant_native import (
    BLOCK_QUANT_ABI, BlockQuantProgram, PreparedBlockQuantProgram,
    _frame, _validate_frame, lower_block_quant_text, package_block_quant_recipe,
)
from tessera.compiler.rocm_pipeline import (
    ROCMExecutablePipeline, ROCMInputLevel, generic_lane_families, promoted_families,
)
from tests.unit.test_native_k16_quant_graph import source


def test_quantizer_family_is_named_and_architecture_owned():
    assert "block_quant" in promoted_families("gfx1201")
    assert "block_quant" not in generic_lane_families()
    for arch in ("gfx1151","gfx1200","gfx1250"):
        assert "block_quant" not in promoted_families(arch)
        with pytest.raises(ValueError):
            ROCMExecutablePipeline(family="block_quant",arch=arch)
    for level in (ROCMInputLevel.GRAPH,ROCMInputLevel.DIRECTIVE):
        with pytest.raises(ValueError,match="Graph -> Schedule -> Tile"):
            ROCMExecutablePipeline(family="block_quant",arch="gfx1201",input_level=level)


@pytest.fixture
def package(production_compiler):
    recipe = lower_block_quant_text(source())
    return recipe,package_block_quant_recipe(recipe)


@pytest.mark.compiler_rocm
def test_quantizer_package_crosses_canonical_native_pipeline(package):
    recipe,native = package
    program,member = _frame(native.target_ir)
    assert 'tessera.pipeline.family = "block_quant"' in native.target_ir
    assert native.image.pipeline_name == "tessera-rocm-executable"
    assert native.image.payload.startswith(b"\x7fELF")
    assert native.descriptor.abi_id == BLOCK_QUANT_ABI
    assert [b.dtype for b in native.descriptor.buffers] == ["bf16","fp8_e4m3","fp32"]
    assert [b.alignment for b in native.descriptor.buffers] == [2,1,4]
    assert _frame(native.backend_ir) == (program,member)
    assert native.descriptor.provenance["graph_digest"] == hashlib.sha256(recipe.graph_ir.encode()).hexdigest()


@pytest.mark.compiler_rocm
@pytest.mark.parametrize("mutation",["bytes","output_roles","signed_shape","geometry","alias","lifetime","root","boolean_scalar"])
def test_quantizer_frame_refuses_resealed_corruption_before_provider(package,mutation):
    _,native = package
    program,member = map(deepcopy,_frame(native.target_ir))
    if mutation == "bytes":program["buffers"][1]["bytes"] += 1
    elif mutation == "output_roles":member["outputs"].reverse()
    elif mutation == "signed_shape":program["buffers"][0]["shape"][0] = True
    elif mutation == "geometry":member["geometry"][3] = 128
    elif mutation == "alias":member["bindings"][1] = member["bindings"][0]
    elif mutation == "lifetime":program["buffers"][1]["last_read"] = 0
    elif mutation == "root":program["root_ir"] += "\n// changed source"
    elif mutation == "boolean_scalar":member["scalars"] = [True]
    member["program_base64"] = base64.b64encode(json.dumps(program).encode()).decode()
    with pytest.raises(ValueError):
        _validate_frame(program,member)


@pytest.mark.compiler_rocm
def test_quantizer_binding_uses_native_output_widths_and_roles(package):
    recipe,native = package
    program,member = _frame(native.target_ir)
    bundle = BlockQuantProgram(json.dumps(program),(json.dumps(member),),(native.image.payload,),recipe.graph_ir)
    owner = PreparedBlockQuantProgram.__new__(PreparedBlockQuantProgram)
    binding = owner._bind_abi(bundle)
    assert binding.outputs == (1,2) and binding.argument_count == 1
    assert [b.bytes for b in binding.storage] == [192,96,24]
    step = binding.steps[0]
    assert step.input_count == 1 and step.output_count == 2 and step.scalar_count == 1
    assert list(step.outputs[:2]) == [1,2] and step.scalars[0] == 6
    with pytest.raises(ValueError,match="ownership"):
        replace(bundle,images=()).validate()
