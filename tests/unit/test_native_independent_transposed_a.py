"""Native transposed-A program image, ABI and orientation corruption tests."""
import copy,itertools,json
from dataclasses import replace
import pytest
from tests.unit.test_public_independent_scaled_primal import case
from tessera.compiler.native_scaled_program import package_native_scaled_jvp
from tessera.compiler.rocm_typed_scaled_native import lower_typed_scaled,package_typed_scaled

@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
@pytest.mark.parametrize("tb,encoded",tuple(itertools.product((False,True),repeat=2)))
def test_transposed_a_native_primal_image_and_serialized_orientation(tb,encoded):
    _,owner,values,_=case(4,tb,encoded,ta=True)
    graph=owner._specialized_autodiff_module(values,{})
    lowered=lower_typed_scaled(graph)
    assert "transposeA = true" in lowered.schedule_ir
    assert "transposeA = true" in lowered.tile_ir
    package=package_typed_scaled(graph,lowered,pipeline_name="tessera-lower-to-rocm")
    from tessera.compiler.native_scaled_program import NativeScaledProgram
    native=NativeScaledProgram.from_manifest(package.descriptor.provenance["native_scaled_primal_program"])
    body=json.loads(native.program_json)
    assert body["steps"][0]["transposeA"] is True
    assert package.image.payload==native.images[0]
    import os,subprocess
    target=package.target_ir
    assert "transposeA = true" in target
    forged=target.replace("transposeA = true","transposeA = false")
    result=subprocess.run([os.environ["TESSERA_OPT"],
        "--pass-pipeline=builtin.module(tessera-rocm-executable{family=matmul input=directive output=binary arch=gfx1201})"],
        input=forged,text=True,capture_output=True,timeout=180)
    assert result.returncode!=0
    assert "independent scaled batch metadata differs" in result.stderr
    for flag in (False,1,"true"):
        corrupt=copy.deepcopy(body)
        corrupt["steps"][0]["transposeA"]=flag
        with pytest.raises(ValueError):
            replace(native,program_json=json.dumps(corrupt)).validate()

@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
@pytest.mark.parametrize("tb",[False,True])
def test_transposed_a_native_jvp_preserves_each_product_orientation(tb):
    _,owner,values,_=case(12,tb,jvp=True,ta=True)
    graph=owner._specialized_autodiff_module(values[:4],{})
    paired=owner._compile_jvp_module(graph)
    assert "transposeA = true" in paired
    # Public native packaging consumes source semantics and emits the paired pass.
    native_graph=replace(graph,module_attrs={**graph.module_attrs,"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    package=package_native_scaled_jvp(native_graph.to_mlir(target="rocm_gfx1201"))
    body=json.loads(package.program_json)
    products=[step for step in body["steps"] if step["operation"]=="tessera.scaled_matmul"]
    assert len(products)==3
    assert all(step["transposeA"] is True for step in products)
