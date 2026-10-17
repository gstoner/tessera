"""Native package content mutations must invalidate warmed ABI certificates."""
from dataclasses import replace
import pytest
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tessera.compiler.rocm_quantized_mlp_program import PreparedQuantizedMLPProgram
from tests.unit.test_native_quantized_mlp_program import source

pytestmark=[pytest.mark.compiler_route,pytest.mark.compiler_rocm]


@pytest.fixture
def package(production_compiler,monkeypatch):
    monkeypatch.setenv('TESSERA_ROCM_PROGRAM_BINDING_CACHE','1')
    return package_quantized_product_graph(source())


def binder():
    return PreparedQuantizedMLPProgram.__new__(PreparedQuantizedMLPProgram)


def test_native_seven_step_abi_is_reused_for_equal_content(package):
    owner=binder();owner._bind_abi.cache_clear()
    first=owner._bind_abi(package)
    second=owner._bind_abi(replace(package))
    assert first is second
    assert owner._bind_abi.cache_info()['hits']==1
    assert len(first.steps)==7 and len(first.storage)==18


def test_nested_descriptor_mutation_cannot_use_cached_native_abi(package):
    owner=binder();owner._bind_abi(package)
    dims=package.gate.product.descriptor.provenance['shape']
    old=dims[0]
    try:
        dims[0]=old+1
        with pytest.raises(ValueError):package.validate()
        with pytest.raises(ValueError):owner._bind_abi(package)
    finally:
        dims[0]=old
    package.validate()


def test_recipe_shape_not_covered_by_old_identity_still_invalidates_cache(package):
    owner=binder();owner._bind_abi(package)
    recipe=replace(package.gate.product_recipe,shape=replace(package.gate.product_recipe.shape,m=4))
    changed=replace(package,gate=replace(package.gate,product_recipe=recipe))
    assert changed.binding_digest==package.binding_digest
    with pytest.raises(ValueError):owner._bind_abi(changed)


def test_signed_zero_metadata_cannot_reuse_positive_zero_certificate(package):
    native=package.gate.product
    descriptor=replace(native.descriptor,provenance={**native.descriptor.provenance,'zero_policy':0.0})
    gate=replace(package.gate,product=replace(native,descriptor=descriptor))
    gate=replace(gate,binding_digest=gate._identity())
    changed=replace(package,gate=gate)
    changed=replace(changed,binding_digest=changed._identity())
    owner=binder();owner._bind_abi(changed)
    changed.gate.product.descriptor.provenance['zero_policy']=-0.0
    with pytest.raises(ValueError):changed.validate()
    with pytest.raises(ValueError):owner._bind_abi(changed)
