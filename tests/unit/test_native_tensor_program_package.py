"""Original-source replay and immutable native member/image packaging."""
from dataclasses import replace
import hashlib

import pytest
import numpy as np

import tessera as ts
from tessera.compiler.native_tensor_program import compile_native_tensor_program
from tests.unit.test_native_tensor_program_pipeline import (
    traced_source, norm_product, relu_product, both_products, reversed_product,
)

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


@pytest.fixture(scope="module")
def package():
    return compile_native_tensor_program(traced_source(norm_product, "f16"))


@pytest.mark.parametrize("function,storage", [
    (norm_product, "bf16"), (relu_product, "f16"), (relu_product, "bf16"),
    (both_products, "f16"), (both_products, "bf16"), (reversed_product, "f16")])
def test_original_source_compiles_to_complete_native_target(function, storage):
    p = compile_native_tensor_program(traced_source(function, storage))
    assert p is not None
    assert len(p.members) == len(p.images) == len(p.frame["stages"])
    assert "tessera_nvidia.program_invoke " in p.target_ir
    assert "tile.program_invoke " not in p.target_ir
    if function is norm_product:
        assert 'accum = "fp32"' in p.target_ir
    for stage, member, image in zip(p.frame["stages"], p.members, p.images, strict=True):
        assert "program_invoke " not in member
        assert (".entry " + stage["entry"]).encode() in image
        assert b".target sm_120a" in image
    seal = hashlib.sha256(p.source_ir.strip().encode()).hexdigest()
    assert f'tessera.native.tensor_program_source_sha256 = "{seal}"' in p.target_ir


@pytest.mark.parametrize("field", ["source", "target", "member", "image", "arity", "compiler"])
def test_replacing_fields_rechecks_source_program_member_and_image(package, field):
    assert package is not None
    changes = {
        "source": {"source_ir": package.source_ir.replace("17x32", "18x32")},
        "target": {"target_ir": package.target_ir.replace("arith.mulf ", "arith.addf ", 1)},
        "member": {"members": (package.members[0] + "\n// changed\n", *package.members[1:])},
        "image": {"images": (package.images[0] + b"\n// changed\n", *package.images[1:])},
        "arity": {"images": package.images[:-1]},
        "compiler": {"compiler_identity": "0" * 64},
    }
    with pytest.raises((ValueError, RuntimeError)):
        replace(package, **changes[field])


def test_frame_inspection_cannot_change_verified_program(package):
    before = package.frame
    altered = package.frame
    altered["stages"][0]["geometry"][0] += 1
    altered["buffers"][0]["shape"][0] += 1
    assert package.frame == before


def test_declined_fused_source_is_preserved():
    source = traced_source(relu_product, "f16")
    source = source.replace('output_dtype = "fp32"', 'activation = "relu", output_dtype = "fp32"')
    assert compile_native_tensor_program(source) is None
    assert 'activation = "relu"' in source


def test_public_compile_method_preserves_explicit_accumulation_policy():
    fn = ts.jit(target="nvidia_sm120")(norm_product)
    p = fn.compile_native_tensor_program(
        np.ones((17, 32), np.float16), np.ones((32, 11), np.float16))
    assert "tessera.rmsnorm" in p.source_ir
    assert 'accum = "fp32"' in p.target_ir
    assert p.frame["stages"][0]["inputs"] == [0]
    assert p.frame["stages"][1]["inputs"] == [2, 1]


def test_frontend_target_projection_preserves_caller_owned_graph():
    fn = ts.jit(target="nvidia_sm120")(norm_product)
    inputs = np.ones((17, 32), np.float16), np.ones((32, 11), np.float16)
    graph = fn._traced_autodiff_module(inputs, {})
    before = graph.to_mlir(target="nvidia_sm120", canonical=True)
    p = compile_native_tensor_program(graph)
    assert p is not None
    assert graph.to_mlir(target="nvidia_sm120", canonical=True) == before
