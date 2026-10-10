"""Continuous producer outputs retain native SSA and buffer lifetimes."""
import copy
import json
import os

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.rocm_typed_scaled_native import (
    supports_floating_scaled_primal, supports_floating_scaled_jvp,
    supports_scaled_reverse)
from tessera.compiler.native_scaled_program import package_native_scaled_primal


def chained(a: ts.Tensor["M", "K", "fp32"],  # noqa: F821 - Tessera symbolic dimensions/dtype, not Python forward references
            b: ts.Tensor["K", "N", "fp32"],  # noqa: F821 - Tessera symbolic dimensions/dtype, not Python forward references
            sa: ts.Tensor["M", "G", "fp32"],  # noqa: F821 - Tessera symbolic dimensions/dtype, not Python forward references
            sb: ts.Tensor["G", "C", "fp32"],  # noqa: F821 - Tessera symbolic dimensions/dtype, not Python forward references
            c: ts.Tensor["N", "P", "fp32"],  # noqa: F821 - Tessera symbolic dimensions/dtype, not Python forward references
            sc: ts.Tensor["M", "J", "fp32"],  # noqa: F821 - Tessera symbolic dimensions/dtype, not Python forward references
            sd: ts.Tensor["J", "D", "fp32"]):  # noqa: F821 - Tessera symbolic dimensions/dtype, not Python forward references
    first = ts.ops.scaled_matmul(a, b, sa, sb,
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [4, 4], "format": "fp32"})
    return ts.ops.scaled_matmul(first, c, sc, sd,
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [4, 4], "format": "fp32"})


def case(shape=(2,9,5,3)):
    m,k,n,p=shape
    g,j,d=(k+3)//4,(n+3)//4,(p+3)//4
    rng = np.random.default_rng(19043)
    values = tuple(rng.uniform(-.5, .5, extent).astype(np.float32)
                   for extent in ((m,k),(k,n),(m,g),(g,j),(n,p),(m,j),(j,d)))
    owner = ts.jit(target="rocm_gfx1201")(chained)
    graph = owner._specialized_autodiff_module(values,{})
    return owner, graph, values


def test_chained_continuous_admission_preserves_graph_and_reverse_boundary():
    _, graph, _ = case()
    before = copy.deepcopy(graph)
    assert supports_floating_scaled_primal(graph)
    assert supports_floating_scaled_jvp(graph, (0,1,2,3,4,5,6))
    assert supports_scaled_reverse(graph, (0,1,2,3,4,5,6))
    assert graph == before


@pytest.mark.parametrize("kind", ["policy", "shape", "forward_reference"])
def test_chained_continuous_rejects_invalid_consumer(kind):
    _, graph, _ = case()
    consumer = graph.functions[0].body[1]
    if kind == "policy":
        consumer.kwargs["numeric_policy"]["execution_mode"] = "approximate"
    elif kind == "shape":
        graph.functions[0].args[4].ir_type = graph.functions[0].args[0].ir_type
    else:
        consumer.operands[0] = "%unwritten"
    assert not supports_floating_scaled_primal(graph)


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching compiler required")
def test_chained_native_primal_owns_producer_buffer_until_consumer():
    _, graph, _ = case()
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    package = package_native_scaled_primal(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest = json.loads(package.program_json)
    assert len(manifest["steps"]) == 2
    producer, consumer = manifest["steps"]
    assert producer["output"] in consumer["inputs"]
    scratch = manifest["buffers"][producer["output"]]
    assert scratch["first_write"] == 0
    assert scratch["last_read"] == 1
    assert producer["output"] not in manifest["outputs"]
    assert all(step["lowering"] == "structured_f32_scaled_product"
               for step in manifest["steps"])
    package.validate()

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching compiler required")
def test_chained_native_jvp_materializes_dependency_members():
    from tessera.compiler.native_scaled_program import package_native_scaled_jvp
    owner, _, values = case()
    differentiated = ts.jit(target="rocm_gfx1201", autodiff="forward", wrt=("a","c"))(chained)
    graph = differentiated._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    package = package_native_scaled_jvp(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest = json.loads(package.program_json)
    assert len(manifest["outputs"]) == 2
    assert any(any(i >= manifest["argument_count"] for i in step["inputs"])
               for step in manifest["steps"])
    package.validate()

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching compiler required")
@pytest.mark.parametrize("roles",[("a",),("c",),("a","b","sa","sb","c","sc","sd")])
def test_chained_native_reverse_retains_required_residuals_and_seed_lineage(roles):
    from tessera.compiler.native_scaled_program import package_native_scaled_vjp
    _, _, values = case()
    owner = ts.jit(target="rocm_gfx1201", autodiff="reverse", wrt=roles)(chained)
    graph = owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    package = package_native_scaled_vjp(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest = json.loads(package.program_json)
    assert manifest["dependency_policy"] == "native_recompute_prefix_v1"
    assert manifest["gradient_roles"] == list(owner.differentiation_request.wrt_indices)
    reductions = [step for step in manifest["steps"] if step["operation"] == "tensor.generate"]
    assert all(step["seed_input"] in step["inputs"] for step in reductions)
    if roles == ("a",):
        assert any(step["seed_input"] >= manifest["argument_count"] for step in reductions)
        assert not any(step["operation"] == "tessera.scaled_matmul" for step in manifest["steps"])
    else:
        assert any(step["operation"] == "tessera.scaled_matmul" for step in manifest["steps"])
    package.validate()

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching compiler required")
@pytest.mark.parametrize("damage",["seed","dependency_kind","output_role","lifetime"])
def test_chained_reverse_rejects_forged_dependency_contract(damage):
    from tests.unit.test_composed_scaled_vjp import forge
    from tessera.compiler.native_scaled_program import package_native_scaled_vjp
    _, _, values = case()
    owner = ts.jit(target="rocm_gfx1201", autodiff="reverse", wrt=("a","c"))(chained)
    graph = owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    package = package_native_scaled_vjp(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest = json.loads(package.program_json)
    reduction = next(step for step in manifest["steps"] if step["operation"]=="tensor.generate")
    if damage=="seed":
        reduction["seed_input"]=next(i for i in reduction["inputs"] if i!=reduction["seed_input"])
    elif damage=="dependency_kind":
        reduction["dependency_kind"]="residual"
    elif damage=="output_role":
        manifest["steps"][manifest["outputs"][0]-manifest["argument_count"]].pop("gradient_argument")
    else:
        residual=next(step for step in manifest["steps"] if step["dependency_kind"]=="residual")
        manifest["buffers"][residual["output"]]["last_read"]=0
    with pytest.raises(ValueError):
        forge(package,manifest).validate()
