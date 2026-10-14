"""Native tensor gate carrier for continuous scaled products."""
import copy
import json
import os

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.rocm_typed_scaled_native import (
    supports_floating_scaled_primal, supports_floating_scaled_jvp, supports_scaled_reverse)
from tessera.compiler.native_scaled_program import (
    package_native_scaled_primal, package_native_scaled_jvp, package_native_scaled_vjp)


SOURCE = '''
def gated(a: ts.Tensor["M", "K", "fp32"],
          b: ts.Tensor["K", "N", "fp32"],
          sa: ts.Tensor["M", "G", "fp32"],
          sb: ts.Tensor["G", "C", "fp32"],
          gate: ts.Tensor["M", "N", "fp32"]):
    product = ts.ops.scaled_matmul(a, b, sa, sb,
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [4, 4], "format": "fp32"})
    return product * gate
'''


def case(mode=None, roles=("a", "b", "sa", "sb", "gate"), shape=(2, 9, 5), ta=False, tb=False):
    options = {} if mode is None else {"autodiff": mode, "wrt": roles}
    source = SOURCE.replace('numeric_policy=', f'transposeA={ta}, transposeB={tb}, numeric_policy=')
    if ta:
        source = source.replace('a: ts.Tensor["M", "K", "fp32"]', 'a: ts.Tensor["K", "M", "fp32"]')
    if tb:
        source = source.replace('b: ts.Tensor["K", "N", "fp32"]', 'b: ts.Tensor["N", "K", "fp32"]')
    owner = ts.from_text(source, target="rocm_gfx1201", **options)
    m,k,n = shape
    g,c = (k+3)//4,(n+3)//4
    rng = np.random.default_rng(94187)
    values = tuple(rng.uniform(-.5,.5,extent).astype(np.float32)
                   for extent in (((k,m) if ta else (m,k)),((n,k) if tb else (k,n)),(m,g),(g,c),(m,n)))
    directions = tuple(rng.uniform(-.5,.5,value.shape).astype(np.float32) for value in values)
    graph = owner._specialized_autodiff_module(values,{})
    return owner, graph, values, directions


@pytest.mark.parametrize("mode", [None, "forward", "reverse"])
def test_multiply_carrier_preserves_frontend_graph(mode):
    _,graph,_,_ = case(mode)
    before = copy.deepcopy(graph)
    assert supports_floating_scaled_primal(graph)
    assert supports_floating_scaled_jvp(graph,(0,1,2,3,4))
    assert supports_scaled_reverse(graph,(0,1,2,3,4))
    assert graph == before


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching native compiler required")
@pytest.mark.parametrize("mode", [None, "forward", "reverse"])
def test_multiply_carrier_owns_native_members_and_lifetimes(mode):
    _,graph,_,_ = case(mode)
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    builders = {None:package_native_scaled_primal, "forward":package_native_scaled_jvp,
                "reverse":package_native_scaled_vjp}
    package = builders[mode](graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest = json.loads(package.program_json)
    assert any(step["operation"] == "tessera.mul" for step in manifest["steps"])
    for step in manifest["steps"]:
        for source in step["inputs"]:
            assert source < step["output"]
            assert manifest["buffers"][source]["last_read"] >= step["step"]
    assert len(manifest["outputs"]) == (1 if mode is None else 2 if mode=="forward" else 5)
    package.validate()


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching native compiler required")
def test_reverse_root_collection_is_bounded_for_shared_product_dag():
    import subprocess

    _, _, values, _ = case()
    source = SOURCE.replace(
        "    return product * gate",
        "\n".join(["    product = product * product"] * 32) + "\n    return product * gate",
    )
    owner = ts.from_text(source, target="rocm_gfx1201", autodiff="reverse", wrt=("gate",))
    graph = owner._specialized_autodiff_module(values, {})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    result = subprocess.run(
        [os.environ["TESSERA_OPT"],
         "--pass-pipeline=builtin.module(tessera-autodiff-paired{export-scaled-transpose=true})"],
        input=graph.to_mlir(target="rocm_gfx1201", canonical=True),
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "tessera.mul" in result.stdout
    assert "tessera.autodiff.scaled_program_json" in result.stdout
