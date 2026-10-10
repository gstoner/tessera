"""Continuous scaled products preserve Graph witness through native Schedule/Tile."""
import itertools
import json
import re

import pytest

from tessera.compiler.native_scaled_program import package_native_scaled_primal, package_native_scaled_jvp
from tests.unit.test_native_floating_scaled_adjoint import floating_source, floating_batch_source


def product_source(ta=False, tb=False, policy=None, jvp=False):
    text = (floating_source(ta, tb) if policy is None
            else floating_batch_source(policy, ta, tb))
    if jvp:
        return text.replace('tessera.autodiff = "reverse"', 'tessera.autodiff = "forward"')
    return re.sub(r' attributes \{tessera.autodiff = "reverse",\s*tessera.autodiff.wrt_indices = \[[^]]*\]\}', '', text)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("policy", [None, "shared_rhs_rows", "shared_lhs", "independent_rhs", "broadcast"])
@pytest.mark.parametrize("jvp", [False, True])
def test_continuous_product_has_native_structured_members(ta, tb, policy, jvp):
    source = product_source(ta, tb, policy, jvp)
    package = (package_native_scaled_jvp(source) if jvp else package_native_scaled_primal(source))
    program = json.loads(package.program_json)
    products = [step for step in program["steps"] if step["operation"] == "tessera.scaled_matmul"]
    assert len(products) == (5 if jvp else 1)
    assert all(step["lowering"] == "structured_f32_scaled_product" for step in products)
    assert "tessera.scaled_matmul" in program["root_ir"]
    assert "tensor.generate" not in program["root_ir"]
    for step in products:
        member = json.loads(package.members_json[step["step"]])
        count = program["buffers"][step["output"]]["elements"]
        assert member["scalars"] == [count]
        assert member["geometry"] == [(count+127)//128,1,1,128,1,1]
        assert package.images[step["step"]].startswith(b"\x7fELF")
    package.validate()


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("tb,encoded", tuple(itertools.product((False, True), repeat=2)))
def test_existing_encoded_primal_keeps_its_product_abi(tb, encoded):
    from tests.unit.test_native_independent_scaled_primal import source
    package = package_native_scaled_primal(source(((), (), (2, 3), ()), tb=tb, encoded=encoded))
    step = json.loads(package.program_json)["steps"][0]
    assert "lowering" not in step
    assert "continuous_contract" not in step
    assert json.loads(package.members_json[0])["scalars"] == [3, 5, 64]
    package.validate()
