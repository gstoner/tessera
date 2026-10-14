"""gfx1201 continuous scale waves with serial matrix gradients and mapped roots."""
import os
import pytest
from tests.device.rocm import test_floating_scaled_adjoint as proof
from tessera.compiler.native_scaled_program import package_native_scaled_vjp

pytestmark=pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")


@pytest.mark.parametrize("policy",["shared_rhs_rows","shared_lhs","independent_rhs","broadcast"])
@pytest.mark.parametrize("ta,tb",[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("roles",[(0,1,2,3),(3,1,0,2)])
def test_wave_scale_adjoint_with_shared_matrix_gradients(policy,ta,tb,roles,monkeypatch):
    def wave_package(source):
        return package_native_scaled_vjp(source,schedule="wave_per_scale_element")
    monkeypatch.setattr(proof,"package_native_scaled_vjp",wave_package)
    proof.test_floating_batch_adjoints_reduce_shared_operands(policy,ta,tb,roles)


@pytest.mark.parametrize("ta,tb",[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("placed",[False,True])
def test_wave_scale_adjoint_preserves_inverse_seed_and_generations(ta,tb,placed,monkeypatch):
    def wave_package(source):
        return package_native_scaled_vjp(source,schedule="wave_per_scale_element")
    monkeypatch.setattr(proof,"package_native_scaled_vjp",wave_package)
    proof.test_native_floating_adjoints_match_oracle_and_retain_outputs(
        ta,tb,(3,1,0,2),placed)
