"""Physical Cartesian oracles, metric identities, and refusal boundaries (MSW-5)."""

import numpy as np
import pytest
from tessera.ga import Cl, MultivectorField, OrthogonalCoordinates
from tessera.ga.calculus import codiff, ext_deriv, hodge_star_field


def chart(system, n=33):
    ranges = ((1.0, 2.0), (0.5, 1.2), (0.2, 0.9))
    return OrthogonalCoordinates(system, tuple(tuple(np.linspace(a, b, n)) for a, b in ranges))


def scalar(c, data):
    v = np.zeros((*c.shape, 8))
    v[..., 0] = data
    return MultivectorField(v, Cl(3, 0), coordinates=c)


def vector(c, data):
    v = np.zeros((*c.shape, 8))
    v[..., [1, 2, 4]] = data
    return MultivectorField(v, Cl(3, 0), coordinates=c)


@pytest.mark.parametrize("system", ["cartesian", "cylindrical", "spherical"])
def test_same_physical_fields_in_three_charts(system):
    c = chart(system)
    xyz = c.cartesian_points()
    f = scalar(c, np.sum(xyz**2 * [1, 2, 3], axis=-1))
    inner = (slice(3, -3),) * 3
    grad = c.vector_to_cartesian(f.gradient().values[..., [1, 2, 4]])
    np.testing.assert_allclose(grad[inner], (xyz * [2, 4, 6])[inner], atol=0.006, rtol=0.001)
    np.testing.assert_allclose(f.laplacian().values[..., 0][inner], 12.0, atol=0.025)
    physical = np.stack((-xyz[..., 1], xyz[..., 0], xyz[..., 2]), axis=-1)
    v = vector(c, c.vector_from_cartesian(physical))
    np.testing.assert_allclose(v.divergence().values[..., 0][inner], 1.0, atol=0.004)
    curl = c.vector_to_cartesian(v.curl().values[..., [1, 2, 4]])
    np.testing.assert_allclose(curl[inner], np.broadcast_to([0.0, 0.0, 2.0], curl[inner].shape), atol=0.004)
    assert f.gradient().coordinates == c
    assert hodge_star_field(f).coordinates == c


@pytest.mark.parametrize("system", ["cylindrical", "spherical"])
def test_weighted_adjoint_and_nilpotence(system):
    c = chart(system, 17)
    rng = np.random.default_rng(72)
    a = rng.normal(size=(*c.shape, 8))
    b = rng.normal(size=a.shape)
    for axis in range(3):
        edge = [slice(None)] * 4
        edge[axis] = slice(0, 3)
        a[tuple(edge)] = b[tuple(edge)] = 0
        edge[axis] = slice(-3, None)
        a[tuple(edge)] = b[tuple(edge)] = 0
    a = MultivectorField(a, Cl(3, 0), coordinates=c)
    b = MultivectorField(b, Cl(3, 0), coordinates=c)
    weight = c.volume_density()[..., None]
    lhs = np.sum(ext_deriv(a).values * b.values * weight)
    rhs = np.sum(a.values * codiff(b).values * weight)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8, rtol=1e-12)
    np.testing.assert_allclose(ext_deriv(ext_deriv(a)).values, 0, atol=1e-10)
    np.testing.assert_allclose(codiff(codiff(b)).values, 0, atol=1e-10)


def test_coordinate_contract_refusals_and_default():
    v = np.zeros((3, 3, 3, 8))
    f = MultivectorField(v, Cl(3, 0))
    assert "Cartesian" in f.coordinate_reason
    with pytest.raises(ValueError, match="FIELD_COORDINATE_CONTRACT"):
        MultivectorField(v, Cl(3, 0), require_coordinates=True)
    for system, axes in [
        ("unknown", ((1, 2, 3),) * 3),
        ("spherical", ((0, 1, 2),) * 3),
        ("spherical", ((1, 2, 3), (0, 1, 2), (1, 2, 3))),
        ("cartesian", ((1, 2, 4),) * 3),
    ]:
        with pytest.raises(ValueError, match="FIELD_COORDINATE_CONTRACT"):
            OrthogonalCoordinates(system, axes)
    with pytest.raises(ValueError, match="FIELD_COORDINATE_CONTRACT"):
        MultivectorField(v, Cl(3, 0), coordinates=chart("spherical", 3), spacing=1.0)


def test_curvilinear_field_never_uses_cartesian_native_abi(monkeypatch):
    from tessera.ga.calculus import _try_apple_gpu_field_op_cl30_f32

    c = chart("spherical", 3)
    f = MultivectorField(np.zeros((*c.shape, 8), np.float32), Cl(3, 0), coordinates=c)
    assert _try_apple_gpu_field_op_cl30_f32(f, "tessera_apple_gpu_clifford_ext_deriv_cl30_f32") is None


def test_coordinate_law_family():
    from tessera.autodiff.coordinate_laws import coordinate_identity_checks

    results = coordinate_identity_checks()
    assert len(results) == 10
    assert all(r.status == "pass" for r in results), results
