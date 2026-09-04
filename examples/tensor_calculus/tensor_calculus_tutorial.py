"""Run with PYTHONPATH=python python3 examples/tensor_calculus/tensor_calculus_tutorial.py."""

import itertools
import numpy as np
from tessera.compiler.contractions import Delta, Epsilon, contract, normal_form
from tessera.ga import Cl, MultivectorField, OrthogonalCoordinates
from tessera.autodiff.coordinate_laws import coordinate_identity_checks


def run():
    rng = np.random.default_rng(5)
    a = rng.normal(size=3)
    np.testing.assert_allclose(contract("ij,j->i", Delta(3), a), a)
    np.testing.assert_allclose(contract("ij,jk->ik", Delta(3), Delta(3)), np.eye(3))
    assert contract("ii->", Delta(3)) == 3
    eps = np.zeros((3, 3, 3))
    for p in itertools.permutations(range(3)):
        eps[p] = (-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
    before = np.einsum("ijk,imn->jkmn", eps, eps)
    after = contract("ijk,imn->jkmn", Epsilon(), Epsilon())
    np.testing.assert_array_equal(before, after)
    # Dense formula sums three products per output; contracted epsilon-delta
    # identity uses two products and one subtraction per output. Count scalar
    # multiplications in these explicit formulas (not backend instructions).
    dense_products = 3 * before.size
    delta_products = 2 * after.size
    np.testing.assert_array_equal(
        after, np.einsum("jm,kn->jkmn", np.eye(3), np.eye(3)) - np.einsum("jn,km->jkmn", np.eye(3), np.eye(3))
    )
    print(
        f"epsilon-delta identity: max error {np.max(np.abs(before - after)):.1e}; scalar products {dense_products} -> {delta_products}; rank-3 input storage {2 * eps.size} -> 0"
    )
    assert normal_form("ij,jk->ik") == normal_form("xy,yz->xz")
    print("alpha-normal contraction:", normal_form("xy,yz->xz").key)
    results = {}
    for system in ("cartesian", "cylindrical", "spherical"):
        c = OrthogonalCoordinates(
            system, tuple(tuple(np.linspace(a, b, 33)) for a, b in ((1, 2), (0.5, 1.2), (0.2, 0.9)))
        )
        xyz = c.cartesian_points()
        data = np.zeros((*c.shape, 8))
        data[..., 0] = np.sum(xyz**2 * [1, 2, 3], axis=-1)
        f = MultivectorField(data, Cl(3, 0), coordinates=c)
        inner = (slice(3, -3),) * 3
        grad = c.vector_to_cartesian(f.gradient().values[..., [1, 2, 4]])
        gradient_error = float(np.max(np.abs(grad[inner] - (xyz * [2, 4, 6])[inner])))
        laplacian_error = float(np.max(np.abs(f.laplacian().values[..., 0][inner] - 12)))
        data = np.zeros_like(data)
        physical = np.stack((-xyz[..., 1], xyz[..., 0], xyz[..., 2]), axis=-1)
        data[..., [1, 2, 4]] = c.vector_from_cartesian(physical)
        v = f.with_values(data)
        divergence_error = float(np.max(np.abs(v.divergence().values[..., 0][inner] - 1)))
        curl = c.vector_to_cartesian(v.curl().values[..., [1, 2, 4]])
        curl_error = float(np.max(np.abs(curl[inner] - [0, 0, 2])))
        assert max(gradient_error, laplacian_error, divergence_error, curl_error) < 0.025
        results[system] = (gradient_error, divergence_error, curl_error, laplacian_error)
        print(system, "max errors (grad/div/curl/Laplacian):", results[system])
    for result in coordinate_identity_checks():
        assert result.status == "pass", result
        print(result.op, result.status, result.detail or result.max_rel_residual)
    return results


if __name__ == "__main__":
    run()
