import itertools
import numpy as np
import pytest
from tessera.compiler.contractions import Delta, Epsilon, contract, normal_form


def epsilon():
    e = np.zeros((3, 3, 3))
    for p in itertools.permutations(range(3)):
        e[p] = (-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
    return e


def test_alpha_normalization_and_output_order():
    assert normal_form("ij,jk->ik") == normal_form("xy,yz->xz")
    assert normal_form("ji") != normal_form("ij")  # implicit outputs transpose ji
    assert normal_form("...ij,...jk->...ik") == normal_form("...ab,...bc->...ac")
    a = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(contract("ji", a), np.einsum("ji", a))


@pytest.mark.parametrize(
    "spec,identities",
    [
        ("ij,j->i", (Delta(3), None)),
        ("ij,jk->ik", (Delta(3), Delta(3))),
        ("ii->", (Delta(3),)),
        ("ii->i", (Delta(3),)),
        ("ij,ij->", (Delta(3), None)),
        ("ij,ij->ij", (Delta(3), None)),
        ("ijk,imn->jkmn", (Epsilon(), Epsilon())),
        ("ijk,ijk->", (Epsilon(), Epsilon())),
        ("ijk,kji->", (Epsilon(), Epsilon())),
        ("ijk,j,k->i", (Epsilon(), None, None)),
        ("kji,j,k->i", (Epsilon(), None, None)),
        ("iik->k", (Epsilon(),)),
    ],
)
def test_identity_rewrites_against_dense_randomized_oracle(spec, identities):
    rng = np.random.default_rng(9)
    for _ in range(8):
        terms = spec.split("->")[0].split(",")
        args = [rng.normal(size=(3,) * len(t)) if v is None else v for t, v in zip(terms, identities)]
        dense = [
            np.eye(v.dimension) if isinstance(v, Delta) else epsilon() if isinstance(v, Epsilon) else v for v in args
        ]
        np.testing.assert_allclose(contract(spec, *args), np.einsum(spec, *dense), atol=1e-12)


def test_dense_einsum_surface_and_derivative():
    import tessera as ts
    from tessera.autodiff import grad

    a = np.arange(6, dtype=float).reshape(2, 3)
    b = np.ones((3, 4))
    np.testing.assert_allclose(ts.ops.einsum("xy,yz->xz", a, b), a @ b)
    f = lambda x: ts.ops.reduce(ts.ops.einsum("xy,yz->xz", x, b))
    np.testing.assert_allclose(grad(f)(a), np.full_like(a, 4))


def test_identity_dimension_mismatch_refused():
    with pytest.raises(ValueError, match="extents"):
        contract("ij,j->i", Delta(3), np.zeros(4))


def test_delta_replacement_elides_the_copy():
    values = np.arange(3.0)
    result = contract("ij,j->i", Delta(3), values)
    assert np.shares_memory(values, result)


@pytest.mark.parametrize("spec", ["ij->", "ij->i", "ij->j"])
def test_free_and_fully_summed_delta(spec):
    np.testing.assert_array_equal(contract(spec, Delta(3)), np.einsum(spec, np.eye(3)))


def test_epsilon_pair_never_materializes_rank_three_operands(monkeypatch):
    original = np.einsum
    shapes = []

    def observed(spec, *values, **kwargs):
        shapes.extend(np.shape(v) for v in values)
        return original(spec, *values, **kwargs)

    monkeypatch.setattr(np, "einsum", observed)
    contract("ijk,imn->jkmn", Epsilon(), Epsilon())
    assert shapes and all(len(shape) <= 2 for shape in shapes)
