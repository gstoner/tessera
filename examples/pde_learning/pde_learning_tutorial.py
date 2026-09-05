"""MSW-8: fixed-feature PINN and Kolmogorov learning, using Tessera only.

Trainable readouts over fixed smooth features avoid reverse-over-reverse AD.
Spatial derivatives are MSW-2 exact jets, precomputed for the frozen features;
no finite difference supplies the learning loss. A finite-difference PDE solve
is deliberately independent and is used only as a validation oracle.
"""

from __future__ import annotations
import numpy as np
import tessera as ts
from tessera import nn, optim
from tessera.autodiff import grad, jet_trace, laplacian_exact
from tessera.rng import RNGKey, uniform, normal

ops = ts.ops
FEATURES = tuple((i, j, k) for i in range(3) for j in range(3) for k in (1, 2))


def spatial_feature(point, i, j):
    x = ops.sum(ops.mul(point, np.array([1.0, 0.0])))
    y = ops.sum(ops.mul(point, np.array([0.0, 1.0])))
    x2 = ops.mul(x, x)
    y2 = ops.mul(y, y)
    bx = ops.sub(1.0, x2)
    by = ops.sub(1.0, y2)
    # Vanishing value and first two derivatives on the Dirichlet boundary.
    base = ops.mul(ops.mul(ops.mul(bx, bx), bx), ops.mul(ops.mul(by, by), by))
    for _ in range(i):
        base = ops.mul(base, x2)
    for _ in range(j):
        base = ops.mul(base, y2)
    return base


def design(points, times):
    spatial = np.empty((len(points), 9))
    lap = np.empty_like(spatial)
    for index, (i, j) in enumerate(((i, j) for i in range(3) for j in range(3))):
        fn = lambda x, i=i, j=j: spatial_feature(x, i, j)
        jf = jet_trace(fn)
        for row, p in enumerate(points):
            spatial[row, index] = fn(p)
            lap[row, index] = laplacian_exact(jf, p)
    V = np.stack([spatial[:, i * 3 + j] * times**k for i, j, k in FEATURES], axis=-1)
    Dt = np.stack([k * spatial[:, i * 3 + j] * times ** (k - 1) for i, j, k in FEATURES], axis=-1)
    L = np.stack([lap[:, i * 3 + j] * times**k for i, j, k in FEATURES], axis=-1)
    return V, Dt, L, 0.15 * spatial[:, 0], 0.15 * lap[:, 0]


def fit(loss, n, steps=1800, lr=0.02):
    theta = np.zeros((n, 1), dtype=np.float64)
    state = None
    derivative = grad(loss)
    before = float(loss(theta))
    for _ in range(steps):
        theta, state = optim.adam(theta, derivative(theta), state, lr=lr, compute_dtype="fp64", state_dtype="fp64")
    after = float(loss(theta))
    assert after < before * 0.03, (before, after)
    return theta, before, after


def numerical_reference(times, n=41, dt=0.0005):
    """Independent method-of-lines Allen-Cahn solve; Dirichlet zero boundary."""
    axis = np.linspace(-1, 1, n)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    h = axis[1] - axis[0]
    u = 0.15 * (1 - x * x) ** 3 * (1 - y * y) ** 3

    def rhs(v):
        result = np.zeros_like(v)
        lap = (v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4 * v[1:-1, 1:-1]) / h**2
        z = v[1:-1, 1:-1]
        result[1:-1, 1:-1] = 0.005 * lap + z - z**3
        return result

    out = {}
    t = 0.0
    for target in times:
        while t < target - 1e-14:
            step = min(dt, target - t)
            a = rhs(u)
            b = rhs(u + step * a / 2)
            c = rhs(u + step * b / 2)
            d = rhs(u + step * c)
            u = u + step * (a + 2 * b + 2 * c + d) / 6
            t += step
        out[target] = u.copy()
    return axis, out


def pinn():
    key = RNGKey.from_seed(805)
    samples = uniform(key, (96, 3), low=0.0, high=1.0, dtype="fp64")
    points = 2 * samples[:, :2] - 1
    times = 0.1 * samples[:, 2]
    V, Dt, L, u0, lap0 = design(points, times)

    def loss(w):
        u = ops.add(u0[:, None], nn.functional.linear(V, w))
        ut = nn.functional.linear(Dt, w)
        lap = ops.add(lap0[:, None], nn.functional.linear(L, w))
        residual = ops.add(ops.sub(ops.sub(ut, ops.mul(lap, 0.005)), u), ops.mul(ops.mul(u, u), u))
        return ops.mean(ops.mul(residual, residual))

    weights, before, after = fit(loss, len(FEATURES))
    checks = (0.025, 0.05, 0.1)
    axis, reference = numerical_reference(checks)
    _, fine_reference = numerical_reference(checks, n=81, dt=0.00025)
    xyz = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    # Values only at validation: derivatives were used for training, not targets.
    spatial = np.stack(
        [np.asarray([spatial_feature(p, i, j) for p in xyz]) for i in range(3) for j in range(3)], axis=-1
    )
    errors = {}
    for t in checks:
        v = np.stack([spatial[:, i * 3 + j] * t**k for i, j, k in FEATURES], axis=-1)
        prediction = (0.15 * spatial[:, 0] + (v @ weights).ravel()).reshape(reference[t].shape)
        error = float(np.sqrt(np.mean((prediction - reference[t]) ** 2)))
        oracle_error = float(np.max(np.abs(reference[t] - fine_reference[t][::2, ::2])))
        assert error < 0.002 and oracle_error < 0.00015, (t, error, oracle_error)
        errors[str(t)] = error
    return {"initial_loss": before, "final_loss": after, "rms_errors": errors}


def kolmogorov():
    """Learn heat-semigroup conditional expectations from Philox samples.

    Payoff |x|², generator nu*Delta in d=2. A quadratic feature network can
    realize the conditional expectation; the Monte Carlo targets are noisy.
    """
    key = RNGKey.from_seed(809)
    x = uniform(key, (128, 2), low=-1.0, high=1.0, dtype="fp64")
    noise = normal(RNGKey.from_seed(810), (128, 1024, 2), dtype="fp64")
    nu = 0.05
    t = 0.2
    target = np.mean(np.sum((x[:, None, :] + np.sqrt(2 * nu * t) * noise) ** 2, axis=-1), axis=1, keepdims=True)
    features = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1]])

    def loss(w):
        error = ops.sub(nn.functional.linear(features, w), target)
        return ops.mean(ops.mul(error, error))

    weights, before, after = fit(loss, 6, steps=1200)
    check = uniform(RNGKey.from_seed(811), (128, 2), low=-1.0, high=1.0, dtype="fp64")
    basis = np.column_stack(
        [np.ones(len(check)), check[:, 0], check[:, 1], check[:, 0] ** 2, check[:, 1] ** 2, check[:, 0] * check[:, 1]]
    )
    # Verify the generator with the exact MSW-2 path as well as the closed form.
    jf = jet_trace(lambda p: ops.sum(ops.mul(p, p)))
    assert abs(laplacian_exact(jf, check[0]) - 4.0) < 1e-12
    expected = np.sum(check**2, axis=-1) + 4 * nu * t
    error = float(np.sqrt(np.mean(((basis @ weights).ravel() - expected) ** 2)))
    assert error < 0.02, error
    return {"initial_loss": before, "final_loss": after, "rms_error": error}


def run():
    result = {"pinn_allen_cahn": pinn(), "kolmogorov_heat": kolmogorov()}
    for name, metrics in result.items():
        print(name, metrics)
    return result


if __name__ == "__main__":
    run()
