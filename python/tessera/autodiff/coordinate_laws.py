"""MSW-5: MSW-4 identities in physical orthonormal chart components."""

import numpy as np
from tessera.ga import Cl, MultivectorField, OrthogonalCoordinates
from tessera.ga.calculus import ext_deriv, codiff


def coordinate_identity_checks():
    from .laws import LawResult

    results = []
    for system in ("cylindrical", "spherical"):

        def fields(n):
            chart = OrthogonalCoordinates(
                system, tuple(tuple(np.linspace(a, b, n)) for a, b in ((1, 2), (0.5, 1.2), (0.2, 0.9)))
            )
            xyz = chart.cartesian_points()

            def field(data):
                return MultivectorField(data, Cl(3, 0), coordinates=chart)

            v = np.zeros((*chart.shape, 8))
            v[..., 0] = np.sum(xyz**2 * [1, 2, 3], axis=-1)
            scalar = field(v)
            v = np.zeros_like(v)
            v[..., [1, 2, 4]] = chart.vector_from_cartesian(xyz**2)
            return chart, xyz, scalar, field(v)

        c, xyz, f, v = fields(17)
        weight = c.volume_density()[..., None]
        rng = np.random.default_rng(41)
        a = rng.normal(size=f.values.shape)
        b = rng.normal(size=a.shape)
        for axis in range(3):
            for side in (slice(0, 3), slice(-3, None)):
                sl = [slice(None)] * 4
                sl[axis] = side
                a[tuple(sl)] = 0
                b[tuple(sl)] = 0
        a = f.with_values(a)
        b = f.with_values(b)
        lhs = np.sum(ext_deriv(a).values * b.values * weight)
        rhs = np.sum(a.values * codiff(b).values * weight)
        errors = {
            "d_squared_is_zero": np.max(np.abs(ext_deriv(ext_deriv(a)).values)),
            "codiff_squared_is_zero": np.max(np.abs(codiff(codiff(a)).values)),
            "weighted_stokes": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1),
        }
        for name, error in errors.items():
            results.append(
                LawResult(
                    system + "." + name,
                    "field_calculus",
                    "vector_identity",
                    "pass" if error < 1e-9 else "fail",
                    1,
                    float(error),
                )
            )
        gradient_errors = []
        product_errors = []
        for n in (17, 33):
            c, xyz, f, v = fields(n)
            inner = (slice(3, -3),) * 3
            g = c.vector_to_cartesian(f.gradient().values[..., [1, 2, 4]])
            gradient_errors.append(float(np.max(np.abs(g[inner] - (xyz * [2, 4, 6])[inner]))))
            product = v.with_values(v.values * f.values[..., 0, None])
            lhs = product.divergence().values[..., 0]
            rhs = f.values[..., 0] * v.divergence().values[..., 0] + np.sum(v.values * f.gradient().values, axis=-1)
            product_errors.append(float(np.max(np.abs(lhs[inner] - rhs[inner]))))
        for name, samples in [("analytic_gradient", gradient_errors), ("leibniz", product_errors)]:
            order = np.log2(samples[0] / samples[1])
            results.append(
                LawResult(
                    system + "." + name,
                    "field_calculus",
                    "vector_identity",
                    "pass" if order > 1.7 else "fail",
                    2,
                    samples[1],
                    f"observed order {order:.3f}; expected 2",
                )
            )
    return results
