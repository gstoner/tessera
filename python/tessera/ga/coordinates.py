"""MSW-5 orthogonal Euclidean charts for the sampled reference calculus.

Components are in the local orthonormal coframe, ordered (r, phi, z) for
cylindrical and (r, theta, phi) for spherical (theta is colatitude). Uniform
coordinate grids, central differences with one-sided order-two endpoints;
no periodic seam, pole, origin, general connection, or native backend implied.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .signature import TesseraAlgebraError


def coordinate_error(detail: str) -> TesseraAlgebraError:
    return TesseraAlgebraError(f"FIELD_COORDINATE_CONTRACT: {detail}")


@dataclass(frozen=True)
class OrthogonalCoordinates:
    """Declared chart and sampled coordinate axes; metric is derived, not guessed."""

    system: str
    axes: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if self.system not in {"cartesian", "cylindrical", "spherical"}:
            raise coordinate_error(f"unknown coordinate system {self.system!r}")
        axes = tuple(tuple(float(x) for x in axis) for axis in self.axes)
        if not axes or (self.system != "cartesian" and len(axes) != 3):
            raise coordinate_error("cylindrical/spherical charts require three axes")
        for axis in axes:
            a = np.asarray(axis)
            if len(a) < 3 or not np.all(np.isfinite(a)):
                raise coordinate_error("each axis needs at least three finite samples")
            steps = np.diff(a)
            if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-14):
                raise coordinate_error("coordinate axes must be increasing and uniform")
        if self.system != "cartesian" and min(axes[0]) <= 0:
            raise coordinate_error("the radial chart excludes r <= 0")
        if self.system == "spherical" and not all(0 < t < np.pi for t in axes[1]):
            raise coordinate_error("spherical colatitude must lie strictly between 0 and pi")
        object.__setattr__(self, "axes", axes)

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(axis[1] - axis[0] for axis in self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis) for axis in self.axes)

    def scale_factors(self) -> np.ndarray:
        """h_i, so ds² = sum(h_i² dq_i²) and sqrt(g) = product(h_i)."""
        q = np.meshgrid(*self.axes, indexing="ij")
        h = np.ones((*self.shape, len(self.axes)), dtype=np.float64)
        if self.system == "cylindrical":
            h[..., 1] = q[0]
        elif self.system == "spherical":
            h[..., 1] = q[0]
            h[..., 2] = q[0] * np.sin(q[1])
        return h

    def volume_density(self) -> np.ndarray:
        return np.prod(self.scale_factors(), axis=-1)

    def cartesian_points(self) -> np.ndarray:
        q = np.meshgrid(*self.axes, indexing="ij")
        if self.system == "cartesian":
            return np.stack(q, axis=-1)
        r = q[0]
        if self.system == "cylindrical":
            phi, z = q[1:]
            return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)
        theta, phi = q[1:]
        return np.stack((r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)), axis=-1)

    def frame(self) -> np.ndarray:
        """Columns are the local orthonormal basis vectors in Cartesian components."""
        out = np.broadcast_to(np.eye(len(self.axes)), (*self.shape, len(self.axes), len(self.axes))).copy()
        if self.system == "cartesian":
            return out
        q = np.meshgrid(*self.axes, indexing="ij")
        if self.system == "cylindrical":
            c, s = np.cos(q[1]), np.sin(q[1])
            out[..., 0, 0], out[..., 1, 0] = c, s
            out[..., 0, 1], out[..., 1, 1] = -s, c
        else:
            ct, st, cp, sp = np.cos(q[1]), np.sin(q[1]), np.cos(q[2]), np.sin(q[2])
            out[..., :, 0] = np.stack((st * cp, st * sp, ct), axis=-1)
            out[..., :, 1] = np.stack((ct * cp, ct * sp, -st), axis=-1)
            out[..., :, 2] = np.stack((-sp, cp, np.zeros_like(cp)), axis=-1)
        return out

    def vector_from_cartesian(self, vector: np.ndarray) -> np.ndarray:
        return np.einsum("...ji,...j->...i", self.frame(), vector)

    def vector_to_cartesian(self, vector: np.ndarray) -> np.ndarray:
        return np.einsum("...ij,...j->...i", self.frame(), vector)
