"""GA5 — Manifolds for integration.

A Manifold provides two things:
    1. ``sample_points()`` — an ``(N, ambient_dim)`` array of points.
    2. ``weights()`` — an ``(N,)`` array of integration weights such
       that ``Σ_i weights[i] · f(points[i]) ≈ ∫_M f · vol``.

The ``Manifold`` ABC also exposes an optional ``boundary()`` for
Stokes-style tests; closed manifolds return ``None``.

V1 concrete manifolds (per Q5 scope-lock recommendation):

    Euclidean    n-dimensional axis-aligned box on a uniform grid
    Sphere       n=2 only — vertices on the unit 2-sphere via spherical
                 Fibonacci with uniform area weighting
    SOn          minimal stub — n=3 only, gives a parameterized rotor
                 sampling for tests that need it

Other manifolds (SE(n), SU(n), hyperbolic H^n) are deferred per Q5.
"""

from __future__ import annotations

import abc
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


class Manifold(abc.ABC):
    """Abstract base class for discretized manifolds.

    Subclasses must implement ``sample_points()`` and ``weights()``.
    """

    @abc.abstractmethod
    def sample_points(self) -> np.ndarray:
        """An ``(N, ambient_dim)`` array of point coordinates."""

    @abc.abstractmethod
    def weights(self) -> np.ndarray:
        """An ``(N,)`` array of integration weights (volume/area per point)."""

    def boundary(self) -> Optional["Manifold"]:
        """The boundary manifold, or ``None`` for closed manifolds."""
        return None

    @property
    def n_points(self) -> int:
        return self.sample_points().shape[0]


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------

class Euclidean(Manifold):
    """Axis-aligned box ``Π_i [a_i, b_i]`` with a uniform grid.

    The grid has ``resolution`` cells per axis (or a per-axis tuple).
    ``sample_points()`` returns the cell-center coordinates;
    ``weights()`` is the cell volume (constant for a uniform grid).
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        *,
        resolution: int | Sequence[int] = 16,
    ) -> None:
        self._bounds = tuple((float(a), float(b)) for a, b in bounds)
        self._dim = len(self._bounds)
        if isinstance(resolution, int):
            self._resolution = tuple(int(resolution) for _ in range(self._dim))
        else:
            self._resolution = tuple(int(r) for r in resolution)
        if len(self._resolution) != self._dim:
            raise ValueError(
                f"resolution must have {self._dim} entries; got {len(self._resolution)}."
            )
        if any(r < 2 for r in self._resolution):
            raise ValueError("Euclidean resolution must be >= 2 per axis.")
        if any(b <= a for a, b in self._bounds):
            raise ValueError(f"each bound must have b > a; got {self._bounds}.")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        return self._bounds

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._resolution

    @property
    def spacing(self) -> Tuple[float, ...]:
        return tuple(
            (b - a) / (r - 1) for (a, b), r in zip(self._bounds, self._resolution)
        )

    def grid_points(self) -> List[np.ndarray]:
        """Per-axis 1-D coordinate arrays (cell-center / corner-aligned)."""
        return [
            np.linspace(a, b, r) for (a, b), r in zip(self._bounds, self._resolution)
        ]

    def sample_points(self) -> np.ndarray:
        mesh = np.meshgrid(*self.grid_points(), indexing="ij")
        flat = np.stack([m.ravel() for m in mesh], axis=-1)
        return flat

    def weights(self) -> np.ndarray:
        cell_volume = float(np.prod(self.spacing))
        return np.full(int(np.prod(self._resolution)), cell_volume, dtype=np.float64)


# ---------------------------------------------------------------------------
# Sphere — the 2-sphere via spherical Fibonacci
# ---------------------------------------------------------------------------

class Sphere(Manifold):
    """Unit ``n``-sphere with ``n_vertices`` sample points.

    V1 supports ``n=2`` only (the surface of the unit ball in ℝ³).
    Vertices are placed via the spherical Fibonacci lattice — uniform
    enough that integral estimators converge quickly. Each vertex
    carries an area weight of ``4π / n_vertices``.

    The 2-sphere is closed: ``boundary()`` returns ``None``. This is
    what makes it the natural test of ``∫ dω = 0`` for any 1-form.
    """

    def __init__(self, n: int = 2, *, n_vertices: int = 512) -> None:
        if n != 2:
            raise NotImplementedError(
                "v1 Sphere supports n=2 only; higher-dimensional spheres are "
                "deferred (Q5)."
            )
        if n_vertices < 8:
            raise ValueError("Sphere requires n_vertices >= 8 for stability.")
        self._n = n
        self._n_vertices = int(n_vertices)
        # Spherical Fibonacci lattice (almost uniform).
        indices = np.arange(self._n_vertices, dtype=np.float64)
        # Avoid index/(N-1) hitting exactly +-1 at endpoints by offsetting.
        offset = 0.5
        y = 1.0 - 2.0 * (indices + offset) / self._n_vertices
        radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
        # Golden-angle increment ensures even angular spacing.
        phi = math.pi * (3.0 - math.sqrt(5.0))
        theta = phi * indices
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        self._vertices = np.stack([x, y, z], axis=-1).astype(np.float64)
        self._weights = np.full(
            self._n_vertices, (4.0 * math.pi) / self._n_vertices, dtype=np.float64
        )

    @property
    def n(self) -> int:
        return self._n

    @property
    def radius(self) -> float:
        return 1.0

    def sample_points(self) -> np.ndarray:
        return self._vertices

    def weights(self) -> np.ndarray:
        return self._weights

    def normals(self) -> np.ndarray:
        """Outward unit normals at each vertex — equal to the vertex itself."""
        return self._vertices

    def boundary(self) -> Optional[Manifold]:
        return None  # S^n is closed for all n; no boundary.


# ---------------------------------------------------------------------------
# SOn — minimal stub for tests that need a rotor parameterization
# ---------------------------------------------------------------------------

class SOn(Manifold):
    """Rotation group ``SO(n)`` — minimal v1 stub.

    Currently supports ``n=3`` only with a simple axis-angle
    parameterization. ``sample_points()`` returns ``(N, 6)`` rows of
    ``(axis_x, axis_y, axis_z, angle, _, _)`` — six dims so the array
    is rectangular; only the first four are meaningful.

    A more principled implementation (Haar measure on rotors, intrinsic
    Lie-group integration) is deferred.
    """

    def __init__(self, n: int = 3, *, n_samples: int = 64, seed: int = 0) -> None:
        if n != 3:
            raise NotImplementedError("v1 SOn supports n=3 only.")
        self._n = n
        self._n_samples = int(n_samples)
        rng = np.random.RandomState(seed)
        axes = rng.normal(size=(self._n_samples, 3))
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        angles = rng.uniform(-math.pi, math.pi, size=(self._n_samples,))
        rows = np.zeros((self._n_samples, 6), dtype=np.float64)
        rows[:, :3] = axes
        rows[:, 3] = angles
        self._rows = rows
        # Uniform weight — placeholder; Haar measure is future work.
        self._weights = np.full(
            self._n_samples, (8.0 * math.pi * math.pi) / self._n_samples,
            dtype=np.float64,
        )

    def sample_points(self) -> np.ndarray:
        return self._rows

    def weights(self) -> np.ndarray:
        return self._weights

    def boundary(self) -> Optional[Manifold]:
        return None  # SO(n) is a closed Lie group.


__all__ = [
    "Euclidean",
    "Manifold",
    "SOn",
    "Sphere",
]
