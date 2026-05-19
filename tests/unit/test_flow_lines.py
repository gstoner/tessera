"""flow_lines — streamline tracing for complex velocity fields.

Coverage:

  - Constant flow ``f(z) = 1`` produces horizontal lines.
  - Rotational flow ``f(z) = i·z`` produces circular arcs around 0.
  - Radial flow ``f(z) = z`` produces outgoing rays.
  - Trajectories stay finite when ``escape_radius`` is set.
  - Output shape is ``(T+1, 2)`` per seed, ``n_seeds²`` seeds.
"""

from __future__ import annotations

import math

import numpy as np

from tessera import flow


def test_constant_flow_is_horizontal_lines() -> None:
    lines = flow.flow_lines(
        lambda z: 1 + 0j,
        x_range=(-1, 1), y_range=(-1, 1),
        n_seeds=3, t_max=0.5, dt=0.05,
    )
    # Each trajectory's y-coordinate should be constant (moved
    # only in x).
    for line in lines:
        y0 = line[0, 1]
        np.testing.assert_allclose(line[:, 1], y0, atol=1e-9)
        # x should have grown monotonically.
        assert (np.diff(line[:, 0]) > 0).all()


def test_rotational_flow_preserves_radius() -> None:
    """``f(z) = i·z`` rotates each point around the origin —
    the magnitude of the trajectory must stay constant."""
    lines = flow.flow_lines(
        lambda z: 1j * z,
        x_range=(0.5, 0.5), y_range=(0.0, 0.0),    # single seed at (0.5, 0)
        n_seeds=1, t_max=1.0, dt=0.01,
    )
    line = lines[0]
    r = np.hypot(line[:, 0], line[:, 1])
    np.testing.assert_allclose(r, r[0], atol=1e-4)


def test_radial_flow_increases_radius() -> None:
    """``f(z) = z`` — outgoing rays; magnitude grows monotonically."""
    lines = flow.flow_lines(
        lambda z: z,
        x_range=(0.1, 0.1), y_range=(0.1, 0.1),    # single seed
        n_seeds=1, t_max=1.0, dt=0.01,
    )
    line = lines[0]
    r = np.hypot(line[:, 0], line[:, 1])
    assert (np.diff(r) > 0).all()


def test_output_shape_per_seed_and_grid() -> None:
    lines = flow.flow_lines(
        lambda z: 1 + 0j,
        x_range=(0, 1), y_range=(0, 1),
        n_seeds=4, t_max=0.1, dt=0.01,
    )
    assert len(lines) == 16    # 4 x 4 grid
    for line in lines:
        assert line.shape[1] == 2


def test_escape_radius_truncates_trajectory() -> None:
    """``f(z) = z`` blows up exponentially; escape_radius=2 cuts
    early."""
    lines = flow.flow_lines(
        lambda z: z,
        x_range=(0.5, 0.5), y_range=(0.0, 0.0),
        n_seeds=1, t_max=10.0, dt=0.01,
        escape_radius=2.0,
    )
    line = lines[0]
    # Should have stopped well before t=10.
    assert line.shape[0] < 1001    # less than t_max/dt + 1
    r_final = np.hypot(line[-1, 0], line[-1, 1])
    assert r_final >= 1.0    # got at least somewhere


def test_seed_starts_at_grid_point() -> None:
    """Each trajectory's first sample is its (x, y) seed."""
    lines = flow.flow_lines(
        lambda z: 1j * z,
        x_range=(0, 1), y_range=(0, 1),
        n_seeds=2, t_max=0.0, dt=0.01,
    )
    seed_xs = [0.0, 0.0, 1.0, 1.0]
    seed_ys = [0.0, 1.0, 0.0, 1.0]
    for line, x, y in zip(lines, seed_xs, seed_ys):
        assert abs(line[0, 0] - x) < 1e-12
        assert abs(line[0, 1] - y) < 1e-12


def test_uniform_diagonal_flow() -> None:
    """``f(z) = 1 + i`` — every seed moves on a 45° diagonal."""
    lines = flow.flow_lines(
        lambda z: 1 + 1j,
        x_range=(0, 0), y_range=(0, 0),
        n_seeds=1, t_max=1.0, dt=0.01,
    )
    line = lines[0]
    # x and y advance at equal rates.
    np.testing.assert_allclose(line[:, 0], line[:, 1], atol=1e-9)
