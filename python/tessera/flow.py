"""``tessera.flow`` — visualize complex functions as flows (Ch. 10).

Needham's "flows and visualizations" chapter treats a complex
function ``f`` as a velocity field on ℂ and traces the resulting
streamlines.  Two natural choices:

  * **Direct flow**: ``dz/dt = f(z)`` — the integral curves are
    the trajectories of points moving with velocity ``f(z)``.
  * **Conjugate (potential) flow**: ``dz/dt = conjugate(f(z))`` —
    used in 2-D inviscid fluid dynamics, where ``f`` is the
    complex velocity potential and the streamlines are the level
    sets of ``Im(F(z))`` for ``F`` an antiderivative.

This module ships :func:`flow_lines` using direct flow with a
4th-order Runge-Kutta integrator.  Returns a list of trajectories
suitable for plotting with any 2-D plotter.

Usage::

    lines = flow_lines(
        lambda z: 1j * z,    # rotational flow (centered)
        x_range=(-2, 2),
        y_range=(-2, 2),
        n_seeds=12,
        t_max=2.0,
        dt=0.01,
    )
    for line in lines:
        plt.plot(line[:, 0], line[:, 1])  # (x, y) per row
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _rk4_step(f: Callable[[complex], complex], z: complex, dt: float) -> complex:
    """One RK4 step of ``dz/dt = f(z)``."""
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def flow_lines(
    f: Callable[[complex], complex],
    *,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
    n_seeds: int = 16,
    t_max: float = 1.0,
    dt: float = 0.01,
    escape_radius: float | None = None,
) -> list[np.ndarray]:
    """Compute streamlines of the complex flow ``dz/dt = f(z)``.

    Parameters
    ----------
    f
        Complex velocity field.  Called with a python ``complex``
        and must return one.
    x_range, y_range
        Seed-grid extents.  ``n_seeds`` seeds are placed on each
        axis, giving an ``n_seeds × n_seeds`` initial-point grid.
    n_seeds
        Grid resolution.
    t_max
        Integration duration.
    dt
        Step size for the 4th-order Runge-Kutta integrator.
    escape_radius
        If the trajectory's magnitude exceeds this, stop early
        (prevents an infinite-magnitude blow-up from filling the
        output).  ``None`` means no escape check.

    Returns
    -------
    list of ``np.ndarray``
        One entry per seed.  Each entry has shape ``(T, 2)``
        with columns ``(x, y)`` — the projected trajectory.
    """
    xs = np.linspace(x_range[0], x_range[1], n_seeds)
    ys = np.linspace(y_range[0], y_range[1], n_seeds)
    n_steps = max(1, int(np.ceil(t_max / dt)))
    out: list[np.ndarray] = []
    for x0 in xs:
        for y0 in ys:
            z = complex(x0, y0)
            traj = np.zeros((n_steps + 1, 2), dtype=np.float64)
            traj[0] = (z.real, z.imag)
            for step in range(n_steps):
                try:
                    z = _rk4_step(f, z, dt)
                except (ZeroDivisionError, OverflowError):
                    # Truncate at the singularity.
                    traj = traj[: step + 1]
                    break
                if not (np.isfinite(z.real) and np.isfinite(z.imag)):
                    traj = traj[: step + 1]
                    break
                if escape_radius is not None and abs(z) > escape_radius:
                    traj = traj[: step + 1]
                    break
                traj[step + 1] = (z.real, z.imag)
            out.append(traj)
    return out


__all__ = ["flow_lines"]
