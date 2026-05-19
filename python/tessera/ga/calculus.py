"""GA5 — Differential-form calculus on multivector fields.

Five primitives:

    HodgeStar           ⋆ω = reverse(ω) · I   (pointwise; Hestenes convention)
    ExtDeriv            exterior derivative d on a sampled field
    Codiff              codifferential d* = ±⋆d⋆
    VecDeriv            geometric gradient ∂F = Σ_i e_i ∂_i F
    Integral            Riemann-sum integration over a Manifold

The pointwise `HodgeStar` is a pure op on a `Multivector` and doesn't
need a field; `ExtDeriv` / `Codiff` / `VecDeriv` operate on a
`MultivectorField` — a numpy-backed sampled field on a uniform
Euclidean grid. `Integral` consumes either a callable or a
`MultivectorField` and a `Manifold` from `tessera.ga.manifold`.

The discretization is finite-difference (central-difference); this is
the v1 reference. Analytical / sparse / structured backends land in
GA8.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from tessera.ga.multivector import Multivector
from tessera.ga.ops import (
    geometric_product,
    reverse,
)
from tessera.ga.signature import Cl, TesseraAlgebraError


# ---------------------------------------------------------------------------
# Pointwise HodgeStar
# ---------------------------------------------------------------------------

def hodge_star(mv: Multivector) -> Multivector:
    """Hodge star ``⋆ω = reverse(ω) · I`` where ``I`` is the pseudoscalar.

    This convention (Hestenes / Doran-Lasenby) gives the expected
    ``⋆⋆ω = ±ω`` involution with sign determined by signature parity
    and grade — see ``test_ga_calculus`` for the explicit table.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_hodge_star_cl30_f32``.
    """
    from tessera.ga.ops import _try_apple_gpu_unary_8x8_cl30_f32
    gpu_out = _try_apple_gpu_unary_8x8_cl30_f32(
        mv, "tessera_apple_gpu_clifford_hodge_star_cl30_f32",
        op_name="clifford_hodge_star")
    if gpu_out is not None:
        return gpu_out
    algebra = mv.algebra
    I = Multivector.from_blade(
        algebra.pseudoscalar, algebra, dtype=mv.dtype
    )
    return geometric_product(reverse(mv), I)


# ---------------------------------------------------------------------------
# MultivectorField — sampled field on a uniform Euclidean grid
# ---------------------------------------------------------------------------

class MultivectorField:
    """A multivector-valued function sampled on a uniform Euclidean grid.

    Coefficient layout: a numpy array of shape
    ``(*spatial_shape, algebra.dim)`` where ``spatial_shape`` is the
    grid dimensions (one per spatial axis) and the last axis is the
    algebra dim. ``spacing`` records the per-axis step size (``h_i``);
    every finite-difference op reads from this.

    Created by ``MultivectorField(values, algebra, spacing)`` or via
    ``MultivectorField.from_callable(fn, algebra, grid_points)``.
    """

    __slots__ = ("_values", "_algebra", "_spacing")

    def __init__(
        self,
        values: Any,
        algebra: Cl,
        *,
        spacing: Union[float, Tuple[float, ...]] = 1.0,
    ) -> None:
        arr = np.asarray(values)
        if not isinstance(algebra, Cl):
            raise TesseraAlgebraError(
                f"MultivectorField.algebra must be a Cl signature; got "
                f"{type(algebra).__name__}."
            )
        if arr.ndim < 2 or arr.shape[-1] != algebra.dim:
            raise TesseraAlgebraError(
                f"MultivectorField values must have at least one spatial axis "
                f"plus an algebra axis of length {algebra.dim}; got shape {arr.shape}."
            )
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64, copy=False)
        spatial_ndim = arr.ndim - 1
        if isinstance(spacing, (int, float)):
            spacing_tuple = tuple(float(spacing) for _ in range(spatial_ndim))
        else:
            spacing_tuple = tuple(float(s) for s in spacing)
        if len(spacing_tuple) != spatial_ndim:
            raise TesseraAlgebraError(
                f"spacing must have {spatial_ndim} entries (one per spatial axis); "
                f"got {len(spacing_tuple)}."
            )
        object.__setattr__(self, "_values", np.ascontiguousarray(arr))
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_spacing", spacing_tuple)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def algebra(self) -> Cl:
        return self._algebra

    @property
    def spacing(self) -> Tuple[float, ...]:
        return self._spacing

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return self._values.shape[:-1]

    @property
    def spatial_ndim(self) -> int:
        return self._values.ndim - 1

    @property
    def dtype(self) -> np.dtype:
        return self._values.dtype

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[np.ndarray], Multivector],
        algebra: Cl,
        *,
        grid_points: list,
    ) -> "MultivectorField":
        """Build a field by evaluating ``fn`` at every Cartesian-product point.

        ``grid_points`` is a list of 1-D coordinate arrays — one per
        spatial axis. ``fn`` receives a coordinate vector and returns a
        ``Multivector`` over ``algebra``.
        """
        mesh = np.meshgrid(*grid_points, indexing="ij")
        spatial_shape = mesh[0].shape
        out = np.zeros((*spatial_shape, algebra.dim), dtype=np.float64)
        flat_coords = np.stack([m.ravel() for m in mesh], axis=-1)
        for flat_idx, coord in enumerate(flat_coords):
            mv = fn(coord)
            idx = np.unravel_index(flat_idx, spatial_shape)
            out[idx] = mv.coefficients
        spacing = tuple(
            float(np.mean(np.diff(g))) if len(g) > 1 else 1.0 for g in grid_points
        )
        return cls(out, algebra, spacing=spacing)

    def at(self, *indices: int) -> Multivector:
        """Extract the multivector at the given grid index."""
        return Multivector(self._values[indices], self._algebra)

    def grade(self, k: int) -> "MultivectorField":
        """Pointwise grade-k projection."""
        out = self._values.copy()
        for blade in self._algebra.blades():
            if blade.grade != k:
                out[..., blade.mask] = 0
        return MultivectorField(out, self._algebra, spacing=self._spacing)

    def __add__(self, other: "MultivectorField") -> "MultivectorField":
        if self._algebra != other._algebra:
            raise TesseraAlgebraError(
                f"field algebra mismatch: {self._algebra!r} vs {other._algebra!r}."
            )
        return MultivectorField(
            self._values + other._values, self._algebra, spacing=self._spacing
        )


# ---------------------------------------------------------------------------
# Field-level ops: ExtDeriv / Codiff / VecDeriv
# ---------------------------------------------------------------------------

def _partial(values: np.ndarray, axis: int, h: float) -> np.ndarray:
    """Central-difference partial derivative along the given spatial axis."""
    # values shape: (*spatial, dim). axis indexes a spatial axis.
    return np.gradient(values, h, axis=axis, edge_order=2)


def _apply_left_basis_product(
    algebra: Cl,
    axis_idx: int,
    derivative_field: np.ndarray,
) -> np.ndarray:
    """Compute (e_{axis_idx+1}) · D where D is a sampled multivector field.

    Returns a field of the same shape; uses the algebra's product table.
    """
    table = algebra.product_table()
    e_mask = algebra.blade(f"e{axis_idx + 1}").mask
    out = np.zeros_like(derivative_field)
    row = table[e_mask]
    for j in range(algebra.dim):
        result_mask, sign = row[j]
        if sign == 0:
            continue
        coeff = derivative_field[..., j]
        if sign == 1:
            out[..., result_mask] = out[..., result_mask] + coeff
        else:
            out[..., result_mask] = out[..., result_mask] - coeff
    return out


def _wedge_left_basis(
    algebra: Cl,
    axis_idx: int,
    derivative_field: np.ndarray,
) -> np.ndarray:
    """Compute (e_{axis_idx+1}) ∧ D pointwise — like _apply_left_basis_product
    but with the disjoint-index gate (shared generators ⇒ zero)."""
    table = algebra.product_table()
    e_mask = algebra.blade(f"e{axis_idx + 1}").mask
    out = np.zeros_like(derivative_field)
    row = table[e_mask]
    for j in range(algebra.dim):
        if (e_mask & j) != 0:
            continue
        result_mask, sign = row[j]
        if sign == 0:
            continue
        coeff = derivative_field[..., j]
        if sign == 1:
            out[..., result_mask] = out[..., result_mask] + coeff
        else:
            out[..., result_mask] = out[..., result_mask] - coeff
    return out


def ext_deriv(field: MultivectorField) -> MultivectorField:
    """Exterior derivative ``dω`` of a multivector field.

    ``dω = Σ_i (e_i ∧ ∂_i ω)`` — by construction antisymmetric, so
    ``d²ω = 0`` on the interior of the grid (modulo finite-difference
    boundary effects).

    **Apple GPU fast path**: Cl(3,0) f32 fields on 3D grids route
    through ``tessera_apple_gpu_clifford_ext_deriv_cl30_f32``.
    Boundary cells are zero-padded by the kernel (the numpy path uses
    `np.gradient` with one-sided 2nd-order at boundaries; the two
    agree on interior cells to fp32 tolerance).
    """
    gpu_out = _try_apple_gpu_field_op_cl30_f32(
        field, "tessera_apple_gpu_clifford_ext_deriv_cl30_f32")
    if gpu_out is not None:
        return gpu_out
    algebra = field.algebra
    if field.spatial_ndim != algebra.n:
        raise TesseraAlgebraError(
            f"ext_deriv requires field.spatial_ndim ({field.spatial_ndim}) to "
            f"equal algebra.n ({algebra.n})."
        )
    out = np.zeros_like(field.values)
    for axis in range(field.spatial_ndim):
        d_axis = _partial(field.values, axis=axis, h=field.spacing[axis])
        out = out + _wedge_left_basis(algebra, axis, d_axis)
    return MultivectorField(out, algebra, spacing=field.spacing)


def vec_deriv(field: MultivectorField) -> MultivectorField:
    """Geometric gradient ``∂F = Σ_i e_i · ∂_i F``.

    Unlike ``ext_deriv`` this uses the full geometric product (not
    wedge), so derivatives of grade-k components can produce
    components at grade ``k-1`` (via inner product) and ``k+1`` (via
    wedge).

    **Apple GPU fast path**: Cl(3,0) f32 fields on 3D grids route
    through ``tessera_apple_gpu_clifford_vec_deriv_cl30_f32``.
    """
    gpu_out = _try_apple_gpu_field_op_cl30_f32(
        field, "tessera_apple_gpu_clifford_vec_deriv_cl30_f32")
    if gpu_out is not None:
        return gpu_out
    algebra = field.algebra
    if field.spatial_ndim != algebra.n:
        raise TesseraAlgebraError(
            f"vec_deriv requires field.spatial_ndim ({field.spatial_ndim}) to "
            f"equal algebra.n ({algebra.n})."
        )
    out = np.zeros_like(field.values)
    for axis in range(field.spatial_ndim):
        d_axis = _partial(field.values, axis=axis, h=field.spacing[axis])
        out = out + _apply_left_basis_product(algebra, axis, d_axis)
    return MultivectorField(out, algebra, spacing=field.spacing)


def hodge_star_field(field: MultivectorField) -> MultivectorField:
    """Apply Hodge star pointwise to every sample of a field."""
    algebra = field.algebra
    out = np.empty_like(field.values)
    # Vectorized: ⋆ω = reverse(ω) · I; both ops linear in coefficients,
    # so we can encode them as a constant matrix multiply.
    reverse_signs = np.array(
        [(-1) ** ((b.grade * (b.grade - 1)) // 2) for b in algebra.blades()],
        dtype=field.dtype,
    )
    rev_values = field.values * reverse_signs
    # Right-multiply by I = pseudoscalar.
    table = algebra.product_table()
    I_mask = algebra.pseudoscalar.mask
    # For each input coefficient j, ⋆ω contributes to result_mask = table[j][I_mask][0].
    out.fill(0)
    for j in range(algebra.dim):
        result_mask, sign = table[j][I_mask]
        if sign == 0:
            continue
        if sign == 1:
            out[..., result_mask] = out[..., result_mask] + rev_values[..., j]
        else:
            out[..., result_mask] = out[..., result_mask] - rev_values[..., j]
    return MultivectorField(out, algebra, spacing=field.spacing)


def codiff(field: MultivectorField) -> MultivectorField:
    """Codifferential ``d*ω = ±⋆d⋆ω`` for a sampled field.

    The sign is grade- and signature-dependent; for v1 we apply the
    ``⋆ ∘ d ∘ ⋆`` composition without inserting the explicit sign —
    callers that need the strict ``d*`` sign convention should apply
    it themselves.

    **Apple GPU fast path**: Cl(3,0) f32 fields on 3D grids route
    through ``tessera_apple_gpu_clifford_codiff_cl30_f32``, which
    composes three MSL dispatches (hodge → ext_deriv → hodge).
    """
    gpu_out = _try_apple_gpu_field_op_cl30_f32(
        field, "tessera_apple_gpu_clifford_codiff_cl30_f32")
    if gpu_out is not None:
        return gpu_out
    return hodge_star_field(ext_deriv(hodge_star_field(field)))


# ---------------------------------------------------------------------------
# Apple GPU dispatch helper — field ops share the same ``(F, Out,
# D0, D1, D2, h0, h1, h2)`` ABI; one helper covers ext_deriv /
# vec_deriv / codiff.
# ---------------------------------------------------------------------------

# Per-symbol → op_name mapping for the field-op family.  Lets the
# helper route through the JIT bridge so the route trace records the
# manifest-resolved op name (not just the raw C ABI symbol).
_FIELD_OP_BRIDGE_OP_NAME = {
    "tessera_apple_gpu_clifford_ext_deriv_cl30_f32": "clifford_ext_deriv",
    "tessera_apple_gpu_clifford_vec_deriv_cl30_f32": "clifford_vec_deriv",
    "tessera_apple_gpu_clifford_codiff_cl30_f32":    "clifford_codiff",
}


def _try_apple_gpu_field_op_cl30_f32(
    field: "MultivectorField", symbol: str,
) -> "Optional[MultivectorField]":
    """Dispatch a Cl(3,0) f32 field op to its Apple GPU kernel via the
    JIT bridge.

    Returns ``None`` when (a) the algebra isn't Cl(3,0), (b) the
    grid isn't 3-D, (c) the dtype isn't f32, (d) the values aren't
    C-contiguous, or (e) the runtime is unavailable.
    """
    from tessera.ga.signature import Cl  # noqa: F401

    if field.algebra.signature != (3, 0, 0):
        return None
    if field.spatial_ndim != 3:
        return None
    if field.dtype != np.float32:
        return None
    if not field.values.flags["C_CONTIGUOUS"]:
        return None
    op_name = _FIELD_OP_BRIDGE_OP_NAME.get(symbol)
    if op_name is None:
        return None  # unknown symbol — no manifest entry to resolve
    try:
        import ctypes
        from tessera.compiler import jit_bridge as _bridge
    except ImportError:
        return None
    F = field.values
    D0, D1, D2 = F.shape[0], F.shape[1], F.shape[2]
    h0, h1, h2 = field.spacing
    out = np.zeros_like(F)
    p = ctypes.POINTER(ctypes.c_float)
    argtypes = (ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_float, ctypes.c_float, ctypes.c_float)
    args = (F.ctypes.data_as(p), out.ctypes.data_as(p),
            ctypes.c_int32(D0), ctypes.c_int32(D1), ctypes.c_int32(D2),
            ctypes.c_float(float(h0)), ctypes.c_float(float(h1)),
            ctypes.c_float(float(h2)))
    try:
        ok = _bridge.dispatch_via_manifest(
            op_name, argtypes=argtypes, args=args,
            args_summary=_bridge.shaped_summary(F),
        )
    except _bridge.JitBridgeMiss:
        return None
    if not ok:
        return None
    return MultivectorField(out, field.algebra, spacing=field.spacing)


# ---------------------------------------------------------------------------
# Integral — Riemann sum over a Manifold
# ---------------------------------------------------------------------------

def integral(
    integrand: Union[Callable[[np.ndarray], Multivector], MultivectorField],
    manifold: Any,
) -> np.ndarray:
    """Integrate over a manifold.

    Two modes:

    * **Callable mode.** ``integrand(point) -> Multivector``. Each
      sample point on the manifold is evaluated and weighted by the
      manifold's volume element. Returns a coefficient array (the
      summed multivector coefficient vector).

    * **Field mode.** ``integrand: MultivectorField`` on a Euclidean
      grid that the manifold's points address by index. Used for
      bulk grid integration; the manifold must be ``Euclidean`` with
      matching shape.
    """
    from tessera.ga.manifold import Euclidean, Manifold

    if not isinstance(manifold, Manifold):
        raise TesseraAlgebraError(
            f"integral requires a Manifold; got {type(manifold).__name__}."
        )

    points = manifold.sample_points()
    weights = manifold.weights()

    if isinstance(integrand, MultivectorField):
        if not isinstance(manifold, Euclidean):
            raise TesseraAlgebraError(
                "Field-mode integral currently requires a Euclidean manifold."
            )
        if integrand.spatial_shape != manifold.shape:
            raise TesseraAlgebraError(
                f"Field shape {integrand.spatial_shape} does not match "
                f"manifold shape {manifold.shape}."
            )
        # Sum coefficient array weighted by per-cell volume.
        dim = integrand.algebra.dim
        flat = integrand.values.reshape(-1, dim)
        # Apple GPU fast path — only for Cl(3,0) f32 fields where the
        # native ``ebm_integral_cl30_f32`` kernel can accept the same
        # ``(field[N,8], weights[N], out[8], n)`` layout.
        gpu_out = _try_apple_gpu_integral_cl30_f32(
            integrand, np.asarray(weights, dtype=np.float64))
        if gpu_out is not None:
            return gpu_out
        return np.einsum("i,ij->j", weights, flat)

    # Callable mode.
    if not callable(integrand):
        raise TesseraAlgebraError(
            "integral integrand must be a callable or a MultivectorField; "
            f"got {type(integrand).__name__}."
        )
    samples = []
    for p in points:
        out = integrand(p)
        if not isinstance(out, Multivector):
            raise TesseraAlgebraError(
                f"integral callable must return a Multivector; got {type(out).__name__}."
            )
        samples.append(out.coefficients)
    sample_arr = np.stack(samples, axis=0)
    return np.einsum("i,ij->j", weights, sample_arr)


def _try_apple_gpu_integral_cl30_f32(
    field: "MultivectorField", weights: np.ndarray,
) -> "Optional[np.ndarray]":
    """Apple GPU fast path for the Cl(3,0) field-mode integral.

    Computes ``out[c] = Σ_i weights[i] * field[i, c]`` natively — the
    same weighted-sum semantics the numpy ``einsum`` path uses.
    Returns ``None`` to fall back when (a) the algebra isn't Cl(3,0),
    (b) the field isn't f32, (c) the weights aren't convertible to
    f32, or (d) the runtime isn't loadable.
    """
    if field.algebra.signature != (3, 0, 0):
        return None
    if field.dtype != np.float32:
        return None
    flat = np.ascontiguousarray(field.values.reshape(-1, 8))
    if flat.shape[1] != 8:
        return None
    weights_f32 = np.ascontiguousarray(weights.astype(np.float32, copy=False))
    if weights_f32.shape[0] != flat.shape[0]:
        return None
    try:
        import ctypes
        from tessera.compiler import jit_bridge as _bridge
    except ImportError:
        return None
    out = np.zeros(8, dtype=np.float32)
    n = flat.shape[0]
    p = ctypes.POINTER(ctypes.c_float)
    try:
        ok = _bridge.dispatch_via_manifest(
            "clifford_integral",
            argtypes=(ctypes.POINTER(ctypes.c_float),
                      ctypes.POINTER(ctypes.c_float),
                      ctypes.POINTER(ctypes.c_float),
                      ctypes.c_int32),
            args=(flat.ctypes.data_as(p), weights_f32.ctypes.data_as(p),
                  out.ctypes.data_as(p), ctypes.c_int32(n)),
            args_summary=_bridge.shaped_summary(flat, weights_f32),
        )
    except _bridge.JitBridgeMiss:
        return None
    if not ok:
        return None
    return out


__all__ = [
    "MultivectorField",
    "codiff",
    "ext_deriv",
    "hodge_star",
    "hodge_star_field",
    "integral",
    "vec_deriv",
]
