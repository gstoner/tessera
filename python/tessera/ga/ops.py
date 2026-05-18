r"""GA3 — Pure-function multivector operations.

Every op here is a pure function over ``tessera.ga.Multivector`` values.
The 1:1 lowering target in GA7 is the ``tessera.clifford`` Graph IR
dialect (`clifford.geo_product`, `clifford.grade`, etc.).

Operations shipped in GA3:

    geometric_product   the geometric (Clifford) product ``a * b``
    grade_projection    extract the grade-k component
    wedge               outer product ``a ∧ b``
    inner               symmetric (scalar) inner product ``<a, b>``
    left_contraction    left contraction ``a ⌋ b``
    reverse             the reversion anti-automorphism ``a†``
    conjugate           Clifford conjugation ``a*``
    grade_involution    grade involution ``\hat a``
    norm                sqrt of <a, reverse(a)>
    exp_mv              power-series exponential (closed-form for Cl(3,0) bivectors)
    log_mv              power-series logarithm (closed-form for Cl(3,0) rotors)
    rotor_from_axis     build a rotor from (bivector axis, angle)
    rotor_sandwich      apply ``R x R^†`` (rotation)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np

from tessera.ga.multivector import Multivector
from tessera.ga.signature import Cl, TesseraAlgebraError


_GradeArg = Union[int, Iterable[int]]


def _coerce_grade_set(k: _GradeArg, algebra: Cl) -> frozenset[int]:
    if isinstance(k, int):
        out = frozenset({k})
    else:
        out = frozenset(int(g) for g in k)
    valid = set(algebra.grades)
    invalid = out - valid
    if invalid:
        raise TesseraAlgebraError(
            f"grades {sorted(invalid)} are not valid for {algebra!r}; "
            f"valid grades are {sorted(valid)}."
        )
    return out


def _same_algebra(a: Multivector, b: Multivector) -> Cl:
    if a.algebra != b.algebra:
        raise TesseraAlgebraError(
            f"GA op requires matching algebras; got {a.algebra!r} and {b.algebra!r}."
        )
    return a.algebra


def _promoted_dtype(a: Multivector, b: Multivector) -> np.dtype:
    return np.result_type(a.dtype, b.dtype)


# ---------------------------------------------------------------------------
# Geometric product (the load-bearing one)
# ---------------------------------------------------------------------------

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """The fundamental Clifford product ``a * b``.

    Implemented as a per-blade-pair scatter into the result using the
    algebra's compile-time-cached Cayley table. Batch axes (leading
    dims of the coefficient arrays) are broadcast.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_geo_product_cl30_f32``.
    """
    algebra = _same_algebra(a, b)
    if algebra.signature == (3, 0, 0):
        gpu_out = _try_apple_gpu_binary_8x8_cl30_f32(
            a, b, "tessera_apple_gpu_clifford_geo_product_cl30_f32",
            op_name="clifford_geometric_product")
        if gpu_out is not None:
            return gpu_out
    table = algebra.product_table()
    dim = algebra.dim
    dtype = _promoted_dtype(a, b)
    a_co = a.coefficients.astype(dtype, copy=False)
    b_co = b.coefficients.astype(dtype, copy=False)
    # Broadcast the leading axes.
    leading_shape = np.broadcast_shapes(a_co.shape[:-1], b_co.shape[:-1])
    out = np.zeros((*leading_shape, dim), dtype=dtype)
    for i in range(dim):
        ai = a_co[..., i]
        if not np.any(ai):
            continue
        row = table[i]
        for j in range(dim):
            result_mask, sign = row[j]
            if sign == 0:
                continue
            bj = b_co[..., j]
            if sign == 1:
                out[..., result_mask] = out[..., result_mask] + ai * bj
            else:
                out[..., result_mask] = out[..., result_mask] - ai * bj
    return Multivector(out, algebra)


# ---------------------------------------------------------------------------
# Grade projection
# ---------------------------------------------------------------------------

def grade_projection(a: Multivector, k: _GradeArg) -> Multivector:
    """Project ``a`` onto the subspace of the requested grade(s).

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_grade_projection_cl30_f32`` with the
    blade-mask encoded as ``sum(1 << blade.mask for grades)``.
    """
    grade_set = _coerce_grade_set(k, a.algebra)
    if a.algebra.signature == (3, 0, 0):
        # The MSL kernel uses a per-grade bitmask: bit g set iff grade g
        # is kept.  Matches the GA10 conformance fixture's encoding —
        # e.g., 0b0101 for grades {0, 2} (scalar + bivector subalgebra).
        mask = 0
        for g in grade_set:
            mask |= (1 << int(g))
        gpu_out = _try_apple_gpu_grade_projection_cl30_f32(a, mask)
        if gpu_out is not None:
            return Multivector(gpu_out.coefficients, a.algebra, grades=grade_set)
    coeffs = a.coefficients.copy()
    for blade in a.algebra.blades():
        if blade.grade not in grade_set:
            coeffs[..., blade.mask] = 0
    return Multivector(coeffs, a.algebra, grades=grade_set)


# ---------------------------------------------------------------------------
# Wedge (outer product) and contractions
# ---------------------------------------------------------------------------

def wedge(a: Multivector, b: Multivector) -> Multivector:
    """The outer (exterior) product ``a ∧ b``.

    For basis blades, ``e_I ∧ e_J = e_{I ∪ J}`` when the index sets are
    disjoint and ``0`` otherwise. The sign comes from reordering, same
    as the geometric product but with the disjoint-index gate.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_wedge_cl30_f32``.
    """
    algebra = _same_algebra(a, b)
    if algebra.signature == (3, 0, 0):
        gpu_out = _try_apple_gpu_binary_8x8_cl30_f32(
            a, b, "tessera_apple_gpu_clifford_wedge_cl30_f32",
            op_name="clifford_wedge")
        if gpu_out is not None:
            return gpu_out
    table = algebra.product_table()
    dim = algebra.dim
    dtype = _promoted_dtype(a, b)
    a_co = a.coefficients.astype(dtype, copy=False)
    b_co = b.coefficients.astype(dtype, copy=False)
    leading_shape = np.broadcast_shapes(a_co.shape[:-1], b_co.shape[:-1])
    out = np.zeros((*leading_shape, dim), dtype=dtype)
    for i in range(dim):
        ai = a_co[..., i]
        if not np.any(ai):
            continue
        for j in range(dim):
            if (i & j) != 0:
                continue  # shared generator -> wedge is zero
            result_mask, sign = table[i][j]
            if sign == 0:
                continue
            bj = b_co[..., j]
            if sign == 1:
                out[..., result_mask] = out[..., result_mask] + ai * bj
            else:
                out[..., result_mask] = out[..., result_mask] - ai * bj
    return Multivector(out, algebra)


def left_contraction(a: Multivector, b: Multivector) -> Multivector:
    """Left contraction ``a ⌋ b``.

    For grade-pure inputs ``a`` (grade r) and ``b`` (grade s), the
    result is the grade-``(s - r)`` part of ``a * b`` when ``s ≥ r``,
    else zero. The general multivector form is the sum of these
    grade-pure contributions.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_left_contraction_cl30_f32``.
    """
    algebra = _same_algebra(a, b)
    if algebra.signature == (3, 0, 0):
        gpu_out = _try_apple_gpu_binary_8x8_cl30_f32(
            a, b, "tessera_apple_gpu_clifford_left_contraction_cl30_f32",
            op_name="clifford_left_contraction")
        if gpu_out is not None:
            return gpu_out
    table = algebra.product_table()
    dim = algebra.dim
    dtype = _promoted_dtype(a, b)
    a_co = a.coefficients.astype(dtype, copy=False)
    b_co = b.coefficients.astype(dtype, copy=False)
    leading_shape = np.broadcast_shapes(a_co.shape[:-1], b_co.shape[:-1])
    out = np.zeros((*leading_shape, dim), dtype=dtype)
    for i in range(dim):
        grade_i = i.bit_count()
        ai = a_co[..., i]
        if not np.any(ai):
            continue
        for j in range(dim):
            grade_j = j.bit_count()
            target_grade = grade_j - grade_i
            if target_grade < 0:
                continue
            result_mask, sign = table[i][j]
            if sign == 0:
                continue
            if result_mask.bit_count() != target_grade:
                continue
            bj = b_co[..., j]
            if sign == 1:
                out[..., result_mask] = out[..., result_mask] + ai * bj
            else:
                out[..., result_mask] = out[..., result_mask] - ai * bj
    return Multivector(out, algebra)


def inner(a: Multivector, b: Multivector) -> np.ndarray:
    """The scalar (symmetric) inner product ``<a, b> = <a * reverse(b)>_0``.

    Returns the scalar coefficient array (not a Multivector) — this is
    a bilinear form, not a multivector-valued op.

    **Apple GPU fast path**: for ``Cl(3,0)`` f32 inputs with matching
    batch shape, this routes through
    ``tessera_apple_gpu_clifford_inner_cl30_f32`` via
    ``tessera._apple_gpu_dispatch``. Falls back silently to the
    pure-Python path when the runtime is unavailable.
    """
    algebra = _same_algebra(a, b)
    if algebra.signature == (3, 0, 0):
        gpu_out = _try_apple_gpu_inner_cl30_f32(a, b)
        if gpu_out is not None:
            return gpu_out
    return geometric_product(a, reverse(b)).scalar_part()


# ---------------------------------------------------------------------------
# Apple GPU fast-path helpers — one per op.  Each returns either the
# GPU output or ``None`` (caller falls back to numpy).  They all share
# the same guard pattern: Cl(3,0) signature, f32 dtype, contiguous
# 8-trailing-axis layout.
# ---------------------------------------------------------------------------

def _is_cl30_f32_8axis(*arrays: np.ndarray) -> bool:
    """Common gate for the ``Cl(3,0)`` f32 GPU kernels."""
    if not arrays:
        return False
    ref = arrays[0]
    for arr in arrays:
        if arr.dtype != np.float32:
            return False
        if arr.ndim < 1 or arr.shape[-1] != 8:
            return False
        if arr.shape != ref.shape:
            return False
    return True


def _try_apple_gpu_unary_8x8_cl30_f32(
    a: Multivector, symbol: str, *, op_name: Optional[str] = None,
) -> Optional[Multivector]:
    """Generic ``(in[8], out[8], batch)`` Apple GPU dispatcher for
    Cl(3,0) unary multivector ops (``exp_mv``, ``reverse``, ...).

    Routes through :func:`tessera.compiler.jit_bridge.dispatch_via_manifest`
    when ``op_name`` is supplied — the bridge then resolves the symbol
    through the manifest and records the route in the thread-local
    trace.  Falls back to a direct ``bind_symbol`` call when ``op_name``
    is omitted (legacy callers).
    """
    if a.algebra.signature != (3, 0, 0):
        return None
    A = a.coefficients
    if not _is_cl30_f32_8axis(A):
        return None
    try:
        import ctypes
    except ImportError:
        return None
    A_c = np.ascontiguousarray(A.reshape(-1, 8))
    batch = A_c.shape[0]
    out = np.zeros_like(A_c)
    p = ctypes.POINTER(ctypes.c_float)
    argtypes = (ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32)
    args = (A_c.ctypes.data_as(p), out.ctypes.data_as(p), ctypes.c_int32(batch))
    if op_name is not None:
        try:
            from tessera.compiler import jit_bridge as _bridge
            ok = _bridge.dispatch_via_manifest(
                op_name, argtypes=argtypes, args=args,
                args_summary=_bridge.shaped_summary(A_c),
            )
        except _bridge.JitBridgeMiss:
            return None
        if not ok:
            return None
        return Multivector(out.reshape(A.shape), a.algebra)
    # Legacy path — direct bind_symbol, no trace recording.
    from tessera._apple_gpu_dispatch import bind_symbol
    fn = bind_symbol(symbol, argtypes)
    if fn is None:
        return None
    fn(*args)
    return Multivector(out.reshape(A.shape), a.algebra)


def _try_apple_gpu_inner_cl30_f32(
    a: Multivector, b: Multivector,
) -> Optional[np.ndarray]:
    """``<a, b>`` on ``Cl(3,0)`` f32 batched inputs.

    Routed through :mod:`tessera.compiler.jit_bridge` so the manifest
    resolves the C ABI symbol and the shared loader binds + invokes
    it.  Records a route trace row when bridge tracing is enabled.
    """
    A = a.coefficients
    B = b.coefficients
    if not _is_cl30_f32_8axis(A, B):
        return None
    try:
        import ctypes
        from tessera.compiler import jit_bridge as _bridge
    except ImportError:
        return None
    A_c = np.ascontiguousarray(A.reshape(-1, 8))
    B_c = np.ascontiguousarray(B.reshape(-1, 8))
    batch = A_c.shape[0]
    out = np.zeros(batch, dtype=np.float32)
    p = ctypes.POINTER(ctypes.c_float)
    try:
        ok = _bridge.dispatch_via_manifest(
            "clifford_inner",
            argtypes=(ctypes.POINTER(ctypes.c_float),
                      ctypes.POINTER(ctypes.c_float),
                      ctypes.POINTER(ctypes.c_float),
                      ctypes.c_int32),
            args=(A_c.ctypes.data_as(p), B_c.ctypes.data_as(p),
                  out.ctypes.data_as(p), ctypes.c_int32(batch)),
            args_summary=_bridge.shaped_summary(A_c, B_c),
        )
    except _bridge.JitBridgeMiss:
        return None
    if not ok:
        return None
    return out.reshape(A.shape[:-1]) if A.ndim > 1 else out[0]


def _try_apple_gpu_norm_cl30_f32(a: Multivector) -> Optional[np.ndarray]:
    """``|a|`` on ``Cl(3,0)`` f32 batched input — routed through the
    JIT bridge so the route is traced + manifest-verified."""
    if a.algebra.signature != (3, 0, 0):
        return None
    A = a.coefficients
    if not _is_cl30_f32_8axis(A):
        return None
    try:
        import ctypes
        from tessera.compiler import jit_bridge as _bridge
    except ImportError:
        return None
    A_c = np.ascontiguousarray(A.reshape(-1, 8))
    batch = A_c.shape[0]
    out = np.zeros(batch, dtype=np.float32)
    p = ctypes.POINTER(ctypes.c_float)
    try:
        ok = _bridge.dispatch_via_manifest(
            "clifford_norm",
            argtypes=(ctypes.POINTER(ctypes.c_float),
                      ctypes.POINTER(ctypes.c_float),
                      ctypes.c_int32),
            args=(A_c.ctypes.data_as(p), out.ctypes.data_as(p),
                  ctypes.c_int32(batch)),
            args_summary=_bridge.shaped_summary(A_c),
        )
    except _bridge.JitBridgeMiss:
        return None
    if not ok:
        return None
    return out.reshape(A.shape[:-1]) if A.ndim > 1 else out[0]


def _try_apple_gpu_rotor_sandwich_cl30_f32(
    rotor: Multivector, x: Multivector,
) -> Optional[Multivector]:
    """``R x R†`` on ``Cl(3,0)`` f32 batched inputs."""
    return _try_apple_gpu_binary_8x8_cl30_f32(
        rotor, x, "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32",
        op_name="clifford_rotor_sandwich")


def _try_apple_gpu_binary_8x8_cl30_f32(
    a: Multivector, b: Multivector, symbol: str, *,
    op_name: Optional[str] = None,
) -> Optional[Multivector]:
    """Generic Cl(3,0) f32 binary 8×8→8 Apple GPU dispatch helper.

    Used by ``geometric_product`` / ``wedge`` / ``left_contraction`` /
    ``rotor_sandwich`` — they share the ``(A, B, Out, batch)`` ABI.
    Routes through the JIT bridge when ``op_name`` is supplied.
    """
    if a.algebra != b.algebra or a.algebra.signature != (3, 0, 0):
        return None
    A = a.coefficients
    B = b.coefficients
    if not _is_cl30_f32_8axis(A, B):
        return None
    try:
        import ctypes
    except ImportError:
        return None
    A_c = np.ascontiguousarray(A.reshape(-1, 8))
    B_c = np.ascontiguousarray(B.reshape(-1, 8))
    batch = A_c.shape[0]
    out = np.zeros_like(A_c)
    p = ctypes.POINTER(ctypes.c_float)
    argtypes = (ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32)
    args = (A_c.ctypes.data_as(p), B_c.ctypes.data_as(p),
            out.ctypes.data_as(p), ctypes.c_int32(batch))
    if op_name is not None:
        try:
            from tessera.compiler import jit_bridge as _bridge
            ok = _bridge.dispatch_via_manifest(
                op_name, argtypes=argtypes, args=args,
                args_summary=_bridge.shaped_summary(A_c, B_c),
            )
        except _bridge.JitBridgeMiss:
            return None
        if not ok:
            return None
        return Multivector(out.reshape(A.shape), a.algebra)
    # Legacy direct-bind path.
    from tessera._apple_gpu_dispatch import bind_symbol
    fn = bind_symbol(symbol, argtypes)
    if fn is None:
        return None
    fn(*args)
    return Multivector(out.reshape(A.shape), a.algebra)


def _try_apple_gpu_grade_projection_cl30_f32(
    a: Multivector, grade_mask: int,
) -> Optional[Multivector]:
    """``Cl(3,0)`` f32 grade projection.  ABI is
    ``(in, out, grade_mask, batch)``."""
    if a.algebra.signature != (3, 0, 0):
        return None
    A = a.coefficients
    if not _is_cl30_f32_8axis(A):
        return None
    try:
        import ctypes
        from tessera._apple_gpu_dispatch import bind_symbol
    except ImportError:
        return None
    fn = bind_symbol(
        "tessera_apple_gpu_clifford_grade_projection_cl30_f32",
        (ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float),
         ctypes.c_int32, ctypes.c_int32),
    )
    if fn is None:
        return None
    A_c = np.ascontiguousarray(A.reshape(-1, 8))
    batch = A_c.shape[0]
    out = np.zeros_like(A_c)
    p = ctypes.POINTER(ctypes.c_float)
    fn(A_c.ctypes.data_as(p), out.ctypes.data_as(p),
       ctypes.c_int32(grade_mask), ctypes.c_int32(batch))
    return Multivector(out.reshape(A.shape), a.algebra)


# ---------------------------------------------------------------------------
# Anti-automorphisms: reverse / conjugate / grade_involution
# ---------------------------------------------------------------------------

def _grade_sign_array(algebra: Cl, sign_fn) -> np.ndarray:
    return np.array(
        [sign_fn(b.grade) for b in algebra.blades()], dtype=np.int8
    )


def reverse(a: Multivector) -> Multivector:
    """Reversion ``a†``: blade of grade k picks up sign ``(-1)^{k(k-1)/2}``.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_reverse_cl30_f32``."""
    gpu_out = _try_apple_gpu_unary_8x8_cl30_f32(
        a, "tessera_apple_gpu_clifford_reverse_cl30_f32",
        op_name="clifford_reverse")
    if gpu_out is not None:
        return gpu_out
    signs = _grade_sign_array(a.algebra, lambda k: (-1) ** ((k * (k - 1)) // 2))
    return Multivector(
        a.coefficients * signs.astype(a.dtype, copy=False),
        a.algebra,
        grades=a.grades,
    )


def grade_involution(a: Multivector) -> Multivector:
    """Grade involution ``â``: blade of grade k picks up sign ``(-1)^k``.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_grade_involution_cl30_f32``."""
    gpu_out = _try_apple_gpu_unary_8x8_cl30_f32(
        a, "tessera_apple_gpu_clifford_grade_involution_cl30_f32",
        op_name="clifford_grade_involution")
    if gpu_out is not None:
        return gpu_out
    signs = _grade_sign_array(a.algebra, lambda k: (-1) ** k)
    return Multivector(
        a.coefficients * signs.astype(a.dtype, copy=False),
        a.algebra,
        grades=a.grades,
    )


def conjugate(a: Multivector) -> Multivector:
    """Clifford conjugation: combination of reverse + grade involution.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_conjugate_cl30_f32``."""
    gpu_out = _try_apple_gpu_unary_8x8_cl30_f32(
        a, "tessera_apple_gpu_clifford_conjugate_cl30_f32",
        op_name="clifford_conjugate")
    if gpu_out is not None:
        return gpu_out
    return reverse(grade_involution(a))


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

def norm_squared(a: Multivector) -> np.ndarray:
    """``|a|² = <a, a>`` — scalar coefficient array."""
    return inner(a, a)


def norm(a: Multivector) -> np.ndarray:
    """Multivector norm ``|a| = sqrt(<a, a>)`` (clipped to non-negative inputs).

    **Apple GPU fast path**: routes ``Cl(3,0)`` f32 batched inputs
    through ``tessera_apple_gpu_clifford_norm_cl30_f32`` via the
    shared dispatcher.  Falls back to the numpy clip+sqrt path
    otherwise.
    """
    gpu_out = _try_apple_gpu_norm_cl30_f32(a)
    if gpu_out is not None:
        return gpu_out
    n2 = np.asarray(norm_squared(a))
    return np.sqrt(np.clip(n2, 0.0, None))


# ---------------------------------------------------------------------------
# Exp / log
# ---------------------------------------------------------------------------

def _is_pure_bivector_3d(a: Multivector) -> bool:
    """True if `a` is a pure grade-2 multivector in Cl(3,0)."""
    if a.algebra.signature != (3, 0, 0):
        return False
    grades = a.active_grades
    return grades == frozenset({2})


def exp_mv(a: Multivector, *, terms: int = 24) -> Multivector:
    """Exponential ``exp(a)``.

    **Apple GPU fast path**: ``Cl(3,0)`` f32 batched inputs route
    through ``tessera_apple_gpu_clifford_exp_cl30_f32``.  The native
    kernel itself uses the closed-form Euler identity for pure
    bivectors and a 24-term power series for the general case — same
    contract as the Python ref below.


    For pure grade-2 multivectors in Cl(3,0), uses the closed-form
    Euler-like identity ``exp(B) = cos(|B|) + sin(|B|) * B / |B|``
    (with ``B² = −|B|²``). For the general case, truncated power
    series with ``terms`` terms (default 24 is sufficient for ``|a| ≲ 2``).
    """
    gpu_out = _try_apple_gpu_unary_8x8_cl30_f32(
        a, "tessera_apple_gpu_clifford_exp_cl30_f32",
        op_name="clifford_exp")
    if gpu_out is not None:
        return gpu_out
    if _is_pure_bivector_3d(a):
        B_norm = np.asarray(norm(a))
        # Avoid division by zero for the zero bivector.
        safe = np.where(B_norm > 1e-12, B_norm, 1.0)
        scalar = Multivector.scalar(
            0.0, a.algebra, shape=a.shape, dtype=a.dtype
        )
        cos_part = scalar.to_numpy()
        cos_part[..., 0] = np.cos(B_norm)
        sin_scaled = a.coefficients * (np.sin(B_norm) / safe)[..., None]
        return Multivector(cos_part + sin_scaled, a.algebra)
    # General power series.
    algebra = a.algebra
    one = Multivector.scalar(1.0, algebra, shape=a.shape, dtype=a.dtype)
    result = one
    power = one
    fact = 1.0
    for k in range(1, terms + 1):
        power = geometric_product(power, a)
        fact *= k
        result = result + (1.0 / fact) * power
    return result


def log_mv(a: Multivector, *, terms: int = 64) -> Multivector:
    """Logarithm ``log(a)``.

    For Cl(3,0) rotors (scalar + bivector with unit norm), uses the
    closed-form ``log(R) = (θ/2) · B̂`` where ``θ = 2·atan2(|B|, s)``
    and ``B̂`` is the unit bivector axis. Otherwise, truncated series
    ``log(a) = sum_{k≥1} (-1)^{k+1} (a - 1)^k / k`` — convergent when
    ``|a - 1| < 1``.

    **Apple GPU fast path**: Cl(3,0) f32 batched inputs route through
    ``tessera_apple_gpu_clifford_log_cl30_f32``.
    """
    gpu_out = _try_apple_gpu_unary_8x8_cl30_f32(
        a, "tessera_apple_gpu_clifford_log_cl30_f32",
        op_name="clifford_log")
    if gpu_out is not None:
        return gpu_out
    if a.algebra.signature == (3, 0, 0):
        # Treat `a` as a candidate rotor: scalar + bivector parts only.
        s = a.scalar_part()
        bivec = grade_projection(a, 2)
        bivec_norm = np.asarray(norm(bivec))
        # Half-angle: θ/2 = atan2(|bivec|, s).
        half_theta = np.arctan2(bivec_norm, s)
        safe = np.where(bivec_norm > 1e-12, bivec_norm, 1.0)
        scale = (half_theta / safe).astype(a.dtype, copy=False)
        return Multivector(bivec.coefficients * scale[..., None], a.algebra)
    # General power series around 1.
    algebra = a.algebra
    one = Multivector.scalar(1.0, algebra, shape=a.shape, dtype=a.dtype)
    delta = a - one
    result = Multivector.zeros(algebra, shape=a.shape, dtype=a.dtype)
    power = one
    for k in range(1, terms + 1):
        power = geometric_product(power, delta)
        coef = ((-1) ** (k + 1)) / float(k)
        result = result + coef * power
    return result


# ---------------------------------------------------------------------------
# Rotors and rotor sandwiches
# ---------------------------------------------------------------------------

def rotor_from_axis(
    bivector_axis: Multivector, angle: float
) -> Multivector:
    """Construct a rotor ``R = exp(-angle/2 · B̂)`` from a unit bivector.

    The rotor rotates vectors by ``angle`` radians in the plane spanned
    by the input bivector. ``bivector_axis`` need not be normalized;
    the function normalizes internally.
    """
    if 2 not in bivector_axis.active_grades:
        raise TesseraAlgebraError(
            "rotor_from_axis: bivector_axis must have a non-zero grade-2 component."
        )
    bivec = grade_projection(bivector_axis, 2)
    bnorm = float(np.asarray(norm(bivec)).item()) if bivec.shape == () else None
    if bnorm is None:
        # Batch case: produce a batched rotor.
        return exp_mv(-0.5 * float(angle) * bivec / np.asarray(norm(bivec)))
    if bnorm < 1e-12:
        raise TesseraAlgebraError(
            "rotor_from_axis: bivector_axis has zero magnitude."
        )
    unit = bivec / bnorm
    return exp_mv(-0.5 * float(angle) * unit)


def rotor_sandwich(rotor: Multivector, x: Multivector) -> Multivector:
    """Apply ``R x R†`` — rotor-conjugation rotation of ``x``.

    **Apple GPU fast path**: ``Cl(3,0)`` f32 batched inputs route
    through ``tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32``.
    """
    if rotor.algebra != x.algebra:
        raise TesseraAlgebraError(
            f"rotor_sandwich: algebras must match; got {rotor.algebra!r} and {x.algebra!r}."
        )
    gpu_out = _try_apple_gpu_rotor_sandwich_cl30_f32(rotor, x)
    if gpu_out is not None:
        return gpu_out
    return geometric_product(geometric_product(rotor, x), reverse(rotor))


__all__ = [
    "conjugate",
    "exp_mv",
    "geometric_product",
    "grade_involution",
    "grade_projection",
    "inner",
    "left_contraction",
    "log_mv",
    "norm",
    "norm_squared",
    "reverse",
    "rotor_from_axis",
    "rotor_sandwich",
    "wedge",
]
