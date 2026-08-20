"""Matrix-function and factorization primitives (MC1).

The derivative family that is neither elementwise nor multilinear: `det`,
`logdet`, `inv`, `solve`, `trace`, `eigh`, `kron`, `vec`, `matrix_power`,
`norm`. `autodiff/linear.py` cannot derive these (they are not linear), and the
pointwise-ODE family cannot reach them (they are not scalar functions applied
elementwise) — they are the third derivative family, and
Bright/Edelman/Johnson (arXiv:2501.14787) §7 and §13 hand every rule below over
in closed form.

Written up as MC1 in ``docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md``.

Two contracts this module holds to, both from that review:

* **Never materialize a Jacobian that a rule expresses.** `d(A^-1) =
  -A^-1 dA A^-1` as a Jacobian is `-(A^-T kron A^-1)`, an `n^2 x n^2` object
  costing `O(n^4)` to apply; as the rule it is two `n x n` products. Every rule
  here is written in the operator form (notes §2.6, §3.3.3).
* **Degeneracy fails closed.** `eigh`'s eigenvector coupling is
  `1/(w_j - w_i)`, which has no limit at a crossing; it routes through the same
  declared `degeneracy_policy` semantic key as `svd` (see
  ``autodiff/degeneracy.py``) rather than emitting a plausible-looking number.

Reference-tier discipline matches ``solvers_ops.py``: fp64 accumulation,
fail-closed shape and singularity contracts, and a real VJP *and* JVP for every
entry so neither mode is the odd one out.
"""

from __future__ import annotations

import numpy as np

from .autodiff.degeneracy import eigh_coupling
from .custom import custom_primitive

_LINALG_DIAGNOSTIC_CODE = "E_LINALG_CONTRACT"


# ── shared contract helpers ─────────────────────────────────────────────────
def _square(A, op: str) -> np.ndarray:
    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError(
            f"{_LINALG_DIAGNOSTIC_CODE}: {op} requires a square matrix; got "
            f"shape {arr.shape}."
        )
    return arr


def _nonsingular(A: np.ndarray, op: str) -> np.ndarray:
    """Fail closed on a singular matrix rather than returning inf/NaN."""
    cond = np.linalg.cond(A)
    if not np.all(np.isfinite(cond)) or np.any(cond > 1.0 / np.finfo(np.float64).eps):
        raise ValueError(
            f"{_LINALG_DIAGNOSTIC_CODE}: {op} is not defined at this input — "
            f"the matrix is numerically singular (condition number "
            f"{np.max(cond):.3e} at or beyond 1/eps). Both the value and its "
            f"derivative fail to exist here; returning a finite number would "
            f"be returning a wrong one."
        )
    return A


def _mT(x: np.ndarray) -> np.ndarray:
    return np.swapaxes(x, -1, -2)


def _storage_dtype(x) -> np.dtype:
    """The dtype a result should be returned in (Decision #15a).

    Accumulation happens in fp64 — that is the reference tier's contract — but
    the *storage* dtype belongs to the tensor, so an fp32 input must not come
    back as fp64. Integer and boolean inputs promote to fp64, since these are
    real-valued functions.
    """
    dt = np.asarray(x).dtype
    if dt.kind in ("i", "u", "b"):
        return np.dtype(np.float64)
    return dt


def _as_storage(value, like) -> np.ndarray:
    return np.asarray(value).astype(_storage_dtype(like), copy=False)


# ── trace: linear, so a transpose rule is the whole derivative ──────────────
def _trace_impl(A):
    arr = _square(A, "trace")
    return _as_storage(np.einsum("...ii->...", arr), A)


def _trace_vjp(dout, A, **_):
    arr = _square(A, "trace")
    eye = np.eye(arr.shape[-1])
    return (np.asarray(dout, dtype=np.float64)[..., None, None] * eye,)


def _trace_jvp(primals, tangents, **_):
    return _trace_impl(primals[0]), _trace_impl(tangents[0])


trace = custom_primitive("trace", vjp=_trace_vjp, jvp=_trace_jvp)(_trace_impl)


# ── vec / unvec: the isomorphism that makes matrix Jacobians expressible ────
def _vec_impl(A):
    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(
            f"{_LINALG_DIAGNOSTIC_CODE}: vec expects at least a matrix; got "
            f"shape {arr.shape}."
        )
    # COLUMN-major stacking, which is what makes the Kronecker identity
    # `(B kron C) vec(Y) = vec(C Y B^T)` hold (notes §3.3.1). Row-major would
    # silently transpose the identity and every rule built on it.
    return _as_storage(
        np.reshape(_mT(arr), arr.shape[:-2] + (arr.shape[-1] * arr.shape[-2],)), A)


def _vec_vjp(dout, A, **_):
    arr = np.asarray(A, dtype=np.float64)
    m, n = arr.shape[-2], arr.shape[-1]
    d = np.asarray(dout, dtype=np.float64)
    return (_mT(np.reshape(d, d.shape[:-1] + (n, m))),)


def _vec_jvp(primals, tangents, **_):
    return _vec_impl(primals[0]), _vec_impl(tangents[0])


vec = custom_primitive("vec", vjp=_vec_vjp, jvp=_vec_jvp)(_vec_impl)


# ── kron: bilinear, and a cost trap if it is ever materialized (MC4) ────────
def _kron_impl(A, B):
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    # Batch-aware over leading axes: `np.kron` treats extra dimensions as part
    # of the product, which is not the batched meaning. The einsum states the
    # intended index structure directly, and batching is then a real claim
    # rather than a declaration the category map makes on the op's behalf.
    p, q = a.shape[-2], a.shape[-1]
    r, s_ = b.shape[-2], b.shape[-1]
    blocks = np.einsum("...ij,...rs->...irjs", a, b)
    return _as_storage(
        blocks.reshape(blocks.shape[:-4] + (p * r, q * s_)), A)


def _kron_vjp(dout, A, B, **_):
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    d = np.asarray(dout, dtype=np.float64)
    p, q = a.shape[-2], a.shape[-1]
    r, s_ = b.shape[-2], b.shape[-1]
    # Blocks: dout[..., i*r:(i+1)*r, j*s:(j+1)*s] pairs A[..., i, j] with B.
    blocks = d.reshape(d.shape[:-2] + (p, r, q, s_))
    dA = np.einsum("...irjs,...rs->...ij", blocks, b)
    dB = np.einsum("...irjs,...ij->...rs", blocks, a)
    return (dA, dB)


def _kron_jvp(primals, tangents, **_):
    A, B = primals
    dA, dB = tangents
    return _kron_impl(A, B), _kron_impl(dA, B) + _kron_impl(A, dB)


kron = custom_primitive("kron", vjp=_kron_vjp, jvp=_kron_jvp)(_kron_impl)


# ── det / logdet (notes §7.1) ───────────────────────────────────────────────
def _det_impl(A):
    return _as_storage(np.linalg.det(_square(A, "det")), A)


def _det_vjp(dout, A, **_):
    arr = _nonsingular(_square(A, "det"), "det backward")
    d = np.asarray(dout, dtype=np.float64)
    # grad det = det(A) * A^-T  (the adjugate, transposed)
    return (d[..., None, None] * _det_impl(arr)[..., None, None]
            * _mT(np.linalg.inv(arr)),)


def _det_jvp(primals, tangents, **_):
    arr = _nonsingular(_square(primals[0], "det"), "det forward-mode")
    dA = np.asarray(tangents[0], dtype=np.float64)
    value = _det_impl(arr)
    # d(det A) = det(A) * tr(A^-1 dA)
    return value, value * np.einsum("...ii->...", np.linalg.solve(arr, dA))


det = custom_primitive("det", vjp=_det_vjp, jvp=_det_jvp)(_det_impl)


def _logdet_impl(A):
    arr = _square(A, "logdet")
    sign, logabsdet = np.linalg.slogdet(arr)
    if np.any(sign <= 0.0):
        raise ValueError(
            f"{_LINALG_DIAGNOSTIC_CODE}: logdet requires a positive determinant; "
            f"got sign {np.asarray(sign).tolist()}. log|det| of a negative-"
            f"determinant matrix is a different function — take `log(abs(det))` "
            f"explicitly if that is what you meant."
        )
    return _as_storage(logabsdet, A)


def _logdet_vjp(dout, A, **_):
    arr = _nonsingular(_square(A, "logdet"), "logdet backward")
    d = np.asarray(dout, dtype=np.float64)
    return (d[..., None, None] * _mT(np.linalg.inv(arr)),)


def _logdet_jvp(primals, tangents, **_):
    arr = _nonsingular(_square(primals[0], "logdet"), "logdet forward-mode")
    dA = np.asarray(tangents[0], dtype=np.float64)
    # d log det A = tr(A^-1 dA)
    return _logdet_impl(arr), np.einsum("...ii->...", np.linalg.solve(arr, dA))


logdet = custom_primitive("logdet", vjp=_logdet_vjp, jvp=_logdet_jvp)(_logdet_impl)


# ── inv / solve (notes §2.6, §6.3) ──────────────────────────────────────────
def _inv_impl(A):
    return _as_storage(
        np.linalg.inv(_nonsingular(_square(A, "inv"), "inv")), A)


def _inv_vjp(dout, A, **_):
    Ainv = _inv_impl(A)
    d = np.asarray(dout, dtype=np.float64)
    # From d(A^-1) = -A^-1 dA A^-1, the adjoint is -A^-T dOut A^-T. Two matrix
    # products; the Jacobian form -(A^-T kron A^-1) is n^2 x n^2 and never built.
    return (-_mT(Ainv) @ d @ _mT(Ainv),)


def _inv_jvp(primals, tangents, **_):
    Ainv = _inv_impl(primals[0])
    dA = np.asarray(tangents[0], dtype=np.float64)
    return Ainv, -Ainv @ dA @ Ainv


inv = custom_primitive("inv", vjp=_inv_vjp, jvp=_inv_jvp)(_inv_impl)


def _solve_impl(A, b):
    arr = _nonsingular(_square(A, "solve"), "solve")
    return _as_storage(np.linalg.solve(arr, np.asarray(b, dtype=np.float64)), b)


def _solve_vjp(dout, A, b, **_):
    arr = _nonsingular(_square(A, "solve"), "solve backward")
    x = _solve_impl(arr, b)
    d = np.asarray(dout, dtype=np.float64)
    # The adjoint solve (notes §6.3): one transposed solve gives db, and dA
    # follows as an outer product — never an explicit dA/dp Jacobian.
    db = np.linalg.solve(_mT(arr), d)
    if x.ndim == arr.ndim - 1:
        dA = -db[..., :, None] * x[..., None, :]
    else:
        dA = -db @ _mT(x)
    return (dA, db)


def _solve_jvp(primals, tangents, **_):
    A, b = primals
    dA, db = tangents
    arr = _nonsingular(_square(A, "solve"), "solve forward-mode")
    x = _solve_impl(arr, b)
    rhs = np.asarray(db, dtype=np.float64) - np.asarray(dA, dtype=np.float64) @ x
    return x, np.linalg.solve(arr, rhs)


solve = custom_primitive("solve", vjp=_solve_vjp, jvp=_solve_jvp)(_solve_impl)


# ── matrix_power ────────────────────────────────────────────────────────────
def _matrix_power_impl(A, n: int = 1):
    return _as_storage(
        np.linalg.matrix_power(_square(A, "matrix_power"), int(n)), A)


def _matrix_power_vjp(dout, A, *, n: int = 1, **_):
    arr = _square(A, "matrix_power")
    k = int(n)
    if k < 0:
        arr = _inv_impl(arr)
        k = -k
    d = np.asarray(dout, dtype=np.float64)
    # d(A^k) = sum_j A^j dA A^(k-1-j)  =>  adjoint sums A^jT dOut A^(k-1-j)T.
    dA = np.zeros_like(arr)
    for j in range(k):
        left = np.linalg.matrix_power(arr, j)
        right = np.linalg.matrix_power(arr, k - 1 - j)
        dA = dA + _mT(left) @ d @ _mT(right)
    if int(n) < 0:
        Ainv = _inv_impl(np.asarray(A, dtype=np.float64))
        dA = -_mT(Ainv) @ dA @ _mT(Ainv)
    return (dA,)


def _matrix_power_jvp(primals, tangents, *, n: int = 1, **_):
    arr = _square(primals[0], "matrix_power")
    dA = np.asarray(tangents[0], dtype=np.float64)
    k = int(n)
    base, base_d = arr, dA
    if k < 0:
        base = _inv_impl(arr)
        base_d = -base @ dA @ base
        k = -k
    out = np.zeros_like(arr)
    for j in range(k):
        out = out + (np.linalg.matrix_power(base, j) @ base_d
                     @ np.linalg.matrix_power(base, k - 1 - j))
    return _matrix_power_impl(arr, n), out


matrix_power = custom_primitive(
    "matrix_power", vjp=_matrix_power_vjp, jvp=_matrix_power_jvp,
)(_matrix_power_impl)


# ── norm: `ord` is a SEMANTIC key, so it fails closed on absence of support ─
_SUPPORTED_NORMS = ("fro", "nuc")


def _check_ord(ord: str, where: str) -> str:
    if ord not in _SUPPORTED_NORMS:
        raise ValueError(
            f"{_LINALG_DIAGNOSTIC_CODE}: {where} does not implement ord={ord!r}; "
            f"supported: {list(_SUPPORTED_NORMS)}. `ord` selects which function "
            f"is being differentiated, so it is a semantic key (Decision #21a) "
            f"and is never silently defaulted to another norm."
        )
    return ord


def _norm_impl(A, *, ord: str = "fro"):
    arr = np.asarray(A, dtype=np.float64)
    _check_ord(ord, "norm")
    if ord == "fro":
        return _as_storage(np.sqrt(np.sum(arr * arr, axis=(-2, -1))), A)
    return _as_storage(
        np.sum(np.linalg.svd(arr, compute_uv=False), axis=-1), A)


def _norm_vjp(dout, A, *, ord: str = "fro", **_):
    arr = np.asarray(A, dtype=np.float64)
    _check_ord(ord, "norm backward")
    d = np.asarray(dout, dtype=np.float64)
    if ord == "fro":
        value = _norm_impl(arr, ord="fro")
        if np.any(value == 0.0):
            raise ValueError(
                f"{_LINALG_DIAGNOSTIC_CODE}: norm(ord='fro') is not "
                f"differentiable at the zero matrix — the gradient A/||A|| has "
                f"no limit there."
            )
        # grad ||A||_F = A / ||A||_F  (notes §5.1, via the Frobenius inner product)
        return (d[..., None, None] * arr / value[..., None, None],)
    U, _s, Vt = np.linalg.svd(arr, full_matrices=False)
    # grad of the nuclear norm is U V^T — well defined at full rank even when
    # singular values repeat, which is why it needs no degeneracy policy.
    return (d[..., None, None] * (U @ Vt),)


def _norm_jvp(primals, tangents, *, ord: str = "fro", **_):
    arr = np.asarray(primals[0], dtype=np.float64)
    dA = np.asarray(tangents[0], dtype=np.float64)
    (grad,) = _norm_vjp(np.ones(arr.shape[:-2] or ()), arr, ord=ord)
    return _norm_impl(arr, ord=ord), np.sum(grad * dA, axis=(-2, -1))


norm = custom_primitive("norm", vjp=_norm_vjp, jvp=_norm_jvp)(_norm_impl)


# ── eigh: the symmetric eigenproblem (notes §13.2.1) ────────────────────────
def _eigh_impl(S):
    arr = _square(S, "eigh")
    sym = 0.5 * (arr + _mT(arr))
    w, Q = np.linalg.eigh(sym)
    return _as_storage(w, S), _as_storage(Q, S)


def _eigh_vjp(dout, S, **_):
    """S = Q diag(w) Q^T for symmetric S.

    ``dw_i = q_i^T dS q_i`` (Hellmann-Feynman) is defined for simple
    eigenvalues; the eigenvector term couples through ``1/(w_j - w_i)`` and
    fails closed at a crossing under the declared policy.
    """
    arr = _square(S, "eigh")
    w, Q = _eigh_impl(arr)
    if isinstance(dout, tuple) and len(dout) == 2:
        dw, dQ = (np.asarray(t, dtype=np.float64) for t in dout)
    else:
        dw = np.asarray(dout, dtype=np.float64)
        dQ = np.zeros_like(Q)
    QtdQ = _mT(Q) @ dQ
    F = eigh_coupling(w, dw=dw, antisymmetric_parts=(QtdQ - _mT(QtdQ),), op="eigh")
    inner = np.zeros_like(QtdQ)
    idx = np.arange(w.shape[-1])
    inner[..., idx, idx] = dw
    inner = inner + F * QtdQ
    dS = Q @ inner @ _mT(Q)
    # S enters only through its symmetric part, so the cotangent is symmetrized.
    return (0.5 * (dS + _mT(dS)),)


def _eigh_jvp(primals, tangents, **_):
    arr = _square(primals[0], "eigh")
    dS = np.asarray(tangents[0], dtype=np.float64)
    dS = 0.5 * (dS + _mT(dS))
    w, Q = _eigh_impl(arr)
    P = _mT(Q) @ dS @ Q
    idx = np.arange(w.shape[-1])
    dw = P[..., idx, idx]
    F = eigh_coupling(w, op="eigh", admit_generalized=False, mode="forward-mode")
    dQ = Q @ (F * P)
    return (w, Q), (dw, dQ)


eigh = custom_primitive("eigh", vjp=_eigh_vjp, jvp=_eigh_jvp)(_eigh_impl)


__all__ = [
    "det", "logdet", "inv", "solve", "trace", "eigh",
    "kron", "vec", "matrix_power", "norm",
]
