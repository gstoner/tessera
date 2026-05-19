"""GA6 — Central-difference verification utility for multivector VJPs/JVPs.

``check_grad_geo`` compares an analytic VJP / JVP to a central-difference
numerical estimate, returning the max absolute and relative errors
between them. Used in `tests/unit/test_ga_autodiff.py` to verify every
registered VJP/JVP on randomly-generated multivectors.

Numerical convention: the "loss" is the Frobenius dot product
``L = Σ_k cotangent_k · output_k`` with a fixed random ``cotangent`` of
the same shape as the op's output. Then ``∂L/∂x`` should equal the
analytic VJP evaluated at ``dout = cotangent``.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from tessera.ga.multivector import Multivector


def _flatten_output_to_scalar(out, cotangent: np.ndarray) -> float:
    """Reduce a Multivector or ndarray output to a scalar via Frobenius
    dot product with `cotangent`."""
    if isinstance(out, Multivector):
        return float(np.sum(out.coefficients * cotangent))
    arr = np.asarray(out)
    if arr.shape != cotangent.shape:
        raise ValueError(
            f"output shape {arr.shape} does not match cotangent shape "
            f"{cotangent.shape}; the test cotangent must match the op's output."
        )
    return float(np.sum(arr * cotangent))


def _numerical_grad_for_multivector(
    fn: Callable,
    args: Tuple,
    arg_idx: int,
    cotangent: np.ndarray,
    eps: float,
) -> Multivector:
    """Central-difference gradient of ``L(args) = <fn(args), cotangent>``
    with respect to ``args[arg_idx]`` (a Multivector)."""
    primal = args[arg_idx]
    if not isinstance(primal, Multivector):
        raise TypeError(
            f"_numerical_grad_for_multivector: args[{arg_idx}] must be a Multivector; "
            f"got {type(primal).__name__}."
        )
    algebra = primal.algebra
    grad_coeffs = np.zeros_like(primal.coefficients, dtype=np.float64)
    base_coeffs = primal.coefficients.astype(np.float64, copy=True)
    flat = base_coeffs.ravel()
    grad_flat = grad_coeffs.ravel()
    for idx in range(flat.size):
        original = flat[idx]
        flat[idx] = original + eps
        perturbed_plus = Multivector(base_coeffs.reshape(primal.coefficients.shape), algebra)
        new_args = list(args)
        new_args[arg_idx] = perturbed_plus
        out_plus = fn(*new_args)
        L_plus = _flatten_output_to_scalar(out_plus, cotangent)
        flat[idx] = original - eps
        perturbed_minus = Multivector(base_coeffs.reshape(primal.coefficients.shape), algebra)
        new_args[arg_idx] = perturbed_minus
        out_minus = fn(*new_args)
        L_minus = _flatten_output_to_scalar(out_minus, cotangent)
        flat[idx] = original
        grad_flat[idx] = (L_plus - L_minus) / (2.0 * eps)
    return Multivector(grad_coeffs.reshape(primal.coefficients.shape), algebra)


def check_grad_geo(
    op_name: str,
    fn: Callable,
    args: Tuple,
    *,
    cotangent: np.ndarray | None = None,
    eps: float = 1e-4,
    atol: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Compare the registered VJP for ``op_name`` to central differences.

    Args:
        op_name: name of the registered VJP (e.g. ``"geometric_product"``).
        fn:      forward function (typically the same op).
        args:    tuple of forward arguments; Multivectors are differentiated,
                 others passed through.
        cotangent: optional fixed cotangent (defaults to a deterministic
                 RNG draw matching the op's output shape).
        eps:     central-difference perturbation.
        atol:    absolute tolerance for the comparison.
        seed:    RNG seed for the default cotangent.

    Returns: ``{"max_abs_err": ..., "max_rel_err": ...}``.
    Raises an ``AssertionError`` on mismatch.
    """
    from tessera.autodiff.geometric.registry import get_vjp_geo

    vjp_fn = get_vjp_geo(op_name)
    if vjp_fn is None:
        raise ValueError(f"no VJP registered for {op_name!r}")

    # Forward pass to determine the output shape.
    out = fn(*args)
    if cotangent is None:
        rng = np.random.RandomState(seed)
        if isinstance(out, Multivector):
            cotangent = rng.randn(*out.coefficients.shape).astype(np.float64)
        else:
            arr = np.asarray(out)
            cotangent = rng.randn(*arr.shape).astype(np.float64)

    # Build dout in the form vjp_fn expects.
    dout: Any
    if isinstance(out, Multivector):
        dout = Multivector(cotangent, out.algebra)
    else:
        # Scalar / ndarray output — pass the cotangent directly.
        dout = cotangent

    # Analytic gradients via registered VJP.
    grads = vjp_fn(dout, *args)
    if not isinstance(grads, tuple):
        grads = (grads,)

    max_abs_err = 0.0
    max_rel_err = 0.0
    for arg_idx, (primal, grad) in enumerate(zip(args, grads)):
        if not isinstance(primal, Multivector):
            continue  # non-differentiable arg
        if grad is None:
            continue
        num_grad = _numerical_grad_for_multivector(
            fn, args, arg_idx, cotangent, eps
        )
        diff = grad.coefficients.astype(np.float64) - num_grad.coefficients.astype(np.float64)
        abs_err = float(np.max(np.abs(diff)))
        denom = np.maximum(
            np.abs(grad.coefficients), np.abs(num_grad.coefficients)
        ).astype(np.float64) + 1e-12
        rel_err = float(np.max(np.abs(diff) / denom))
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
        assert abs_err < atol, (
            f"VJP mismatch on {op_name!r} for arg {arg_idx}: "
            f"max abs err {abs_err:.3e} > tol {atol:.3e}.\n"
            f"  analytic[:6] = {grad.coefficients.ravel()[:6]}\n"
            f"  numerical[:6] = {num_grad.coefficients.ravel()[:6]}"
        )
    return {"max_abs_err": max_abs_err, "max_rel_err": max_rel_err}


def check_jvp_geo(
    op_name: str,
    fn: Callable,
    primals: Tuple,
    tangents: Tuple,
    *,
    eps: float = 1e-4,
    atol: float = 1e-3,
) -> dict:
    """Verify a JVP against a finite-difference forward pushforward.

    Numerical JVP: ``(fn(primals + eps · tangents) - fn(primals - eps · tangents)) / (2 eps)``.
    """
    from tessera.autodiff.geometric.registry import get_jvp_geo

    jvp_fn = get_jvp_geo(op_name)
    if jvp_fn is None:
        raise ValueError(f"no JVP registered for {op_name!r}")

    out_jvp = jvp_fn(tangents, primals)

    # Numerical forward difference.
    plus_args = []
    minus_args = []
    for primal, tangent in zip(primals, tangents):
        if isinstance(primal, Multivector) and isinstance(tangent, Multivector):
            plus_args.append(
                Multivector(primal.coefficients + eps * tangent.coefficients, primal.algebra)
            )
            minus_args.append(
                Multivector(primal.coefficients - eps * tangent.coefficients, primal.algebra)
            )
        else:
            plus_args.append(primal)
            minus_args.append(primal)

    out_plus = fn(*plus_args)
    out_minus = fn(*minus_args)
    if isinstance(out_plus, Multivector):
        num_jvp_coeffs = (out_plus.coefficients - out_minus.coefficients) / (2.0 * eps)
        analytic_coeffs = out_jvp.coefficients.astype(np.float64)
        diff = analytic_coeffs - num_jvp_coeffs.astype(np.float64)
    else:
        num_jvp_coeffs = (np.asarray(out_plus) - np.asarray(out_minus)) / (2.0 * eps)
        analytic_coeffs = np.asarray(out_jvp, dtype=np.float64)
        diff = analytic_coeffs - num_jvp_coeffs.astype(np.float64)
    abs_err = float(np.max(np.abs(diff)))
    rel_err = float(np.max(np.abs(diff) / (np.maximum(np.abs(analytic_coeffs), np.abs(num_jvp_coeffs)) + 1e-12)))
    assert abs_err < atol, (
        f"JVP mismatch on {op_name!r}: max abs err {abs_err:.3e} > tol {atol:.3e}."
    )
    return {"max_abs_err": abs_err, "max_rel_err": rel_err}


__all__ = ["check_grad_geo", "check_jvp_geo"]
