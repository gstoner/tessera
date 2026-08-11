"""Executable proof obligations for compiler-owned derivative rules.

Finite differences are one witness, not the contract.  This module records the
algebraic identity each native Graph tangent owes and supplies reusable
directional-derivative and forward/reverse-duality checks.  Nonsmooth and
stochastic operations carry an explicit policy instead of inheriting a smooth
finite-difference test that would be mathematically meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

import numpy as np


class ProofKind(str, Enum):
    LINEAR_IDENTITY = "linear_identity"
    DIRECTIONAL_AND_DUALITY = "directional_and_duality"
    STOCHASTIC_REPLAY = "stochastic_replay"
    IMPLICIT_RESIDUAL = "implicit_residual"


@dataclass(frozen=True)
class DerivativeProofContract:
    family: str
    kind: ProofKind
    identity: str
    domain: str
    boundary_policy: str = "not_applicable"


def _contracts() -> tuple[DerivativeProofContract, ...]:
    linear = {
        "add": "d(x+y)=dx+dy",
        "sub": "d(x-y)=dx-dy",
        "transpose": "d transpose(x)=transpose(dx)",
        "reshape": "d reshape(x)=reshape(dx)",
        "squeeze": "d squeeze(x)=squeeze(dx)",
        "unsqueeze": "d unsqueeze(x)=unsqueeze(dx)",
        "expand": "d expand(x)=expand(dx)",
        "broadcast": "d broadcast(x)=broadcast(dx)",
        "permute": "d permute(x)=permute(dx)",
        "flatten": "d flatten(x)=flatten(dx)",
        "view": "d view(x)=view(dx)",
        "fft": "d FFT(x)=FFT(dx) with identical normalization",
        "ifft": "d IFFT(x)=IFFT(dx) with identical normalization",
        "rfft": "d RFFT(x)=RFFT(dx) with identical packed-real identity",
        "irfft": "d IRFFT(x)=IRFFT(dx) with identical logical length",
        "dct": "d DCT_k(x)=DCT_k(dx) with identical type/normalization",
        "all_reduce": "d all_reduce(x)=all_reduce(dx) for sum/mean",
        "reduce_scatter": "d reduce_scatter(x)=reduce_scatter(dx) for sum/mean",
        "all_gather": "d all_gather(x)=all_gather(dx)",
        "all_to_all": "d all_to_all(x)=all_to_all(dx)",
        "reduce": "d reduce(x)=reduce(dx) for sum/mean",
    }
    out = [
        DerivativeProofContract(name, ProofKind.LINEAR_IDENTITY, identity,
                                "floating tensors; legal static shape/axis")
        for name, identity in linear.items()
    ]
    out.extend((
        DerivativeProofContract(
            "mul", ProofKind.DIRECTIONAL_AND_DUALITY,
            "d(x*y)=dx*y+x*dy", "floating tensors"),
        DerivativeProofContract(
            "matmul", ProofKind.DIRECTIONAL_AND_DUALITY,
            "d(A@B)=dA@B+A@dB", "legal contraction shapes"),
        DerivativeProofContract(
            "spectral_filter", ProofKind.DIRECTIONAL_AND_DUALITY,
            "d(X*H)=dX*H+X*dH",
            "equal-shape complex spectra"),
        DerivativeProofContract(
            "spectral_conv", ProofKind.DIRECTIONAL_AND_DUALITY,
            "d conv(x,w)=conv(dx,w)+conv(x,dw)",
            "real floating tensors; legal full-convolution axis"),
        DerivativeProofContract(
            "stft", ProofKind.DIRECTIONAL_AND_DUALITY,
            "d STFT(x,w)=STFT(dx,w)+STFT(x,dw) with fixed framing",
            "real signal/window; fixed hop, axis, padding, and normalization"),
        DerivativeProofContract(
            "istft", ProofKind.LINEAR_IDENTITY,
            "d_X ISTFT(X,w)=ISTFT(dX,w) for a fixed window",
            "complex spectrum; fixed window, hop, axis, length, and normalization",
            "active window products are rejected because overlap-add normalization "
            "depends quadratically on the window"),
        DerivativeProofContract(
            "tanh", ProofKind.DIRECTIONAL_AND_DUALITY,
            "dy=dx*(1-tanh(x)^2)", "finite floating inputs"),
        DerivativeProofContract(
            "sigmoid", ProofKind.DIRECTIONAL_AND_DUALITY,
            "dy=dx*y*(1-y)", "finite floating inputs"),
        DerivativeProofContract(
            "softmax", ProofKind.DIRECTIONAL_AND_DUALITY,
            "dy=y*(dx-sum(y*dx,axis))", "finite non-empty reduction axis"),
        DerivativeProofContract(
            "layer_norm", ProofKind.DIRECTIONAL_AND_DUALITY,
            "dy=J_layer_norm(x,scale,bias)[dx,dscale,dbias]",
            "finite floating tensors; positive epsilon; non-empty last axis"),
        DerivativeProofContract(
            "rmsnorm", ProofKind.DIRECTIONAL_AND_DUALITY,
            "dy=J_rmsnorm(x,scale)[dx,dscale]",
            "finite floating tensors; positive epsilon; non-empty last axis"),
        DerivativeProofContract(
            "stop_gradient", ProofKind.LINEAR_IDENTITY,
            "d stop_gradient(x)=0", "all supported floating tensors"),
        DerivativeProofContract(
            "dropout", ProofKind.STOCHASTIC_REPLAY,
            "dy=philox_mask(seed,counter,p)*dx/(1-p)",
            "training dropout with explicit Philox seed/counter",
            "unseeded active dropout is rejected"),
        DerivativeProofContract(
            "es_low_rank_correction", ProofKind.STOCHASTIC_REPLAY,
            "d_x correction(x,key)=correction(dx,key)",
            "fixed key/epoch/member ids; floating x",
            "integer key/member ids and RNG bits have no tangent"),
        DerivativeProofContract(
            "solver_implicit", ProofKind.IMPLICIT_RESIDUAL,
            "R_u*du=-R_x*dx at the converged primal solution",
            "converged residual with nonsingular declared linearization",
            "singular/nonconverged solves are rejected"),
    ))
    return tuple(out)


DERIVATIVE_PROOF_MATRIX: dict[str, DerivativeProofContract] = {
    contract.family: contract for contract in _contracts()
}


def central_directional_difference(
    fn: Callable[..., np.ndarray], primals: Iterable[np.ndarray],
    tangents: Iterable[np.ndarray], *, step: float = 1.0e-3,
) -> np.ndarray:
    def proof_precision(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        dtype = np.complex128 if np.iscomplexobj(array) else np.float64
        return np.asarray(array, dtype=dtype)

    xs = tuple(proof_precision(x) for x in primals)
    vs = tuple(proof_precision(v) for v in tangents)
    plus = fn(*(x + step * v for x, v in zip(xs, vs)))
    minus = fn(*(x - step * v for x, v in zip(xs, vs)))
    return (np.asarray(plus) - np.asarray(minus)) / (2.0 * step)


def assert_directional_close(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float = 2.0e-4,
    atol: float = 2.0e-5,
) -> None:
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_forward_reverse_duality(
    tangent: np.ndarray, output_cotangent: np.ndarray,
    input_tangents: Iterable[np.ndarray], input_cotangents: Iterable[np.ndarray],
    *, rtol: float = 2.0e-5, atol: float = 2.0e-6,
) -> None:
    lhs = float(np.vdot(np.asarray(tangent), np.asarray(output_cotangent)).real)
    rhs = sum(
        float(np.vdot(np.asarray(v), np.asarray(cotangent)).real)
        for v, cotangent in zip(input_tangents, input_cotangents)
    )
    np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)


__all__ = [
    "DERIVATIVE_PROOF_MATRIX", "DerivativeProofContract", "ProofKind",
    "assert_directional_close", "assert_forward_reverse_duality",
    "central_directional_difference",
]
