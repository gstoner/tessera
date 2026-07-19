"""CUDA floating-point semantic routes used by the NVIDIA compiler.

The contract intentionally separates IEEE arithmetic operators, CUDA
``libdevice`` functions, and PTX approximate instructions.  Optimization level
does not select a math route: a lowering must name one explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CUDA_MATH_CONTRACT_VERSION = "tessera.nvidia.cuda_math.v1"

MathRoute = Literal[
    "ieee_operator",
    "cuda_libdevice",
    "ptx_ex2_approx_f32",
]


@dataclass(frozen=True)
class CudaMathContract:
    route: MathRoute
    rounding: str
    ftz: bool
    approximate: bool
    max_ulp: float | None
    accuracy_scope: str

    @property
    def requires_nonzero_atol(self) -> bool:
        """Whether near-zero comparisons need an absolute error budget."""
        return self.approximate


_CONTRACTS: dict[MathRoute, CudaMathContract] = {
    "ieee_operator": CudaMathContract(
        route="ieee_operator",
        rounding="round_to_nearest_ties_to_even",
        ftz=False,
        approximate=False,
        max_ulp=0.0,
        accuracy_scope="CUDA built-in arithmetic operator",
    ),
    "cuda_libdevice": CudaMathContract(
        route="cuda_libdevice",
        rounding="function_defined",
        ftz=False,
        approximate=True,
        max_ulp=None,
        accuracy_scope="CUDA libdevice function-specific specification",
    ),
    "ptx_ex2_approx_f32": CudaMathContract(
        route="ptx_ex2_approx_f32",
        rounding="approximation_defined",
        ftz=False,
        approximate=True,
        max_ulp=2.0,
        accuracy_scope="PTX ex2.approx.f32 over its full input range",
    ),
}


def cuda_math_contract(route: MathRoute) -> CudaMathContract:
    """Return the immutable semantic contract for an emitted CUDA math route."""
    try:
        return _CONTRACTS[route]
    except KeyError as exc:  # defensive for untyped callers/deserialized input
        raise ValueError(f"unknown CUDA math route {route!r}") from exc


__all__ = [
    "CUDA_MATH_CONTRACT_VERSION",
    "CudaMathContract",
    "MathRoute",
    "cuda_math_contract",
]
