"""Tracer-first frontend authority and legacy differential certificates.

E2E-REAL-6 cannot delete the AST frontend in one step.  This module makes the
transition explicit: a concrete tensor signature is captured by the tracer and
promoted to canonical Graph IR, while a retained AST module is admitted only as
a candidate/oracle with a content-addressed structural and numerical
certificate.  Effectful programs fail closed because executing them twice would
change observable state or stochastic identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .effects import Effect, infer_graph_effects
from .graph_ir import GraphIRModule, IROp, IRType


SCHEMA = "tessera.frontend_differential.v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _op_signature(
    op: IROp,
    value_ids: Mapping[str, str],
    value_types: Mapping[str, IRType],
) -> Mapping[str, Any]:
    def canonical_value(name: str) -> str:
        return value_ids.get(name.lstrip("%"), "external")

    kwargs = dict(op.kwargs)
    if op.op_name in {"tessera.fft", "tessera.ifft", "tessera.rfft", "tessera.dct"}:
        operand_type = value_types.get(op.operands[0].lstrip("%")) if op.operands else None
        axis = int(kwargs.get("axis", -1))
        if operand_type is not None and operand_type.rank:
            normalized_axis = axis if axis >= 0 else operand_type.rank + axis
            if 0 <= normalized_axis < operand_type.rank:
                try:
                    default_length = int(operand_type.shape[normalized_axis])
                except ValueError:
                    default_length = None
                if kwargs.get("logical_length") == default_length:
                    kwargs.pop("logical_length", None)
    elif op.op_name in {"tessera.spectral_filter", "tessera.spectral_conv"}:
        operand_type = value_types.get(op.operands[0].lstrip("%")) if op.operands else None
        axis = int(kwargs.get("axis", -1))
        if operand_type is not None and operand_type.rank:
            normalized_axis = axis if axis >= 0 else operand_type.rank + axis
            if 0 <= normalized_axis < operand_type.rank:
                try:
                    default_length = int(operand_type.shape[normalized_axis])
                except ValueError:
                    default_length = None
                if kwargs.get("logical_length") == default_length:
                    kwargs.pop("logical_length", None)
    elif op.op_name in {"tessera.stft", "tessera.istft"} and len(op.operands) >= 2:
        window_type = value_types.get(op.operands[1].lstrip("%"))
        if window_type is not None and window_type.shape:
            try:
                window_length = int(window_type.shape[-1])
            except ValueError:
                window_length = None
            if kwargs.get("logical_length") == window_length:
                kwargs.pop("logical_length", None)
    return {
        "op": op.op_name,
        "operands": [canonical_value(name) for name in op.operands],
        "result_count": len(op.result_names),
        "kwargs": {
            str(key): repr(value)
            for key, value in sorted(kwargs.items())
            if not str(key).startswith("_")
        },
    }


def graph_signature(module: GraphIRModule) -> tuple[Mapping[str, Any], ...]:
    """Return an SSA-name-independent straight-line topology signature."""
    if len(module.functions) != 1:
        raise ValueError("frontend differential requires one Graph function")
    function = module.functions[0]
    value_ids = {arg.name.lstrip("%"): f"arg:{index}"
                 for index, arg in enumerate(function.args)}
    value_types = {arg.name.lstrip("%"): arg.ir_type for arg in function.args}
    signature: list[Mapping[str, Any]] = []
    for index, op in enumerate(function.body):
        signature.append(_op_signature(op, value_ids, value_types))
        for result_index, name in enumerate(op.result_names):
            value_ids[name.lstrip("%")] = f"op:{index}:{result_index}"
            if result_index < len(op.inferred_types):
                value_types[name.lstrip("%")] = op.inferred_types[result_index]
            elif op.inferred_type is not None:
                value_types[name.lstrip("%")] = op.inferred_type
    return tuple(signature)


@dataclass(frozen=True)
class FrontendDifferentialCertificate:
    contract: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return str(self.contract["digest"])

    def validate(self) -> None:
        body = dict(self.contract)
        actual = str(body.pop("digest", ""))
        if body.get("schema") != SCHEMA or actual != _digest(body):
            raise ValueError("frontend differential certificate has stale identity")
        if not body.get("structural_match") or not body.get("numerical_match"):
            raise ValueError("legacy frontend candidate lacks differential parity")


def certify_frontends(
    *,
    legacy_module: GraphIRModule,
    tracer_module: GraphIRModule,
    legacy_outputs: Sequence[Any],
    tracer_outputs: Sequence[Any],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> FrontendDifferentialCertificate:
    """Certify one pure concrete signature or reject it fail-closed."""
    tracer_ops = [op for function in tracer_module.functions for op in function.body]
    effect, offenders = infer_graph_effects(tracer_ops)
    if effect != Effect.pure:
        raise ValueError(
            "frontend differential refuses effectful or stochastic programs: "
            + ", ".join(offenders)
        )
    legacy_signature = graph_signature(legacy_module)
    tracer_signature = graph_signature(tracer_module)
    structural_match = legacy_signature == tracer_signature
    if len(legacy_outputs) != len(tracer_outputs):
        numerical_match = False
        max_error = float("inf")
    else:
        numerical_match = True
        max_error = 0.0
        for expected, actual in zip(legacy_outputs, tracer_outputs):
            lhs = np.asarray(expected)
            rhs = np.asarray(actual)
            if lhs.shape != rhs.shape:
                numerical_match = False
                max_error = float("inf")
                break
            if lhs.size:
                max_error = max(max_error, float(np.max(np.abs(lhs - rhs))))
            numerical_match &= bool(np.allclose(lhs, rhs, rtol=rtol, atol=atol,
                                                equal_nan=True))
    body = {
        "schema": SCHEMA,
        "legacy_graph_digest": _digest(legacy_signature),
        "tracer_graph_digest": _digest(tracer_signature),
        "structural_match": structural_match,
        "numerical_match": numerical_match,
        "max_abs_error": max_error,
        "rtol": float(rtol),
        "atol": float(atol),
    }
    certificate = FrontendDifferentialCertificate({**body, "digest": _digest(body)})
    certificate.validate()
    return certificate


__all__ = [
    "FrontendDifferentialCertificate",
    "SCHEMA",
    "certify_frontends",
    "graph_signature",
]
