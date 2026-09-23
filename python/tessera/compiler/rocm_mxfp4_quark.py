"""Fail-closed assessment of Quark MXFP4 checkpoint projections.

Shape agreement is not byte-order or numerical proof.  Quark's ``reorder``
packing and dynamic MXFP4 activation scales are separate contracts from the
proved gfx1201 W4A8 package.  This module reads metadata only; it never
converts weights or admits an executable route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .rocm_mxfp4 import MXFP4_QUARK_REORDER_LAYOUT_V1
from .rocm_mxfp4_native import MXFP4RouteReceipt, select_mxfp4_schedule


@dataclass(frozen=True)
class QuarkMXFP4ProjectionAssessment:
    module: str
    n: int
    k: int
    weight_bytes: int
    scale_bytes: int
    source_layout: str
    activation_storage: str
    route: MXFP4RouteReceipt
    unresolved_contracts: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "module": self.module,
            "n": self.n,
            "k": self.k,
            "weight_bytes": self.weight_bytes,
            "scale_bytes": self.scale_bytes,
            "source_layout": self.source_layout,
            "activation_storage": self.activation_storage,
            "route": self.route.as_dict(),
            "unresolved_contracts": list(self.unresolved_contracts),
        }


def _record(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _u8_shape(tensor: Mapping[str, Any], name: str) -> tuple[int, int]:
    if tensor.get("dtype") != "U8":
        raise ValueError(f"{name} must be uint8 checkpoint storage")
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
    ):
        raise ValueError(f"{name} must have a positive rank-two shape")
    return shape[0], shape[1]


def assess_quark_mxfp4_projection(
    config: Mapping[str, Any],
    *,
    module: str,
    m: int,
    weight: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> QuarkMXFP4ProjectionAssessment:
    """Describe one Quark W4A4 projection without treating it as W4A8.

    ``weight`` and ``scale`` are safetensors-header entries for ``module.weight``
    and ``module.weight_scale``.  Their payload bytes need a separate Quark
    dequantization oracle before any conversion can be implemented.
    """
    if not module or module.endswith((".weight", ".weight_scale")):
        raise ValueError("module must identify a projection, not a tensor")
    quant = _record(config.get("quantization_config"), "quantization_config")
    export = _record(quant.get("export"), "quantization_config.export")
    global_quant = _record(
        quant.get("global_quant_config"), "quantization_config.global_quant_config"
    )
    weight_quant = _record(global_quant.get("weight"), "weight quantization")
    input_quant = _record(global_quant.get("input_tensors"), "input quantization")
    if quant.get("quant_method") != "quark" or (
        export.get("weight_format") != "real_quantized"
        or export.get("pack_method") != "reorder"
    ):
        raise ValueError("projection requires Quark real-quantized reorder export")
    for name, policy in (("weight", weight_quant), ("activation", input_quant)):
        if (
            policy.get("dtype") != "fp4"
            or policy.get("group_size") != 32
            or policy.get("scale_format") != "e8m0"
        ):
            raise ValueError(f"{name} is not OCP MXFP4 K32/E8M0")
    if weight_quant.get("is_dynamic") is not False or (
        input_quant.get("is_dynamic") is not True
    ):
        raise ValueError("projection requires static weights and dynamic activations")
    if quant.get("layer_quant_config") or quant.get("layer_type_quant_config"):
        raise ValueError("per-layer Quark overrides need a separate assessment")
    excluded = quant.get("exclude", [])
    if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
        raise ValueError("Quark exclusion list must contain module names")
    if any(module == item or module.startswith(item + ".") for item in excluded):
        raise ValueError(f"{module} is excluded from MXFP4 quantization")
    n, packed_k = _u8_shape(weight, "packed weight")
    scale_n, scale_k = _u8_shape(scale, "E8M0 scale")
    k = packed_k * 2
    if k % 32 or (scale_n, scale_k) != (n, k // 32):
        raise ValueError("Quark packed weight and E8M0 scale shapes disagree")
    schedule = select_mxfp4_schedule(m, n, k)
    route = MXFP4RouteReceipt(
        False,
        schedule.workload,
        MXFP4_QUARK_REORDER_LAYOUT_V1,
        None,
        None,
        "Quark reorder bytes have no proved converter and dynamic MXFP4 "
        "activations require a distinct W4A4 ABI",
        schedule,
    )
    return QuarkMXFP4ProjectionAssessment(
        module=module,
        n=n,
        k=k,
        weight_bytes=n * packed_k,
        scale_bytes=n * scale_k,
        source_layout=MXFP4_QUARK_REORDER_LAYOUT_V1,
        activation_storage="mxfp4_e2m1",
        route=route,
        unresolved_contracts=(
            "Quark reorder bytes need an independent E2M1 nibble/scale oracle",
            "dynamic MXFP4 activation groups need a distinct W4A4 lowering and ABI",
        ),
    )
