"""Content-addressed native attention reverse packages.

This module is the E2E-REAL-6C ownership boundary.  A traced forward attention
operation is converted once into the canonical tensor-valued reverse Graph
carrier, lowered through Schedule and Tile, and packaged by the target owner.
The runtime receives that package; it never receives the forward Graph and
cannot re-synthesize a backend program from it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
from .rocm_native import ROCMNativeProgram, package_scheduled_attention_backward as package_rocm
from .scheduled_attention_backward import (
    ScheduledAttentionBackwardArtifact,
    lower_scheduled_attention_backward,
)
from .x86_native import X86NativePackage, package_scheduled_attention_backward as package_x86


_ATTENTION_FORWARD_OPS = {
    "flash_attn",
    "gqa_attention",
    "mqa_attention",
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _array_dtype(value: Any) -> str:
    spelling = str(getattr(value, "dtype", ""))
    mapping = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
    }
    try:
        return mapping[spelling]
    except KeyError as exc:
        raise ValueError(f"native attention VJP does not support storage dtype {spelling!r}") from exc


def _tensor_type(value: Any, *, force_dtype: str | None = None):
    shape = tuple(int(extent) for extent in getattr(value, "shape", ()))
    if not shape:
        raise ValueError("native attention VJP requires ranked tensor operands")
    return tensor_ir_type(shape, force_dtype or _array_dtype(value))


@dataclass(frozen=True)
class NativeAttentionVJPPackage:
    """One traced parent plus its exact scheduled and physical reverse product."""

    source_graph_ir: str
    source_graph_ir_digest: str
    scheduled: ScheduledAttentionBackwardArtifact
    native: X86NativePackage | ROCMNativeProgram
    target: str
    operand_names: tuple[str, str, str]
    bias_name: str | None

    @property
    def schedule_artifact_hash(self) -> str:
        return self.scheduled.schedule_ir_digest

    @property
    def tile_program_digest(self) -> str:
        return self.scheduled.tile_digest

    @property
    def native_image_digest(self) -> str:
        return self.native.image.image_digest

    @property
    def artifact_hash(self) -> str:
        identity = ":".join(
            (
                "tessera.native_attention_vjp.v1",
                self.target,
                self.source_graph_ir_digest,
                self.scheduled.graph_digest,
                self.schedule_artifact_hash,
                self.tile_program_digest,
                self.native_image_digest,
            )
        )
        return _digest(identity)

    def validate(self) -> None:
        if _digest(self.source_graph_ir) != self.source_graph_ir_digest:
            raise ValueError("native attention VJP source Graph digest is stale")
        self.scheduled.validate()
        expected_target = "x86" if self.target == "x86" else "rocm"
        if self.scheduled.target != expected_target:
            raise ValueError("native attention VJP Schedule target is stale")
        if self.native.tile_ir != self.scheduled.tile_ir:
            raise ValueError("native attention VJP package did not consume the exact Tile artifact")
        descriptors = (
            (self.native.descriptor,)
            if isinstance(self.native, X86NativePackage)
            else self.native.descriptors
        )
        if not descriptors or any(
            descriptor.provenance.get("schedule_digest")
            not in {None, self.scheduled.schedule_digest}
            for descriptor in descriptors
        ):
            raise ValueError("native attention VJP descriptor has stale Schedule identity")


def _backward_module(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangent: Any,
) -> tuple[GraphIRModule, tuple[str, str, str], str | None]:
    bare = source.op_name.removeprefix("tessera.")
    if bare not in _ATTENTION_FORWARD_OPS:
        raise ValueError(f"native attention VJP does not own {source.op_name!r}")
    values = dict(zip(source_arg_names, ordered_inputs, strict=True))
    public_names = dict(zip(source_arg_names, arg_names, strict=True))
    source_names = tuple(str(name).removeprefix("%") for name in source.operands)
    if len(source_names) not in {3, 4} or any(name not in values for name in source_names):
        raise ValueError("native attention VJP requires Q/K/V and optional bias operands")
    graph_q, graph_key, graph_value = source_names[:3]
    graph_bias = source_names[3] if len(source_names) == 4 else None
    q_name, key_name, value_name = (
        public_names[name] for name in (graph_q, graph_key, graph_value)
    )
    bias_name = public_names[graph_bias] if graph_bias is not None else None
    q, key, value = (values[name] for name in (graph_q, graph_key, graph_value))
    do_type = _tensor_type(out_cotangent)
    q_type, key_type, value_type = (_tensor_type(item) for item in (q, key, value))
    if do_type.dtype != q_type.dtype:
        raise ValueError("native attention VJP cotangent storage must match Q storage")
    args = [
        IRArg("out_cotangent", do_type),
        IRArg(q_name, q_type),
        IRArg(key_name, key_type),
        IRArg(value_name, value_type),
    ]
    operand_names = ["%out_cotangent", f"%{q_name}", f"%{key_name}", f"%{value_name}"]
    operand_types = [str(do_type), str(q_type), str(key_type), str(value_type)]
    if bias_name is not None:
        assert graph_bias is not None
        bias_type = _tensor_type(values[graph_bias])
        if bias_type.dtype != "fp32":
            raise ValueError("native attention VJP bias must use fp32 storage")
        args.append(IRArg(bias_name, bias_type))
        operand_names.append(f"%{bias_name}")
        operand_types.append(str(bias_type))
    result_types = [
        _tensor_type(q, force_dtype="fp32"),
        _tensor_type(key, force_dtype="fp32"),
        _tensor_type(value, force_dtype="fp32"),
    ]
    kwargs = dict(source.kwargs)
    if target == "x86":
        # x86 owns a saved-LSE policy.  The scheduled carrier must state that
        # choice explicitly instead of inheriting a public-call default.
        kwargs["lse_checkpoint"] = "saved"
    module = GraphIRModule(
        functions=[
            GraphIRFunction(
                name="native_attention_vjp",
                args=args,
                result_types=result_types,
                body=[
                    IROp(
                        result="dq,dk,dv",
                        op_name="tessera.flash_attn_bwd",
                        operands=operand_names,
                        operand_types=operand_types,
                        result_type="(" + ", ".join(map(str, result_types)) + ")",
                        inferred_type=result_types[0],
                        inferred_types=tuple(result_types),
                        kwargs=kwargs,
                    )
                ],
                return_values=["%dq", "%dk", "%dv"],
            )
        ]
    )
    return module, (q_name, key_name, value_name), bias_name


def build_native_attention_vjp_package(
    *,
    source_graph_ir: str,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangent: Any,
) -> NativeAttentionVJPPackage:
    """Lower and package one bounded public attention reverse request."""

    if not source_graph_ir:
        raise ValueError("native attention VJP requires tracer-owned source Graph IR")
    if target not in {"x86", "rocm"}:
        raise ValueError(f"native attention VJP has no physical owner for {target!r}")
    module, operand_names, bias_name = _backward_module(
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        arg_names=arg_names,
        source_arg_names=source_arg_names,
        out_cotangent=out_cotangent,
    )
    scheduled_target = "x86" if target == "x86" else "rocm_gfx1151"
    scheduled = lower_scheduled_attention_backward(module, target=scheduled_target)
    if target == "x86":
        native: X86NativePackage | ROCMNativeProgram = package_x86(
            scheduled, pipeline_name="tessera-lower-to-x86"
        )
    else:
        native = package_rocm(
            scheduled, pipeline_name="tessera-lower-to-rocm"
        )
    package = NativeAttentionVJPPackage(
        source_graph_ir=source_graph_ir,
        source_graph_ir_digest=_digest(source_graph_ir),
        scheduled=scheduled,
        native=native,
        target=target,
        operand_names=operand_names,
        bias_name=bias_name,
    )
    package.validate()
    return package


def execute_native_attention_vjp_package(
    package: NativeAttentionVJPPackage,
    *,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangent: Any,
) -> tuple[tuple[Any, Any, Any], Mapping[str, Any] | None]:
    """Hand the already-built package to the runtime without Graph re-entry."""

    from tessera.runtime import _execute_native_attention_vjp_package

    package.validate()
    values: Mapping[str, Any] = dict(zip(arg_names, ordered_inputs, strict=True))
    return _execute_native_attention_vjp_package(
        package, values=values, out_cotangent=out_cotangent
    )


__all__ = [
    "NativeAttentionVJPPackage",
    "build_native_attention_vjp_package",
    "execute_native_attention_vjp_package",
]
