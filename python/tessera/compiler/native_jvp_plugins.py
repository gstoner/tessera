"""Family-owned planning for executable native forward products.

``JitFn`` owns Python call binding and package caching, but it must not become a
second lowering registry.  This module is the canonical family-plugin boundary:
each plugin consumes one verified Graph operation and produces an immutable
ordered child-package plan.  The C++ forward autodiff pass remains the semantic
authority for the paired JVP IR bound by the parent artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from .native_jvp import child_digest


@dataclass(frozen=True)
class NativeJVPFamilyPlan:
    family: str
    steps: tuple[Mapping[str, Any], ...]
    declaration: "NativeJVPPluginDeclaration | None" = None


@dataclass(frozen=True)
class NativeJVPPluginDeclaration:
    """Typed ownership declaration across the complete compiler spine."""

    family: str
    graph_consumers: tuple[str, ...]
    schedule_consumer: str | None
    tile_consumer: str | None
    target_consumers: Mapping[str, str]
    migration_state: str = "canonical"

    def validate(self) -> None:
        if not self.family or not self.graph_consumers:
            raise ValueError("native JVP declaration requires family and Graph consumers")
        if self.migration_state not in {"canonical", "canonical_composite", "compatibility"}:
            raise ValueError("native JVP declaration has an invalid migration state")
        if self.migration_state in {"canonical", "canonical_composite"} and (
            self.schedule_consumer is None
            or not self.schedule_consumer.startswith("schedule.")
            or self.tile_consumer is None
            or not self.tile_consumer.startswith("tile.")
        ):
            raise ValueError("canonical native JVP plugins require Schedule and Tile consumers")
        if not {"x86", "rocm"}.issubset(self.target_consumers) or any(
            not value for value in self.target_consumers.values()
        ):
            raise ValueError("native JVP declaration requires x86 and ROCm Target consumers")


Planner = Callable[..., NativeJVPFamilyPlan]
_PLUGINS: dict[str, tuple[NativeJVPPluginDeclaration, Planner]] = {}


def register_native_jvp_plugin(
    *op_names: str,
    family: str,
    schedule_consumer: str | None,
    tile_consumer: str | None,
    target_consumers: Mapping[str, str],
    migration_state: str = "canonical",
) -> Callable[[Planner], Planner]:
    """Register one explicit Graph-op consumer; duplicate ownership is invalid."""
    declaration = NativeJVPPluginDeclaration(
        family=family,
        graph_consumers=tuple(f"tessera.{name}" for name in op_names),
        schedule_consumer=schedule_consumer,
        tile_consumer=tile_consumer,
        target_consumers=dict(target_consumers),
        migration_state=migration_state,
    )
    declaration.validate()

    def decorate(planner: Planner) -> Planner:
        for name in op_names:
            if name in _PLUGINS:
                raise RuntimeError(f"native JVP plugin already owns {name!r}")
            _PLUGINS[name] = (declaration, planner)
        return planner
    return decorate


def _step(
    step_id: str,
    child: Mapping[str, Any],
    inputs: Sequence[str],
    *,
    output_index: int = -1,
    outputs: Sequence[str] | None = None,
    depends_on: Sequence[str] = (),
) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "id": step_id,
        "child_digest": child_digest(child),
        "child_metadata": dict(child),
        "inputs": list(inputs),
        "output_index": output_index,
        "depends_on": list(depends_on),
    }
    if outputs is not None:
        result["outputs"] = list(outputs)
    return result


def _scheduled_reduce_step(
    step_id: str,
    source: Any,
    value: Any,
    binding: str,
    *,
    target: str,
) -> Mapping[str, Any]:
    """Build one real Schedule→Tile→Target reduction descriptor child."""
    import numpy as np
    from tessera.runtime import RuntimeArtifact
    from .graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
    from .scheduled_kernel import lower_scheduled_kernel

    input_shape = tuple(int(dim) for dim in value.shape)
    axis = int(source.kwargs.get("axis", -1))
    kind = str(source.kwargs.get("kind", source.op_name.removeprefix("tessera.")))
    keepdims = bool(source.kwargs.get("keepdims", False))
    probe = np.empty(input_shape, dtype=np.float32)
    output_shape = tuple(
        int(dim) for dim in (
            np.sum(probe, axis=axis, keepdims=keepdims)
            if kind == "sum"
            else np.mean(probe, axis=axis, keepdims=keepdims)
        ).shape
    )
    input_type = tensor_ir_type(tuple(str(dim) for dim in input_shape), "fp32")
    output_type = tensor_ir_type(tuple(str(dim) for dim in output_shape), "fp32")
    module = GraphIRModule(functions=[GraphIRFunction(
        name=f"native_jvp_{kind}_{step_id}",
        args=[IRArg("x", input_type)],
        result_types=[output_type],
        body=[IROp(
            result="o", op_name=f"tessera.{kind}", operands=["%x"],
            operand_types=[str(input_type)], result_type=str(output_type),
            inferred_type=output_type,
            kwargs={"axis": axis, "keepdims": keepdims},
        )],
        return_values=["%o"],
    )])
    compiler_target = "x86" if target == "x86" else "rocm_gfx1151"
    scheduled = lower_scheduled_kernel(module, target=compiler_target)
    package: Any
    if target == "x86":
        from .x86_native import package_scheduled_kernel as package_x86_scheduled_kernel
        package = package_x86_scheduled_kernel(
            scheduled, pipeline_name="tessera-lower-to-x86"
        )
    else:
        from .rocm_native import package_scheduled_kernel as package_rocm_scheduled_kernel
        package = package_rocm_scheduled_kernel(
            scheduled, pipeline_name="tessera-lower-to-rocm"
        )
    runtime = RuntimeArtifact(
        schedule_ir=scheduled.schedule_ir,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={"compiler_path": f"{target}_scheduled_reduce_jvp_child"},
        native_image=package.image,
        launch_descriptor=package.descriptor,
    ).to_dict()
    scalar_values = (
        {"Outer": scheduled.outer, "AxisExtent": scheduled.axis_extent,
         "Inner": scheduled.inner}
    )
    return {
        "id": step_id,
        "child_digest": child_digest(runtime),
        "child_artifact": runtime,
        "inputs": [binding],
        "depends_on": [],
        "descriptor_invocation": {
            "input": scheduled.input_name,
            "output": scheduled.output_name,
            "output_shape": list(output_shape),
            "output_dtype": "float32",
            "scalars": scalar_values,
        },
    }


def _execution(target: str, execution_mode: str) -> dict[str, Any]:
    return {
        "target": target,
        "executable": True,
        "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
        "execution_mode": execution_mode,
    }


@register_native_jvp_plugin(
    "dropout", family="philox_dropout",
    schedule_consumer="schedule.rng_program", tile_consumer="tile.rng_kernel",
    target_consumers={"x86": "x86.avx512_philox_dropout",
                      "rocm": "rocm.gfx1151_philox_dropout",
                      "nvidia_sm120": "nvidia.sm120_philox_dropout"},
)
def _plan_philox_dropout(*, source: Any, primal_inputs: Sequence[Any],
                         wrt_indices: tuple[int, ...], target: str,
                         execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if target != "nvidia_sm120":
        raise ValueError("compiler-JVP Philox packaging currently requires sm120")
    if len(primal_inputs) != 1 or wrt_indices != (0,):
        raise ValueError("Philox dropout JVP requires one active data input")
    kwargs = dict(source.kwargs)
    if bool(kwargs.get("training", True)) and float(kwargs.get("p", 0.5)) > 0.0:
        if kwargs.get("seed") is None and kwargs.get("key") is None:
            raise ValueError("Philox dropout JVP requires an explicit key/seed")
    child = {
        **_execution(target, execution_mode),
        "compiler_path": "nvidia_rng_compiled", "arg_names": ["x"],
        "ops": [{"op_name": "tessera.dropout", "result": "o",
                 "operands": ["x"], "kwargs": kwargs}],
    }
    # Both children carry the identical key/counter attributes. The compiler's
    # DropoutOp tangent rule clones the seeded op, so the mask is replayed—not
    # resampled—for the tangent input.
    return NativeJVPFamilyPlan("philox_dropout", (
        _step("primal", child, ("primal_0",)),
        _step("tangent", child, ("tangent_0",)),
    ))


@register_native_jvp_plugin(
    "reduce", "sum", "mean", family="reduce",
    schedule_consumer="schedule.reduce", tile_consumer="tile.reduce_kernel",
    target_consumers={"x86": "x86.avx512_reduction", "rocm": "rocm.gfx1151_reduction"},
)
def _plan_reduce(*, source: Any, primal_inputs: Sequence[Any], wrt_indices: tuple[int, ...],
                 target: str, execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if len(primal_inputs) != 1 or wrt_indices != (0,):
        raise ValueError("native reduction JVP requires one active input")
    bare = source.op_name.removeprefix("tessera.")
    kind = str(source.kwargs.get("kind", bare))
    if kind not in {"sum", "mean"}:
        raise ValueError(f"native reduction JVP requires sum or mean; got {kind!r}")
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{target}_reduce_compiled",
        "arg_names": ["x"],
        "ops": [{
            "op_name": f"tessera.{kind}",
            "result": source.result,
            "operands": ["x"],
            "kwargs": {key: value for key, value in source.kwargs.items() if key != "kind"},
        }],
    }
    return NativeJVPFamilyPlan("reduce", (
        _step("primal", child, ("primal_0",)),
        _step("tangent", child, ("tangent_0",)),
    ))


@register_native_jvp_plugin(
    "fft", "ifft", "rfft", "irfft", family="spectral",
    schedule_consumer="schedule.fft", tile_consumer="tile.fft_kernel",
    target_consumers={"x86": "x86.avx512_fft", "rocm": "rocm.gfx1151_fft"},
)
def _plan_fft(*, source: Any, primal_inputs: Sequence[Any], wrt_indices: tuple[int, ...],
              target: str, execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if len(primal_inputs) != 1 or wrt_indices != (0,):
        raise ValueError("native FFT JVP requires one active input")
    from .scheduled_fft import lower_scheduled_fft

    value = primal_inputs[0]
    scheduled = lower_scheduled_fft(
        target="x86" if target == "x86" else "rocm_gfx1151",
        op_name=source.op_name,
        input_shape=tuple(int(dim) for dim in value.shape),
        axis=int(source.kwargs.get("axis", -1)),
        n=source.kwargs.get("n", source.kwargs.get("logical_length")),
        normalization=str(source.kwargs.get("normalization", "backward")),
        hermitian_weight=str(source.kwargs.get("hermitian_weight", "none")),
    )
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{target}_fft_compiled",
        "arg_names": ["x"],
        "scheduled_fft": scheduled.to_metadata(),
    }
    return NativeJVPFamilyPlan("spectral", (
        _step("primal", child, ("primal_0",)),
        _step("tangent", child, ("tangent_0",)),
    ))


@register_native_jvp_plugin(
    "dct", family="spectral_dct", schedule_consumer="schedule.spectral_program",
    tile_consumer="tile.spectral_program_kernel",
    target_consumers={"x86": "x86.avx512_dct", "rocm": "rocm.gfx1151_dct"},
)
def _plan_dct(*, source: Any, primal_inputs: Sequence[Any],
              wrt_indices: tuple[int, ...], target: str,
              execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if len(primal_inputs) != 1 or wrt_indices != (0,):
        raise ValueError("native DCT JVP requires one active input")
    from .scheduled_spectral import lower_scheduled_spectral

    value = primal_inputs[0]
    scheduled = lower_scheduled_spectral(
        target="x86" if target == "x86" else "rocm_gfx1151",
        op_name="tessera.dct",
        input_shapes=(tuple(int(dim) for dim in value.shape),),
        axis=int(source.kwargs.get("axis", -1)),
        dct_type=int(source.kwargs.get("type", 2)),
        normalization=str(
            source.kwargs.get("normalization", source.kwargs.get("norm", "backward"))
        ),
    )
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{target}_spectral_compiled",
        "arg_names": ["x"],
        "scheduled_spectral": scheduled.to_metadata(),
    }
    return NativeJVPFamilyPlan("spectral_dct", (
        _step("primal", child, ("primal_0",)),
        _step("tangent", child, ("tangent_0",)),
    ))


@register_native_jvp_plugin(
    "rmsnorm", "rmsnorm_safe", "layer_norm", family="normalization",
    schedule_consumer="schedule.native_jvp_program",
    tile_consumer="tile.native_jvp_program", migration_state="canonical_composite",
    target_consumers={"x86": "x86.avx512_normalization", "rocm": "rocm.gfx1151_normalization"},
)
def _plan_normalization(*, source: Any, primal_inputs: Sequence[Any],
                        wrt_indices: tuple[int, ...], target: str,
                        execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if not wrt_indices or 0 not in wrt_indices:
        raise ValueError("normalization JVP requires an active data input")
    operands = [f"primal_{index}" for index in range(len(primal_inputs))]
    names = operands + [f"tangent_{index}" for index in wrt_indices]
    kind = source.op_name.removeprefix("tessera.")
    actions: list[dict[str, Any]] = [
        {"id": "primal_normalization", "consumer": "tile.norm_kernel", "depends_on": []},
        {"id": "tangent_projection", "consumer": "target.norm_backward", "depends_on": []},
    ]
    if len(primal_inputs) >= 2:
        actions.extend([
            {"id": "normalized_operand", "consumer": "tile.norm_kernel", "depends_on": []},
            {"id": "affine_projection", "consumer": "target.binary_mul",
             "depends_on": ["tangent_projection"]},
        ])
        if 1 in wrt_indices:
            actions.append({
                "id": "affine_tangent", "consumer": "target.binary_mul_add",
                "depends_on": ["normalized_operand", "affine_projection"],
            })
    if len(primal_inputs) >= 3 and 2 in wrt_indices:
        actions.append({
            "id": "bias_tangent", "consumer": "target.binary_add",
            "depends_on": [actions[-1]["id"]],
        })
    schedule_body = {
        "schema": "tessera.normalization_jvp_schedule.v1",
        "kind": kind,
        "input_shapes": [list(value.shape) for value in primal_inputs],
        "storage_dtypes": [str(value.dtype) for value in primal_inputs],
        "epsilon": float(source.kwargs.get("eps", 1.0e-5)),
        "primal_names": operands,
        "wrt_indices": list(wrt_indices),
        "actions": actions,
    }
    scheduled_contract = {
        **schedule_body,
        "schedule_digest": child_digest(schedule_body),
    }
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{target}_norm_jvp_compiled",
        "autodiff_phase": "forward",
        "wrt_indices": list(wrt_indices),
        "arg_names": names,
        "scheduled_normalization_jvp": scheduled_contract,
    }
    return NativeJVPFamilyPlan("normalization", (
        _step("normalization_product", child, names, outputs=("primal", "tangent")),
    ))


@register_native_jvp_plugin(
    "spectral_filter", "spectral_conv", "stft", "istft",
    family="spectral_compound",
    schedule_consumer="schedule.spectral_program",
    tile_consumer="tile.spectral_program_kernel",
    target_consumers={"x86": "x86.avx512_spectral", "rocm": "rocm.gfx1151_spectral"},
)
def _plan_compound_spectral(*, source: Any, primal_inputs: Sequence[Any],
                            wrt_indices: tuple[int, ...], target: str,
                            execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    from .scheduled_spectral import lower_scheduled_spectral

    bare = source.op_name.removeprefix("tessera.")
    if len(primal_inputs) != 2:
        raise ValueError(f"native {bare} JVP requires two operands")
    if bare == "istft" and not set(wrt_indices).issubset({0, 1}):
        raise ValueError("native ISTFT JVP has only spectrum and window operands")
    scheduled = lower_scheduled_spectral(
        target="x86" if target == "x86" else "rocm_gfx1151",
        op_name=source.op_name,
        input_shapes=tuple(tuple(int(dim) for dim in value.shape) for value in primal_inputs),
        axis=int(source.kwargs.get("axis", -1)),
        hop=source.kwargs.get("hop"),
        normalization=str(source.kwargs.get("normalization", "backward")),
    )
    operands = ["primal_0", "primal_1"]
    names = operands + [f"tangent_{index}" for index in wrt_indices]
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{target}_spectral_jvp_compiled",
        "autodiff_phase": "forward",
        "arg_names": names,
        "primal_names": operands,
        "wrt_indices": list(wrt_indices),
        "scheduled_spectral": scheduled.to_metadata(),
    }
    return NativeJVPFamilyPlan("spectral_compound", (
        _step("spectral_product", child, names, outputs=("primal", "tangent")),
    ))


def plan_native_jvp_family(
    *, source: Any, primal_inputs: Sequence[Any], wrt_indices: tuple[int, ...],
    target: str, architecture: str, execution_mode: str,
) -> NativeJVPFamilyPlan:
    """Dispatch to the sole registered owner for ``source`` or fail closed."""
    del architecture  # exact architecture is enforced by the parent artifact
    bare = source.op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None:
        raise ValueError(
            f"no native {target} JVP family plugin exists for {bare!r}; "
            "the compiler IR transform remains available"
        )
    declaration, planner = entry
    if target not in declaration.target_consumers:
        raise ValueError(
            f"native {target} JVP has no Target consumer for {bare!r}"
        )
    plan = planner(
        source=source,
        primal_inputs=primal_inputs,
        wrt_indices=wrt_indices,
        target=target,
        execution_mode=execution_mode,
    )
    if plan.family != declaration.family:
        raise ValueError(
            f"native JVP plugin declared {declaration.family!r} but planned {plan.family!r}"
        )
    return replace(plan, declaration=declaration)


def native_jvp_plugin_owners() -> Mapping[str, str]:
    """Stable inspection surface used by registry-totality tests and docs."""
    return {name: planner.__name__ for name, (_, planner) in sorted(_PLUGINS.items())}


def native_jvp_plugin_declarations() -> Mapping[str, NativeJVPPluginDeclaration]:
    """Return the explicit Graph/Schedule/Tile/Target ownership registry."""
    return {name: declaration for name, (declaration, _) in sorted(_PLUGINS.items())}


def build_native_jvp_family_artifact(
    *, source: Any, primal_inputs: Sequence[Any], wrt_indices: tuple[int, ...],
    target: str, architecture: str, execution_mode: str, source_graph_ir: str,
    paired_jvp_ir: str, arg_names: Sequence[str],
) -> tuple[NativeJVPFamilyPlan, Any]:
    """Plan and construct a native package entirely inside the family boundary."""
    from .native_jvp import build_native_jvp_artifact

    plan = plan_native_jvp_family(
        source=source, primal_inputs=primal_inputs, wrt_indices=wrt_indices,
        target=target, architecture=architecture, execution_mode=execution_mode,
    )
    if plan.family == "reduce":
        plan = replace(plan, steps=(
            _scheduled_reduce_step(
                "primal", source, primal_inputs[0], "primal_0", target=target
            ),
            _scheduled_reduce_step(
                "tangent", source, primal_inputs[0], "tangent_0", target=target
            ),
        ))
    if plan.declaration is None:
        raise ValueError("native JVP family plan lacks a consumer declaration")
    if plan.family == "normalization":
        child_metadata = plan.steps[0].get("child_metadata")
        contract = (
            child_metadata.get("scheduled_normalization_jvp")
            if isinstance(child_metadata, Mapping) else None
        )
        if not isinstance(contract, Mapping):
            raise ValueError("normalization JVP lacks its Schedule program")
        schedule_actions = list(contract["actions"])
    else:
        schedule_actions = [
            {
                "id": str(step["id"]),
                "consumer": plan.declaration.schedule_consumer,
                "depends_on": list(step.get("depends_on", [])),
            }
            for step in plan.steps
        ]
    schedule_program = {
        "schema": "tessera.native_jvp_schedule.v1",
        "family": plan.family,
        "consumer": plan.declaration.schedule_consumer,
        "actions": schedule_actions,
    }
    tile_program = {
        "schema": "tessera.native_jvp_tile.v1",
        "family": plan.family,
        "consumer": plan.declaration.tile_consumer,
        "actions": [
            {
                "id": str(step["id"]),
                "child_digest": str(step["child_digest"]),
                "target_consumer": plan.declaration.target_consumers[target],
            }
            for step in plan.steps
        ],
    }
    artifact = build_native_jvp_artifact(
        target=target, architecture=architecture, family=plan.family,
        source_graph_ir=source_graph_ir, paired_jvp_ir=paired_jvp_ir,
        wrt_indices=wrt_indices, arg_names=arg_names, steps=plan.steps,
        consumer_declaration={
            "graph": list(plan.declaration.graph_consumers),
            "schedule": plan.declaration.schedule_consumer,
            "tile": plan.declaration.tile_consumer,
            "target": plan.declaration.target_consumers[target],
            "migration_state": plan.declaration.migration_state,
        } if plan.declaration is not None else None,
        schedule_program=schedule_program,
        tile_program=tile_program,
    )
    return plan, artifact


__all__ = [
    "NativeJVPFamilyPlan",
    "NativeJVPPluginDeclaration",
    "build_native_jvp_family_artifact",
    "native_jvp_plugin_declarations",
    "native_jvp_plugin_owners",
    "plan_native_jvp_family",
    "register_native_jvp_plugin",
]
