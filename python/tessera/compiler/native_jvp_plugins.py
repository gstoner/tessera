"""Family-owned planning for executable native forward products.

``JitFn`` owns Python call binding and package caching, but it must not become a
second lowering registry.  This module is the canonical family-plugin boundary:
each plugin consumes one verified Graph operation and produces an immutable
ordered child-package plan.  The C++ forward autodiff pass remains the semantic
authority for the paired JVP IR bound by the parent artifact.
"""

from __future__ import annotations

import hashlib
import re
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
    target_consumers={"x86": "x86.avx512_fft", "rocm": "rocm.gfx1151_fft",
                      "nvidia_sm120": "nvidia.sm120_cufft_workspace"},
)
def _plan_fft(*, source: Any, primal_inputs: Sequence[Any], wrt_indices: tuple[int, ...],
              target: str, execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if len(primal_inputs) != 1 or wrt_indices != (0,):
        raise ValueError("native FFT JVP requires one active input")
    from .scheduled_fft import lower_scheduled_fft

    value = primal_inputs[0]
    scheduled = lower_scheduled_fft(
        target=(target if target == "nvidia_sm120" else
                "x86" if target == "x86" else "rocm_gfx1151"),
        op_name=source.op_name,
        input_shape=tuple(int(dim) for dim in value.shape),
        axis=int(source.kwargs.get("axis", -1)),
        n=source.kwargs.get("n", source.kwargs.get("logical_length")),
        normalization=str(
            source.kwargs.get("normalization", source.kwargs.get("norm", "backward"))
        ),
        hermitian_weight=str(source.kwargs.get("hermitian_weight", "none")),
    )
    compiler_prefix = "nvidia" if target == "nvidia_sm120" else target
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{compiler_prefix}_fft_compiled",
        "arg_names": ["x"],
        "ops": [{
            "op_name": source.op_name,
            "result": source.result,
            "operands": ["x"],
            "kwargs": dict(source.kwargs),
        }],
        "scheduled_fft": scheduled.to_metadata(),
    }
    return NativeJVPFamilyPlan("spectral", (
        _step("primal", child, ("primal_0",)),
        _step("tangent", child, ("tangent_0",)),
    ))


@register_native_jvp_plugin(
    "dct", family="spectral_dct", schedule_consumer="schedule.spectral_program",
    tile_consumer="tile.spectral_program_kernel",
    target_consumers={"x86": "x86.avx512_dct", "rocm": "rocm.gfx1151_dct",
                      "nvidia_sm120": "nvidia.sm120_spectral_policy"},
)
def _plan_dct(*, source: Any, primal_inputs: Sequence[Any],
              wrt_indices: tuple[int, ...], target: str,
              execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if len(primal_inputs) != 1 or wrt_indices != (0,):
        raise ValueError("native DCT JVP requires one active input")
    from .scheduled_spectral import lower_scheduled_spectral

    value = primal_inputs[0]
    storage = {
        "float32": "f32", "float16": "f16", "bfloat16": "bf16",
    }.get(str(value.dtype))
    if storage is None:
        raise ValueError(f"native DCT JVP has unsupported storage {value.dtype}")
    scheduled = lower_scheduled_spectral(
        target=(target if target == "nvidia_sm120" else
                "x86" if target == "x86" else "rocm_gfx1151"),
        op_name="tessera.dct",
        input_shapes=(tuple(int(dim) for dim in value.shape),),
        axis=int(source.kwargs.get("axis", -1)),
        dct_type=int(source.kwargs.get("type", 2)),
        storage=storage,
        normalization=str(
            source.kwargs.get("normalization", source.kwargs.get("norm", "backward"))
        ),
    )
    compiler_prefix = "nvidia" if target == "nvidia_sm120" else target
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{compiler_prefix}_spectral_compiled",
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
    target_consumers={"x86": "x86.avx512_spectral", "rocm": "rocm.gfx1151_spectral",
                      "nvidia_sm120": "nvidia.sm120_spectral_policy"},
)
def _plan_compound_spectral(*, source: Any, primal_inputs: Sequence[Any],
                            wrt_indices: tuple[int, ...], target: str,
                            execution_mode: str, architecture: str = "gfx1151",
                            ir_contract: Mapping[str, Any] | None = None,
                            **_: Any) -> NativeJVPFamilyPlan:
    from .scheduled_spectral import lower_scheduled_spectral

    bare = source.op_name.removeprefix("tessera.")
    if len(primal_inputs) != 2:
        raise ValueError(f"native {bare} JVP requires two operands")
    if bare == "istft" and not set(wrt_indices).issubset({0, 1}):
        raise ValueError("native ISTFT JVP has only spectrum and window operands")
    storage: str | None
    if bare == "spectral_filter":
        if any(str(value.dtype) != "complex64" for value in primal_inputs):
            raise ValueError(
                "native spectral_filter JVP requires two complex64 operands"
            )
        # complex64 is the logical interleaved-fp32 policy. It is not a
        # separate physical storage dtype in the scheduled spectral carrier.
        storage = "f32"
        real_operand = primal_inputs[0]
    else:
        real_operand = primal_inputs[1] if bare == "istft" else primal_inputs[0]
        storage = {
            "float32": "f32", "float16": "f16", "bfloat16": "bf16",
        }.get(str(real_operand.dtype))
    if storage is None:
        raise ValueError(f"native {bare} JVP has unsupported real storage {real_operand.dtype}")
    oracle_arguments = _source_kwargs_spectral_arguments(
        source=source, primal_inputs=primal_inputs, storage=storage,
        target=target, architecture=architecture,
    )
    if bare == "istft":
        # ODS-WIRE-2: the ISTFT product's contract comes from the compiler's
        # paired JVP IR (the GraphToSchedule consumer of tessera.istft_jvp),
        # not from the source op's kwargs. The kwargs derivation is kept as a
        # declared #31 oracle and must lower to the identical scheduled
        # spectral program; a package whose derivations disagree is refused.
        if ir_contract is None:
            raise ValueError(
                "native ISTFT JVP requires the scheduled tessera.istft_jvp "
                "contract from the compiler's paired JVP IR"
            )
        scheduled = lower_scheduled_spectral(**istft_jvp_spectral_arguments(
            ir_contract, primal_inputs=primal_inputs, wrt_indices=wrt_indices,
            target=target, architecture=architecture,
        ))
        try:
            oracle = lower_scheduled_spectral(**oracle_arguments)
        except ValueError as exc:
            raise ValueError(
                "native ISTFT JVP: the declared source-kwargs oracle refused a "
                f"product the compiler scheduled ({exc}); refusing the package"
            ) from exc
        if oracle.to_metadata() != scheduled.to_metadata():
            raise ValueError(
                "native ISTFT JVP: the compiler's scheduled istft_jvp contract "
                "and the declared source-kwargs oracle lower to different "
                "spectral programs; refusing the package"
            )
    else:
        scheduled = lower_scheduled_spectral(**oracle_arguments)
    operands = ["primal_0", "primal_1"]
    names = operands + [f"tangent_{index}" for index in wrt_indices]
    compiler_prefix = "nvidia" if target == "nvidia_sm120" else target
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{compiler_prefix}_spectral_jvp_compiled",
        "autodiff_phase": "forward",
        "arg_names": names,
        "primal_names": operands,
        "wrt_indices": list(wrt_indices),
        "scheduled_spectral": scheduled.to_metadata(),
    }
    return NativeJVPFamilyPlan("spectral_compound", (
        _step("spectral_product", child, names, outputs=("primal", "tangent")),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# ODS-WIRE-2: the ISTFT product contract comes from compiler IR
# ─────────────────────────────────────────────────────────────────────────────

_SPECTRAL_JVP_SCHEMA = "tessera.spectral_jvp.v1"
_JVP_CONTRACT_RE = re.compile(
    r'schedule\.jvp_contract = "(?P<contract>[^"]*)"'
)
_JVP_HASH_RE = re.compile(r'schedule\.artifact_hash = "(?P<hash>[0-9a-f]{64})"')
_STORAGE_FROM_POLICY = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}
_STORAGE_FROM_DTYPE = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}


def _module_profile(target: str, architecture: str) -> tuple[str, str]:
    """The exact (tessera.target, tessera.arch) the Schedule consumer admits."""
    if target == "x86" and architecture == "zen5_avx512":
        return "x86", "zen5-avx512"
    if target == "rocm" and architecture in {"gfx1151", "gfx1201"}:
        return "rocm", architecture
    if target == "nvidia_sm120" and architecture == "sm120":
        return "nvidia_sm120", "sm120"
    raise ValueError(
        f"native ISTFT JVP has no exact Schedule profile for {target}/{architecture}"
    )


def istft_jvp_contract_from_paired_ir(
    paired_jvp_ir: str, *, target: str, architecture: str
) -> dict[str, Any]:
    """Schedule the compiler's paired JVP IR and read its ISTFT contract.

    Runs ``--tessera-graph-to-schedule`` -- whose ``tessera.istft_jvp`` arm is
    the production authority for the product's semantic contract -- over the
    paired IR that ``--tessera-autodiff-forward`` emitted, under the exact
    target profile. Returns the parsed contract. Nothing here re-derives a
    value: the only checks are that the contract hash is the SHA-256 of the
    contract text and that exactly one matching ``schedule.artifact`` exists.
    """
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt

    module_target, module_arch = _module_profile(target, architecture)
    header = re.search(r"^module( attributes \{)?", paired_jvp_ir, re.MULTILINE)
    if header is None:
        raise ValueError("native ISTFT JVP paired IR has no module")
    line_end = paired_jvp_ir.find("\n", header.start())
    header_line = paired_jvp_ir[header.start(): line_end if line_end >= 0 else None]
    if "tessera.target =" in header_line or "tessera.arch =" in header_line:
        raise ValueError("native ISTFT JVP paired IR already names a target profile")
    profile = f'tessera.target = "{module_target}", tessera.arch = "{module_arch}"'
    if header.group(1):
        scheduled_input = (
            paired_jvp_ir[: header.end()] + profile + ", " + paired_jvp_ir[header.end():]
        )
    else:
        scheduled_input = (
            paired_jvp_ir[: header.end()] + f" attributes {{{profile}}}"
            + paired_jvp_ir[header.end():]
        )
    tool = find_tessera_opt()
    if tool is None:
        raise ValueError("native ISTFT JVP requires the production tessera-opt")
    try:
        scheduled = run_tessera_opt(
            tool, scheduled_input, "--tessera-graph-to-schedule"
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    products = [line for line in scheduled.splitlines() if "tessera.istft_jvp " in line]
    if len(products) != 1:
        raise ValueError(
            f"native ISTFT JVP expects one scheduled tessera.istft_jvp; found {len(products)}"
        )
    contract_match = _JVP_CONTRACT_RE.search(products[0])
    hash_match = _JVP_HASH_RE.search(products[0])
    if contract_match is None or hash_match is None:
        raise ValueError("scheduled tessera.istft_jvp carries no hashed contract")
    contract_text = contract_match.group("contract")
    digest = hash_match.group("hash")
    if hashlib.sha256(contract_text.encode()).hexdigest() != digest:
        raise ValueError("scheduled tessera.istft_jvp contract does not match its hash")
    artifacts = [
        line for line in scheduled.splitlines()
        if line.lstrip().startswith(("schedule.artifact {", '"schedule.artifact"('))
        and f'hash = "{digest}"' in line
    ]
    if len(artifacts) != 1 or "family=spectral_jvp;kind=tessera.istft" not in artifacts[0]:
        raise ValueError(
            "scheduled tessera.istft_jvp requires exactly one matching schedule.artifact"
        )
    fields: dict[str, Any] = {}
    for item in contract_text.split(";"):
        key, separator, value = item.partition("=")
        if not separator or key in fields:
            raise ValueError(f"malformed spectral JVP contract field {item!r}")
        fields[key] = value
    if fields.get("schema") != _SPECTRAL_JVP_SCHEMA or fields.get("kind") != "tessera.istft":
        raise ValueError("scheduled tessera.istft_jvp contract schema mismatch")
    if (fields.get("target"), fields.get("arch")) != (module_target, module_arch):
        raise ValueError("scheduled tessera.istft_jvp contract names another profile")
    return {**fields, "artifact_hash": digest}


def _tensor_shape(type_text: str) -> tuple[int, ...]:
    match = re.fullmatch(r"tensor<((?:\d+x)*)(.+)>", type_text)
    if match is None:
        raise ValueError(f"spectral JVP contract has a non-static type {type_text!r}")
    return tuple(int(dim) for dim in match.group(1).split("x") if dim)


def istft_jvp_spectral_arguments(
    contract: Mapping[str, Any], *, primal_inputs: Sequence[Any],
    wrt_indices: tuple[int, ...], target: str, architecture: str,
) -> dict[str, Any]:
    """``lower_scheduled_spectral`` arguments from the compiler's contract.

    The contract is checked against the launch it will serve -- the primal
    shapes and dtypes and the requested tangent set -- and refused on any
    disagreement; it is never patched from the launch.
    """
    spectrum, window = primal_inputs
    if (_tensor_shape(str(contract["spectrum"])) != tuple(spectrum.shape)
            or _tensor_shape(str(contract["window"])) != tuple(window.shape)):
        raise ValueError("scheduled ISTFT JVP contract shapes disagree with the launch")
    storage = _STORAGE_FROM_POLICY.get(str(contract["numeric_storage"]))
    if storage is None or storage != _STORAGE_FROM_DTYPE.get(str(window.dtype)):
        raise ValueError("scheduled ISTFT JVP contract storage disagrees with the window")
    if str(contract["numeric_accum"]) != "fp32":
        raise ValueError("scheduled ISTFT JVP contract requires fp32 accumulation")
    active = tuple(int(index) for index in str(contract["active_tangents"]).split(","))
    if active != tuple(sorted(wrt_indices)):
        raise ValueError(
            f"scheduled ISTFT JVP activity {active} disagrees with wrt {wrt_indices}"
        )
    if str(contract["pad_mode"]) != "constant":
        raise ValueError("scheduled ISTFT JVP contract names a non-constant pad mode")
    return {
        "target": (target if target == "nvidia_sm120" else
                   "x86" if target == "x86" else f"rocm_{architecture}"),
        "op_name": "tessera.istft",
        "input_shapes": (tuple(spectrum.shape), tuple(window.shape)),
        "axis": int(contract["axis"]),
        "hop": int(contract["hop"]),
        "normalization": str(contract["normalization"]),
        "storage": storage,
        "center": str(contract["center"]) == "1",
        "pad_mode": "constant",
        "output_length": int(contract["output_length"]),
        "n_fft": int(contract["logical_length"]),
        "onesided": str(contract["onesided"]) == "1",
    }


def _source_kwargs_spectral_arguments(
    *, source: Any, primal_inputs: Sequence[Any], storage: str, target: str,
    architecture: str,
) -> dict[str, Any]:
    """Derive ``lower_scheduled_spectral`` arguments from the source op's kwargs.

    For ``spectral_filter``/``spectral_conv``/``stft`` this is still the
    production derivation. For ``istft`` it is a **declared Decision #31
    oracle** since ODS-WIRE-2: the production contract is the compiler's
    scheduled ``tessera.istft_jvp`` (:func:`istft_jvp_contract_from_paired_ir`),
    every package re-derives this and refuses a disagreement, and
    ``tests/unit/test_istft_jvp_ir_contract.py`` is its differential test.
    Retire it only after that test has covered the configurations it covers.
    """
    bare = source.op_name.removeprefix("tessera.")
    return {
        # The exact chip's composite profile (gfx1151 or gfx1201).
        "target": (target if target == "nvidia_sm120" else
                   "x86" if target == "x86" else f"rocm_{architecture}"),
        "op_name": source.op_name,
        "input_shapes": tuple(
            tuple(int(dim) for dim in value.shape) for value in primal_inputs
        ),
        "axis": int(source.kwargs.get("axis", -1)),
        "hop": source.kwargs.get("hop", source.kwargs.get("hop_length")),
        "normalization": str(source.kwargs.get(
            "normalization", source.kwargs.get("norm", "backward")
        )),
        "storage": storage,
        "center": bool(source.kwargs.get("center", False)),
        "pad_mode": str(source.kwargs.get("pad_mode", "constant")),
        "output_length": source.kwargs.get(
            "length", source.kwargs.get("output_length")
        ),
        "n_fft": (
            source.kwargs.get("n_fft", source.kwargs.get("logical_length"))
            if bare in {"stft", "istft"} else None
        ),
        "onesided": bool(source.kwargs.get("onesided", True)),
    }


def plan_native_jvp_family(
    *, source: Any, primal_inputs: Sequence[Any], wrt_indices: tuple[int, ...],
    target: str, architecture: str, execution_mode: str,
    ir_contract: Mapping[str, Any] | None = None,
) -> NativeJVPFamilyPlan:
    """Dispatch to the sole registered owner for ``source`` or fail closed.

    Which (target, architecture, family) triples may run is enforced by the
    parent artifact (``native_jvp.architecture_admits``); planners receive the
    architecture so a per-chip package is lowered for the chip it runs on."""
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
        architecture=architecture,
        ir_contract=ir_contract,
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

    ir_contract = (
        istft_jvp_contract_from_paired_ir(
            paired_jvp_ir, target=target, architecture=architecture
        )
        if source.op_name == "tessera.istft" else None
    )
    plan = plan_native_jvp_family(
        source=source, primal_inputs=primal_inputs, wrt_indices=wrt_indices,
        target=target, architecture=architecture, execution_mode=execution_mode,
        ir_contract=ir_contract,
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
    # ROCm consumers are declared for the primary chip; a package admitted on
    # another chip names that chip, never gfx1151.
    target_consumer = plan.declaration.target_consumers[target]
    if target == "rocm" and architecture != "gfx1151":
        target_consumer = target_consumer.replace("gfx1151", architecture)
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
    schedule_program: dict[str, Any] = {
        "schema": "tessera.native_jvp_schedule.v1",
        "family": plan.family,
        "consumer": plan.declaration.schedule_consumer,
        "actions": schedule_actions,
    }
    if ir_contract is not None:
        # The Graph->Schedule artifact the compiler minted for the paired
        # tessera.istft_jvp; the package is bound to it, not to the kwargs.
        schedule_program["graph_schedule_artifact"] = str(ir_contract["artifact_hash"])
        schedule_program["graph_schedule_consumer"] = (
            "tessera-graph-to-schedule:tessera.istft_jvp"
        )
    tile_program = {
        "schema": "tessera.native_jvp_tile.v1",
        "family": plan.family,
        "consumer": plan.declaration.tile_consumer,
        "actions": [
            {
                "id": str(step["id"]),
                "child_digest": str(step["child_digest"]),
                "target_consumer": target_consumer,
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
            "target": target_consumer,
            "migration_state": plan.declaration.migration_state,
        } if plan.declaration is not None else None,
        schedule_program=schedule_program,
        tile_program=tile_program,
    )
    return plan, artifact


__all__ = [
    "istft_jvp_contract_from_paired_ir",
    "istft_jvp_spectral_arguments",
    "NativeJVPFamilyPlan",
    "NativeJVPPluginDeclaration",
    "build_native_jvp_family_artifact",
    "native_jvp_plugin_declarations",
    "native_jvp_plugin_owners",
    "plan_native_jvp_family",
    "register_native_jvp_plugin",
]
