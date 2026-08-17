"""Family-owned execution for native reverse products.

``JitFn`` owns call binding and records the selected execution result.  It must
not own another operation-family dispatch table or construct backend packages.
This module is the migration boundary for reverse products: every registered
family declares the Graph, Schedule, Tile, and Target consumers that own its
physical execution.

Normalization, bounded compound spectral products, canonical rank-4 attention,
Lion, factored/full Adafactor, and causal sequence-mixer backward are migrated.
SGD, Momentum/Nesterov, and the gfx1151 Adam/AdamW consumers also use that
boundary; unsupported target pairs fail closed before package construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .._jit_boundary import TesseraJitError


@dataclass(frozen=True)
class NativeVJPPluginDeclaration:
    """Typed ownership declaration across the complete compiler spine."""

    family: str
    graph_consumers: tuple[str, ...]
    schedule_consumer: str
    tile_consumer: str
    target_consumers: Mapping[str, str]
    migration_state: str = "canonical_composite"
    differential_policy: str = "pure_only"

    def validate(self) -> None:
        if not self.family or not self.graph_consumers:
            raise ValueError("native VJP declaration requires family and Graph consumers")
        if not self.schedule_consumer.startswith("schedule."):
            raise ValueError("native VJP declaration requires a Schedule consumer")
        if not self.tile_consumer.startswith("tile."):
            raise ValueError("native VJP declaration requires a Tile consumer")
        if self.migration_state not in {"canonical", "canonical_composite"}:
            raise ValueError("native VJP declaration has an invalid migration state")
        if self.differential_policy not in {
            "pure_only",
            "zero_dropout_attention",
            "non_reexecuting_state_lineage",
        }:
            raise ValueError("native VJP declaration has an invalid differential policy")
        if not self.target_consumers or any(not value for value in self.target_consumers.values()):
            raise ValueError("native VJP declaration requires concrete Target consumers")


@dataclass(frozen=True)
class NativeVJPResult:
    gradients: tuple[Any, ...]
    execution: Mapping[str, Any]


Executor = Callable[..., NativeVJPResult]
_PLUGINS: dict[str, tuple[NativeVJPPluginDeclaration, Executor]] = {}


def register_native_vjp_plugin(
    *op_names: str,
    family: str,
    schedule_consumer: str,
    tile_consumer: str,
    target_consumers: Mapping[str, str],
    differential_policy: str = "pure_only",
) -> Callable[[Executor], Executor]:
    """Register one explicit Graph-op owner; duplicate ownership is invalid."""
    declaration = NativeVJPPluginDeclaration(
        family=family,
        graph_consumers=tuple(f"tessera.{name}" for name in op_names),
        schedule_consumer=schedule_consumer,
        tile_consumer=tile_consumer,
        target_consumers=dict(target_consumers),
        differential_policy=differential_policy,
    )
    declaration.validate()

    def decorate(executor: Executor) -> Executor:
        for name in op_names:
            if name in _PLUGINS:
                raise RuntimeError(f"native VJP plugin already owns {name!r}")
            _PLUGINS[name] = (declaration, executor)
        return executor

    return decorate


@register_native_vjp_plugin(
    "rmsnorm",
    "rmsnorm_safe",
    "layer_norm",
    family="normalization",
    schedule_consumer="schedule.native_vjp_program",
    tile_consumer="tile.native_vjp_program",
    target_consumers={
        "x86": "x86.avx512_normalization",
        "rocm": "rocm.gfx1151_normalization",
        "nvidia_sm120": "nvidia.sm120_normalization",
        "apple_gpu": "apple.metal_normalization",
    },
)
def _execute_normalization(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Package and launch one registered normalization VJP."""
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    inputs = [np.ascontiguousarray(np.asarray(value)) for value in ordered_inputs]
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )


    if len(cotangents) != 1:
        raise TesseraJitError("normalization backward requires one output cotangent")
    dy = np.ascontiguousarray(np.asarray(cotangents[0]))
    bare = source.op_name.removeprefix("tessera.")
    family = "layer_norm" if bare == "layer_norm" else "rmsnorm"
    operand_names = list(arg_names)
    gradient_names = [f"d_{name}" for name in operand_names]
    target_identity = {
        "x86": ("cpu_avx512", "native_cpu", "x86_avx512"),
        "rocm": ("hip_runtime", "native_gpu", "rocm_gfx1151"),
        "nvidia_sm120": ("cuda_driver", "native_gpu", "nvidia_sm120"),
        "apple_gpu": ("metal_runtime", "native_gpu", "apple7"),
    }
    try:
        execution_mode, execution_kind, evidence_target = target_identity[target]
    except KeyError as exc:
        raise TesseraJitError(
            f"native normalization VJP has no Target consumer for {target!r}"
        ) from exc
    path = (
        "nvidia_norm_bwd_compiled"
        if target == "nvidia_sm120"
        else f"{target}_{family}_bwd_compiled"
    )
    artifact = RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": path,
        "executable": True,
        "execution_kind": execution_kind,
        "execution_mode": execution_mode,
        "autodiff_phase": "backward",
        "native_vjp_family": declaration.family,
        "native_vjp_schedule_consumer": declaration.schedule_consumer,
        "native_vjp_tile_consumer": declaration.tile_consumer,
        "native_vjp_target_consumer": declaration.target_consumers[target],
        "out_cotangent": "dy",
        "arg_names": operand_names + ["dy"],
        "output_names": gradient_names,
        "ops": [{
            "op_name": source.op_name,
            "result": source.result,
            "operands": operand_names,
            "kwargs": dict(source.kwargs),
        }],
    })
    result = launch(artifact, tuple([*inputs, dy]))
    if not result.get("ok") or result.get("execution_mode") != execution_mode:
        raise TesseraJitError(
            f"verified {target} normalization backward launch failed: "
            + str(result.get("reason"))
        )
    output = result["output"]
    all_gradients = (
        tuple(output) if isinstance(output, (tuple, list)) else (output,)
    )
    by_name = dict(zip(operand_names, all_gradients))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "normalization VJP requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": path,
            "execution_kind": execution_kind,
            "execution_mode": execution_mode,
            "evidence_target": evidence_target,
            "implementation": "family_plugin",
            "residual_policy": "recompute_all",
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
        },
    )


@register_native_vjp_plugin(
    "spectral_filter",
    "spectral_conv",
    family="spectral_backward",
    schedule_consumer="schedule.spectral_backward",
    tile_consumer="tile.spectral_backward_kernel",
    target_consumers={
        "x86": "x86.avx512_spectral_backward",
        "rocm": "rocm.gfx1151_spectral_backward",
    },
)
def _execute_compound_spectral(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Build and execute one traced compound-spectral reverse package."""
    from tessera.runtime import RuntimeArtifact, launch

    from .native_spectral_vjp import (
        build_native_spectral_vjp_package,
        compile_rocm_native_spectral_vjp,
    )

    if not source_graph_ir:
        raise TesseraJitError(
            "compound spectral VJP requires tracer-owned source Graph IR"
        )
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    if len(cotangents) != 1:
        raise TesseraJitError("compound spectral VJP requires one output cotangent")
    try:
        package = build_native_spectral_vjp_package(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangent=cotangents[0],
        )
        if target == "rocm":
            package = compile_rocm_native_spectral_vjp(package)
    except (RuntimeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    result = launch(
        RuntimeArtifact(metadata=package.runtime_metadata()),
        tuple([cotangents[0], *ordered_inputs]),
    )
    expected_mode = "cpu_avx512" if target == "x86" else "hip_runtime"
    if not result.get("ok") or result.get("execution_mode") != expected_mode:
        raise TesseraJitError(
            f"verified {target} compound spectral backward launch failed: "
            + str(result.get("reason"))
        )
    outputs = result.get("output")
    gradients = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
    by_name = dict(zip(arg_names, gradients))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "compound spectral VJP requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": f"{target}_spectral_backward_compiled",
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": expected_mode,
            "evidence_target": "x86_avx512" if target == "x86" else "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": "save_inputs",
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "schedule_artifact_hash": package.schedule_artifact_hash,
            "tile_program_digest": package.tile_program_digest,
            "frontend_authority": "tracer",
        },
    )


@register_native_vjp_plugin(
    "lion",
    family="lion_vjp",
    schedule_consumer="schedule.lion_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_lion_backward",
        "rocm": "rocm.gfx1151_lion_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_lion(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Build and launch one non-reexecuting, state-lineage Lion package."""
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    from .native_lion_vjp import build_native_lion_vjp_package

    if not source_graph_ir:
        raise TesseraJitError("Lion VJP requires tracer-owned source Graph IR")
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    try:
        package = build_native_lion_vjp_package(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangents=cotangents,
            frontend_certificate=frontend_certificate,
        )
        result = launch(
            RuntimeArtifact(metadata=package.runtime_metadata()),
            tuple(
                np.ascontiguousarray(np.asarray(value), dtype=np.float32)
                for value in (*ordered_inputs, *cotangents)
            ),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    execution_mode = "cpu_avx512" if target == "x86" else "hip_runtime"
    if not result.get("ok") or result.get("execution_mode") != execution_mode:
        raise TesseraJitError(
            f"verified {target} Lion backward launch failed: "
            + str(result.get("reason"))
        )
    gradients = tuple(result["output"])
    by_name = dict(zip(package.argument_names, gradients, strict=True))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "Lion VJP requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": f"{target}_lion_bwd_compiled",
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "evidence_target": "x86_avx512" if target == "x86" else "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": "none",
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "frontend_certificate_digest": package.frontend_certificate.digest,
            "schedule_artifact_hash": package.scheduled.schedule_digest,
            "state_lineage_digest": package.scheduled.state_contract["artifact_hash"],
            "tile_program_digest": package.tile_program_digest,
            "artifact_hash": package.artifact_hash,
            "frontend_authority": "tracer",
            "proof_mode": "structural_non_reexecuting",
        },
    )


def _execute_stateful_package(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
    builder: Callable[..., Any],
    residual_policy: str,
) -> NativeVJPResult:
    """Build and launch one state-lineage package without replaying Graph IR."""
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    if not source_graph_ir:
        raise TesseraJitError(
            f"{declaration.family} requires tracer-owned source Graph IR"
        )
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    try:
        package = builder(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangents=cotangents,
            frontend_certificate=frontend_certificate,
        )
        result = launch(
            RuntimeArtifact(metadata=package.runtime_metadata()),
            tuple(
                np.ascontiguousarray(np.asarray(value), dtype=np.float32)
                for value in (*ordered_inputs, *cotangents)
            ),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    execution_mode = "cpu_avx512" if target == "x86" else "hip_runtime"
    if not result.get("ok") or result.get("execution_mode") != execution_mode:
        raise TesseraJitError(
            f"verified {target} {declaration.family} launch failed: "
            + str(result.get("reason"))
        )
    gradients = tuple(result["output"])
    by_name = dict(zip(package.argument_names, gradients, strict=True))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            f"{declaration.family} requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": package.compiler_path,
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "evidence_target": "x86_avx512" if target == "x86" else "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": residual_policy,
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "frontend_certificate_digest": package.frontend_certificate.digest,
            "schedule_artifact_hash": package.scheduled.schedule_digest,
            "state_lineage_digest": package.state_lineage_digest,
            "tile_program_digest": package.tile_program_digest,
            "artifact_hash": package.artifact_hash,
            "frontend_authority": "tracer",
            "proof_mode": "structural_non_reexecuting",
        },
    )


@register_native_vjp_plugin(
    "sgd",
    family="optimizer_vjp",
    schedule_consumer="schedule.optimizer_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_sgd_backward",
        "rocm": "rocm.gfx1151_sgd_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
@register_native_vjp_plugin(
    "momentum", "nesterov",
    family="optimizer_vjp",
    schedule_consumer="schedule.optimizer_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_momentum_backward",
        "rocm": "rocm.gfx1151_momentum_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
@register_native_vjp_plugin(
    "adam", "adamw",
    family="optimizer_vjp",
    schedule_consumer="schedule.optimizer_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={"rocm": "rocm.gfx1151_adam_backward"},
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_optimizer_vjp(
    *, source: Any, target: str, ordered_inputs: Sequence[Any],
    arg_names: Sequence[str], source_arg_names: Sequence[str],
    out_cotangents: Any, wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration, source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    from .native_stateful_vjp import build_native_optimizer_vjp_package

    return _execute_stateful_package(
        source=source, target=target, ordered_inputs=ordered_inputs,
        arg_names=arg_names, out_cotangents=out_cotangents,
        wrt_names=wrt_names, declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
        builder=build_native_optimizer_vjp_package,
        residual_policy="save_explicit_optimizer_state",
    )


@register_native_vjp_plugin(
    "adafactor",
    family="adafactor_vjp",
    schedule_consumer="schedule.adafactor_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_adafactor_backward",
        "rocm": "rocm.gfx1151_adafactor_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_adafactor(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    from .native_stateful_vjp import build_native_adafactor_vjp_package

    return _execute_stateful_package(
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        arg_names=arg_names,
        out_cotangents=out_cotangents,
        wrt_names=wrt_names,
        declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
        builder=build_native_adafactor_vjp_package,
        residual_policy="recompute_optimizer_state",
    )


@register_native_vjp_plugin(
    "gated_deltanet",
    "kimi_delta_attention",
    "modified_delta_attention",
    family="sequence_mixer_backward",
    schedule_consumer="schedule.sequence_mixer_backward",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_sequence_mixer_backward",
        "rocm": "rocm.gfx1151_sequence_mixer_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_sequence_mixer(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    from .native_stateful_vjp import build_native_sequence_mixer_vjp_package

    return _execute_stateful_package(
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        arg_names=arg_names,
        out_cotangents=out_cotangents,
        wrt_names=wrt_names,
        declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
        builder=build_native_sequence_mixer_vjp_package,
        residual_policy="launch_owned_checkpoint_workspace",
    )


@register_native_vjp_plugin(
    "flash_attn",
    "gqa_attention",
    "mqa_attention",
    family="attention_backward",
    schedule_consumer="schedule.attention_backward",
    tile_consumer="tile.attention_backward_kernel",
    target_consumers={
        "x86": "x86.avx512_attention_backward",
        "rocm": "rocm.gfx1151_attention_backward_program",
    },
    differential_policy="zero_dropout_attention",
)
def _execute_attention(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Build and execute one traced canonical attention reverse package."""
    from .native_attention_vjp import (
        build_native_attention_vjp_package,
        execute_native_attention_vjp_package,
    )

    if not source_graph_ir:
        raise TesseraJitError("attention VJP requires tracer-owned source Graph IR")
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    if len(cotangents) != 1:
        raise TesseraJitError("attention VJP requires one output cotangent")
    try:
        package = build_native_attention_vjp_package(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            source_arg_names=source_arg_names,
            out_cotangent=cotangents[0],
        )
        gradients = execute_native_attention_vjp_package(
            package,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangent=cotangents[0],
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    by_name = dict(zip(package.operand_names, gradients, strict=True))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "attention VJP has no physical gradient for operands: "
            + ", ".join(missing)
        )
    execution_mode = "cpu_avx512" if target == "x86" else "hip_runtime"
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": f"{target}_flash_attn_bwd_compiled",
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "evidence_target": "x86_avx512" if target == "x86" else "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": package.scheduled.lse_checkpoint_selection,
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "schedule_artifact_hash": package.schedule_artifact_hash,
            "tile_program_digest": package.tile_program_digest,
            "native_image_digest": package.native_image_digest,
            "artifact_hash": package.artifact_hash,
            "frontend_authority": "tracer",
        },
    )


def execute_native_vjp_family(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str] | None = None,
    out_cotangents: Any,
    wrt_names: Sequence[str],
    source_graph_ir: str | None = None,
    frontend_certificate: Any | None = None,
) -> NativeVJPResult | None:
    """Execute the sole registered owner for ``source``, or return no match."""
    bare = source.op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None:
        return None
    declaration, executor = entry
    if target not in declaration.target_consumers:
        return None
    return executor(
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        arg_names=arg_names,
        source_arg_names=source_arg_names or arg_names,
        out_cotangents=out_cotangents,
        wrt_names=wrt_names,
        declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
    )


def native_vjp_plugin_declarations() -> Mapping[str, NativeVJPPluginDeclaration]:
    """Return the explicit Graph/Schedule/Tile/Target ownership registry."""
    return {name: declaration for name, (declaration, _) in sorted(_PLUGINS.items())}


def native_vjp_plugin_owners() -> Mapping[str, str]:
    """Stable inspection surface for totality and audit tests."""
    return {name: executor.__name__ for name, (_, executor) in sorted(_PLUGINS.items())}


def native_vjp_plugin_available(op_name: str, target: str) -> bool:
    """Whether one exact Graph op/Target pair has migrated plugin authority."""
    bare = op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    return entry is not None and target in entry[0].target_consumers


def native_vjp_frontend_proof_policy(op_name: str, target: str) -> str | None:
    """Return the declared frontend proof policy for one owned target pair."""
    bare = op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None or target not in entry[0].target_consumers:
        return None
    return entry[0].differential_policy


def native_vjp_differential_safe(source: Any, target: str, effect: str) -> bool:
    """Whether the plugin permits the concrete call to run in a parity gate."""
    bare = source.op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None or target not in entry[0].target_consumers:
        return False
    policy = entry[0].differential_policy
    if policy == "non_reexecuting_state_lineage":
        return False
    if effect == "pure":
        return True
    if policy == "zero_dropout_attention":
        dropout = source.kwargs.get("dropout_p", source.kwargs.get("dropout", 0.0))
        return float(dropout or 0.0) == 0.0
    return False


def native_vjp_differential_effect_exemptions(
    source: Any, target: str, effect: str
) -> tuple[str, ...]:
    """Return exact Graph ops admitted by a validated plugin effect policy."""
    bare = source.op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None or target not in entry[0].target_consumers:
        return ()
    if (
        entry[0].differential_policy == "zero_dropout_attention"
        and native_vjp_differential_safe(source, target, effect)
    ):
        return (source.op_name,)
    return ()


__all__ = [
    "NativeVJPPluginDeclaration",
    "NativeVJPResult",
    "execute_native_vjp_family",
    "native_vjp_differential_effect_exemptions",
    "native_vjp_plugin_available",
    "native_vjp_differential_safe",
    "native_vjp_frontend_proof_policy",
    "native_vjp_plugin_declarations",
    "native_vjp_plugin_owners",
]
