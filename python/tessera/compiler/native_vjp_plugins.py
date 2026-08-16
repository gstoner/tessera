"""Family-owned execution for native reverse products.

``JitFn`` owns call binding and records the selected execution result.  It must
not own another operation-family dispatch table or construct backend packages.
This module is the migration boundary for reverse products: every registered
family declares the Graph, Schedule, Tile, and Target consumers that own its
physical execution.

The first migrated family is normalization.  Other native backward families
remain compatibility paths in ``JitFn`` until they acquire equivalent lineage
and differential gates.
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

    def validate(self) -> None:
        if not self.family or not self.graph_consumers:
            raise ValueError("native VJP declaration requires family and Graph consumers")
        if not self.schedule_consumer.startswith("schedule."):
            raise ValueError("native VJP declaration requires a Schedule consumer")
        if not self.tile_consumer.startswith("tile."):
            raise ValueError("native VJP declaration requires a Tile consumer")
        if self.migration_state not in {"canonical", "canonical_composite"}:
            raise ValueError("native VJP declaration has an invalid migration state")
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
) -> Callable[[Executor], Executor]:
    """Register one explicit Graph-op owner; duplicate ownership is invalid."""
    declaration = NativeVJPPluginDeclaration(
        family=family,
        graph_consumers=tuple(f"tessera.{name}" for name in op_names),
        schedule_consumer=schedule_consumer,
        tile_consumer=tile_consumer,
        target_consumers=dict(target_consumers),
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
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
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


def execute_native_vjp_family(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
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
        out_cotangents=out_cotangents,
        wrt_names=wrt_names,
        declaration=declaration,
    )


def native_vjp_plugin_declarations() -> Mapping[str, NativeVJPPluginDeclaration]:
    """Return the explicit Graph/Schedule/Tile/Target ownership registry."""
    return {name: declaration for name, (declaration, _) in sorted(_PLUGINS.items())}


def native_vjp_plugin_owners() -> Mapping[str, str]:
    """Stable inspection surface for totality and audit tests."""
    return {name: executor.__name__ for name, (_, executor) in sorted(_PLUGINS.items())}


__all__ = [
    "NativeVJPPluginDeclaration",
    "NativeVJPResult",
    "execute_native_vjp_family",
    "native_vjp_plugin_declarations",
    "native_vjp_plugin_owners",
]
