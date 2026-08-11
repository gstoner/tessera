"""Family-owned planning for executable native forward products.

``JitFn`` owns Python call binding and package caching, but it must not become a
second lowering registry.  This module is the canonical family-plugin boundary:
each plugin consumes one verified Graph operation and produces an immutable
ordered child-package plan.  The C++ forward autodiff pass remains the semantic
authority for the paired JVP IR bound by the parent artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .native_jvp import child_digest


@dataclass(frozen=True)
class NativeJVPFamilyPlan:
    family: str
    steps: tuple[Mapping[str, Any], ...]


Planner = Callable[..., NativeJVPFamilyPlan]
_PLUGINS: dict[str, Planner] = {}


def register_native_jvp_plugin(*op_names: str) -> Callable[[Planner], Planner]:
    """Register one explicit Graph-op consumer; duplicate ownership is invalid."""
    def decorate(planner: Planner) -> Planner:
        for name in op_names:
            if name in _PLUGINS:
                raise RuntimeError(f"native JVP plugin already owns {name!r}")
            _PLUGINS[name] = planner
        return planner
    return decorate


def _step(
    step_id: str,
    child: Mapping[str, Any],
    inputs: Sequence[str],
    *,
    output_index: int = -1,
    outputs: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "id": step_id,
        "child_digest": child_digest(child),
        "child_metadata": dict(child),
        "inputs": list(inputs),
        "output_index": output_index,
    }
    if outputs is not None:
        result["outputs"] = list(outputs)
    return result


def _execution(target: str, execution_mode: str) -> dict[str, Any]:
    return {
        "target": target,
        "executable": True,
        "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
        "execution_mode": execution_mode,
    }


@register_native_jvp_plugin("reduce", "sum", "mean")
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


@register_native_jvp_plugin("fft", "ifft", "rfft", "irfft")
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


@register_native_jvp_plugin("dct")
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


@register_native_jvp_plugin("rmsnorm", "rmsnorm_safe", "layer_norm")
def _plan_normalization(*, source: Any, primal_inputs: Sequence[Any],
                        wrt_indices: tuple[int, ...], target: str,
                        execution_mode: str, **_: Any) -> NativeJVPFamilyPlan:
    if not wrt_indices or 0 not in wrt_indices:
        raise ValueError("normalization JVP requires an active data input")
    operands = [f"primal_{index}" for index in range(len(primal_inputs))]
    names = operands + [f"tangent_{index}" for index in wrt_indices]
    child = {
        **_execution(target, execution_mode),
        "compiler_path": f"{target}_norm_jvp_compiled",
        "autodiff_phase": "forward",
        "wrt_indices": list(wrt_indices),
        "arg_names": names,
        "ops": [{"op_name": source.op_name, "result": source.result,
                 "operands": operands, "kwargs": dict(source.kwargs)}],
    }
    return NativeJVPFamilyPlan("normalization", (
        _step("normalization_product", child, names, outputs=("primal", "tangent")),
    ))


@register_native_jvp_plugin("spectral_filter", "spectral_conv", "stft", "istft")
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
    planner = _PLUGINS.get(bare)
    if planner is None:
        raise ValueError(
            f"no native {target} JVP family plugin exists for {bare!r}; "
            "the compiler IR transform remains available"
        )
    return planner(
        source=source,
        primal_inputs=primal_inputs,
        wrt_indices=wrt_indices,
        target=target,
        execution_mode=execution_mode,
    )


def native_jvp_plugin_owners() -> Mapping[str, str]:
    """Stable inspection surface used by registry-totality tests and docs."""
    return {name: planner.__name__ for name, planner in sorted(_PLUGINS.items())}


__all__ = [
    "NativeJVPFamilyPlan",
    "native_jvp_plugin_owners",
    "plan_native_jvp_family",
    "register_native_jvp_plugin",
]
