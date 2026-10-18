"""Declared composite VJP ownership over original frontend Graphs.

JitFn forwards call binding here; selection, certification and packaging stay
with the composite family. Numerical execution belongs to native program owners.
"""
from dataclasses import dataclass, replace
import os
from typing import Callable

from .._jit_boundary import TesseraJitError
from .native_vjp_plugins import NativeVJPPluginDeclaration, NativeVJPResult, _record_execution_certificate


@dataclass(frozen=True)
class CompositeVJP:
    declaration: NativeVJPPluginDeclaration
    matches: Callable
    execute: Callable


def _matches_source(module, target, request):
    if target != "nvidia_sm120":
        return False
    from .prepared_nvidia_active_source import candidate
    return candidate(module, request.wrt_indices)


def _execute_source(jit, module, args, kwargs, cotangents, declaration):
    from tessera import runtime as rt
    from .effects import infer_graph_effects
    from .nvidia_source_reverse_runtime import runtime_artifact
    from .prepared_nvidia_source_reverse import PreparedSourceReverse
    ordered = jit._ordered_inputs(args, kwargs)
    if ordered is None:
        raise TesseraJitError("source reverse requires every primal root")
    seeds = tuple(cotangents) if isinstance(cotangents, (tuple, list)) else (cotangents,)
    if len(seeds) != 1:
        raise TesseraJitError("source reverse requires one explicit FP32 output cotangent")
    effect, _ = infer_graph_effects(module.functions[0].body)
    if effect.name != "pure":
        raise TesseraJitError("source reverse requires a pure original Graph")
    values = (*ordered, *seeds)
    resident = [hasattr(value, "__cuda_array_interface__") for value in values]
    def traced_module(signature):
        cached = jit._traced_frontend_specializations.get(signature)
        return cached if cached is not None else jit._traced_autodiff_module(args, kwargs)
    if any(resident):
        if not all(resident):
            raise TesseraJitError("source reverse requires all CUDA roots or all host roots")
        from .frontend_authority import certify_resident_frontends
        from .resident_nvidia_tensor import cuda_frontend_specs
        specs = cuda_frontend_specs(ordered, ranks=(2,))
        module = traced_module(tuple((str(dtype), shape) for shape, dtype in specs))
        certificate = certify_resident_frontends(
            legacy_module=jit._ensure_legacy_graph_ir(), tracer_module=module,
            signature=specs, graph_consumers=declaration.graph_consumers)
        jit.last_frontend_differential = certificate
    else:
        certificate = jit.frontend_differential(*args, **kwargs)
        module = traced_module(tuple((str(value.dtype), tuple(value.shape)) for value in ordered))
    if not _matches_source(module, "nvidia_sm120", jit.differentiation_request):
        raise TesseraJitError("certified source reverse Graph differs from its declared profile")
    # Only module attributes change. Share the verified function SSA without
    # mutating it; copying every Graph object on each call serves no contract.
    projected = replace(module, module_attrs={**module.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"',
        "tessera.native.sm120_reverse_request": "unit"})
    source = projected.to_mlir(target="nvidia_sm120", canonical=True)
    key = (declaration.family, source)
    if jit._native_prepared_vjp_pid is not None and jit._native_prepared_vjp_pid != os.getpid():
        raise RuntimeError("retained native VJP cannot cross fork")
    with jit._native_prepared_vjp_lock:
        if jit._native_prepared_vjp_pid is None:
            jit._native_prepared_vjp_pid = os.getpid()
        owner = jit._native_prepared_vjp_calls.get(key)
        if owner is None:
            owner = PreparedSourceReverse(source)
            if len(jit._native_prepared_vjp_calls) >= 8:
                retired = next(iter(jit._native_prepared_vjp_calls))
                jit._native_prepared_vjp_calls.pop(retired).close()
            jit._native_prepared_vjp_calls[key] = owner
        artifact = runtime_artifact(owner)
        launched = rt.launch(artifact, values)
        if not launched.get("ok"):
            raise TesseraJitError("compiled source reverse launch failed: " + str(launched.get("reason")))
        gradients = launched["output"]
        execution = {**owner.receipt, **{
            key: launched[key] for key in (
                "compiler_path", "execution_kind", "execution_mode", "physical_attestation", "artifact_hash")
            if key in launched}}
        execution.update(family=declaration.family, implementation="composite_family_plugin",
                         residual_policy="recompute_all", graph_consumer="tessera.rmsnorm",
                         schedule_consumer=declaration.schedule_consumer,
                         tile_consumer=declaration.tile_consumer,
                         target_consumer=declaration.target_consumers["nvidia_sm120"],
                         frontend_certificate_digest=certificate.digest)
        return _record_execution_certificate(
            declaration=declaration, source=module.functions[0].body[0], target="nvidia_sm120",
            ordered_inputs=ordered, out_cotangents=cotangents,
            result=NativeVJPResult(tuple(gradients), execution, artifact),
            frontend_certificate=certificate)


_SOURCE_DECLARATION = NativeVJPPluginDeclaration(
    family="source_normalization_matmul",
    graph_consumers=("tessera.rmsnorm", "tessera.relu", "tessera.matmul"),
    schedule_consumer="schedule.artifact", tile_consumer="tile.row_region_kernel",
    target_consumers={"nvidia_sm120": "nvidia.sm120_source_reverse_program"})
_SOURCE_DECLARATION.validate()
_COMPOSITES = {"source_normalization_matmul": CompositeVJP(
    _SOURCE_DECLARATION, _matches_source, _execute_source)}


def native_vjp_composite_declarations():
    return {name: entry.declaration for name, entry in _COMPOSITES.items()}


def execute_native_vjp_composite(jit, module, args, kwargs, *, out_cotangents):
    request = jit.differentiation_request
    if request is None:
        return None
    from .matmul_pipeline import normalize_target_kind
    target = normalize_target_kind(jit.target)
    matches = [entry for entry in _COMPOSITES.values() if entry.matches(module, target, request)]
    if len(matches) > 1:
        raise TesseraJitError("multiple composite VJP families claim this Graph")
    if not matches:
        return None
    entry = matches[0]
    return entry.execute(jit, module, args, kwargs, out_cotangents, entry.declaration)
