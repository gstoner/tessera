"""Public composite VJP admission for the original RMSNorm/attention source."""
from dataclasses import replace
import os
import hashlib
import threading

from .._jit_boundary import TesseraJitError
from .native_vjp_plugins import NativeVJPResult, _record_execution_certificate


def matches(module, target, request):
    return (target == "nvidia_sm120" and len(module.functions) == 1
            and len(module.functions[0].args) == 3 and len(module.functions[0].result_types) == 2
            and [op.op_name for op in module.functions[0].body] == ["tessera.rmsnorm", "tessera.flash_attn"]
            and module.functions[0].body[1].kwargs.get("lse_checkpoint") == "saved")


def execute(jit, module, args, kwargs, cotangents, declaration, *, sequence_bounds=None, bounded_key=None):
    from .native_source_attention_reverse import export_native_source_attention_reverse
    from .prepared_source_attention_reverse import PreparedSourceAttentionReverse
    from .effects import infer_graph_effects
    from .native_vjp_plugins import native_vjp_differential_safe, native_vjp_differential_effect_exemptions
    attention = module.functions[0].body[1]
    effect, _ = infer_graph_effects(module.functions[0].body)
    try:
        zero_dropout = float(attention.kwargs.get("dropout_p", attention.kwargs.get("dropout", 0.0)) or 0.0) == 0.0
    except (TypeError, ValueError):
        zero_dropout = False
    if not zero_dropout or not native_vjp_differential_safe(attention, "nvidia_sm120", effect.name):
        raise TesseraJitError("source attention reverse requires zero dropout before frontend certification")
    exemptions = native_vjp_differential_effect_exemptions(attention, "nvidia_sm120", effect.name)
    ordered = jit._ordered_inputs(args, kwargs)
    seeds = tuple(cotangents) if isinstance(cotangents, (tuple, list)) else (cotangents,)
    if ordered is None or len(seeds) != 2:
        raise TesseraJitError("source attention reverse needs every primal and both O/LSE cotangents")
    values = (*ordered, *seeds)
    resident = tuple(hasattr(v, "__cuda_array_interface__") for v in values)
    if any(resident) or sequence_bounds is not None:
        if any(resident) and not all(resident):
            raise TesseraJitError("source attention reverse requires all host or all CUDA roots")
        from .frontend_authority import certify_resident_frontends
        from .resident_nvidia_tensor import cuda_frontend_specs
        if any(resident):
            specs = cuda_frontend_specs(ordered, ranks=(4,))
        else:
            import numpy as np
            specs = tuple((tuple(np.shape(v)), np.asarray(v).dtype) for v in ordered)
        signature = tuple((str(dtype), shape) for shape, dtype in specs)
        module = jit._traced_frontend_specializations.get(signature)
        if module is None:
            module = jit._traced_autodiff_module(args, kwargs)
        certificate = certify_resident_frontends(
            legacy_module=jit._ensure_legacy_graph_ir(), tracer_module=module,
            signature=specs, graph_consumers=declaration.graph_consumers)
        jit.last_frontend_differential = certificate
    else:
        certificate = jit.frontend_differential(*args, _permitted_effect_ops=exemptions, **kwargs)
        signature = tuple((str(v.dtype), tuple(v.shape)) for v in ordered)
        module = jit._traced_frontend_specializations.get(signature)
        if module is None:
            module = jit._traced_autodiff_module(args, kwargs)
    if not matches(module, "nvidia_sm120", jit.differentiation_request):
        raise TesseraJitError("certified source attention reverse differs from its declared profile")
    projected = replace(module, module_attrs={**module.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"',
        "tessera.native.source_attention_reverse_request": "unit",
        "tessera.native.source_attention_reverse_auto_request": "unit"})
    if sequence_bounds is not None:
        projected = replace(projected, module_attrs={**projected.module_attrs,
            "tessera.attention_sequence_bounds": "array<i64: " + ", ".join(map(str, sequence_bounds)) + ">"})
    source = projected.to_mlir(target="nvidia_sm120", canonical=True)
    key = bounded_key if bounded_key is not None else (declaration.family, source)
    if jit._native_prepared_vjp_pid is not None and jit._native_prepared_vjp_pid != os.getpid():
        raise ValueError("source attention reverse cannot cross fork")
    with jit._native_prepared_vjp_lock:
        if jit._native_prepared_vjp_pid is None:
            jit._native_prepared_vjp_pid = os.getpid()
        owner = jit._native_prepared_vjp_calls.get(key)
        if owner is None:
            owner = PreparedSourceAttentionReverse(export_native_source_attention_reverse(source))
            if len(jit._native_prepared_vjp_calls) >= 8:
                retired = next(iter(jit._native_prepared_vjp_calls))
                jit._native_prepared_vjp_calls.pop(retired).close()
            jit._native_prepared_vjp_calls[key] = owner
        if sequence_bounds is not None:
            owner._bounded_frontend_module = module
            owner._bounded_frontend_certificate = certificate
        return _invoke(jit, owner, module, certificate, ordered, cotangents, values, declaration)


def _invoke(jit, owner, module, certificate, ordered, cotangents, values, declaration):
    from tessera import runtime as rt
    from .portable_source_attention_reverse import COMPILER_PATH
    gradients = owner(*values)
    receipt = owner.receipt
    if receipt is None:
        raise RuntimeError("source attention reverse completed without a native receipt")
    # The owner retains immutable compiler-validated images and contracts.
    # Warm calls need neither a JSON wrapper nor portable revalidation.
    # Artifact export materializes an independent manifest only on request.
    attestation = rt._physical_execution_attestation(
        target="nvidia_sm120", execution_kind="native_gpu", execution_mode="cuda_driver",
        artifact_hash=owner._contract_digest)
    execution = {**receipt, "compiler_path": COMPILER_PATH, "execution_mode": "cuda_driver",
                 "physical_attestation": attestation, "artifact_hash": owner._contract_digest}
    execution.update(family=declaration.family, implementation="composite_family_plugin",
                     residual_policy="recompute_all", graph_consumer="tessera.rmsnorm",
                     schedule_consumer=declaration.schedule_consumer,
                     tile_consumer=declaration.tile_consumer,
                     target_consumer=declaration.target_consumers["nvidia_sm120"],
                     source_graph_ir_digest=owner._source_graph_digest,
                     frontend_certificate_digest=certificate.digest)
    result = _record_execution_certificate(
        declaration=declaration, source=module.functions[0].body[0], target="nvidia_sm120",
        ordered_inputs=ordered, out_cotangents=cotangents,
        result=NativeVJPResult(tuple(gradients), execution, None),
        frontend_certificate=certificate)
    jit._native_backward_artifact_factory = owner.runtime_artifact
    return result


def execute_bounded(jit, args, kwargs, cotangents, declaration, bounds):
    """Reuse one certified original-source capacity owner across Sq/Sk frames.

    The certificate describes the original static trace. Each execution receipt
    independently binds actual input/gradient metadata and the native shape guard.
    """
    from .native_bounded_attention_jvp import _primal_specs
    ordered = jit._ordered_inputs(args, kwargs)
    if ordered is None:
        raise TesseraJitError("bounded source reverse requires every primal root")
    specs = _primal_specs(jit, ordered)
    if len(specs) != 3 or any(len(shape) != 4 for shape, _ in specs):
        raise TesseraJitError("bounded source reverse requires three rank-four roots")
    fixed = tuple((str(dtype), tuple(n for axis, n in enumerate(shape) if axis != 2))
                  for shape, dtype in specs)
    graph = jit._ensure_legacy_graph_ir().to_mlir(target="nvidia_sm120", canonical=True)
    key = ("bounded_source_attention_reverse", bounds,
           tuple(jit.differentiation_request.wrt_indices),
           hashlib.sha256(graph.encode()).hexdigest(), threading.current_thread(), fixed)
    if jit._native_prepared_vjp_pid is not None and jit._native_prepared_vjp_pid != os.getpid():
        raise ValueError("source attention reverse cannot cross fork")
    with jit._native_prepared_vjp_lock:
        owner = jit._native_prepared_vjp_calls.get(key)
        if owner is None:
            module = jit._specialized_autodiff_module(args, kwargs)
            if not matches(module, "nvidia_sm120", jit.differentiation_request):
                raise TesseraJitError("bounded native backward requires RMSNorm and saved-LSE attention")
            return execute(jit, module, args, kwargs, cotangents, declaration,
                           sequence_bounds=bounds, bounded_key=key)
        seeds = tuple(cotangents) if isinstance(cotangents, (tuple, list)) else (cotangents,)
        if len(seeds) != 2:
            raise TesseraJitError("source attention reverse needs both O/LSE cotangents")
        values = (*ordered, *seeds)
        resident = tuple(hasattr(v, "__cuda_array_interface__") for v in values)
        if any(resident) and not all(resident):
            raise TesseraJitError("source attention reverse requires all host or all CUDA roots")
        certificate = owner._bounded_frontend_certificate
        jit.last_frontend_differential = certificate
        return _invoke(jit, owner, owner._bounded_frontend_module, certificate,
                       ordered, cotangents, values, declaration)
