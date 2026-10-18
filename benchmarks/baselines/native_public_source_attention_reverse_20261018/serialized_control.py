"""Public composite VJP admission for the original RMSNorm/attention source."""
from dataclasses import replace
import os

from .._jit_boundary import TesseraJitError
from .native_vjp_plugins import NativeVJPResult, _record_execution_certificate


def matches(module, target, request):
    return (target == "nvidia_sm120" and len(module.functions) == 1
            and len(module.functions[0].args) == 3 and len(module.functions[0].result_types) == 2
            and [op.op_name for op in module.functions[0].body] == ["tessera.rmsnorm", "tessera.flash_attn"]
            and module.functions[0].body[1].kwargs.get("lse_checkpoint") == "saved")


def execute(jit, module, args, kwargs, cotangents, declaration):
    from tessera import runtime as rt
    from .native_source_attention_reverse import export_native_source_attention_reverse
    from .prepared_source_attention_reverse import PreparedSourceAttentionReverse
    from .source_attention_reverse_runtime import runtime_artifact
    ordered = jit._ordered_inputs(args, kwargs)
    seeds = tuple(cotangents) if isinstance(cotangents, (tuple, list)) else (cotangents,)
    if ordered is None or len(seeds) != 2:
        raise TesseraJitError("source attention reverse needs every primal and both O/LSE cotangents")
    values = (*ordered, *seeds)
    resident = tuple(hasattr(v, "__cuda_array_interface__") for v in values)
    if any(resident):
        if not all(resident):
            raise TesseraJitError("source attention reverse requires all host or all CUDA roots")
        from .frontend_authority import certify_resident_frontends
        from .resident_nvidia_tensor import cuda_frontend_specs
        specs = cuda_frontend_specs(ordered, ranks=(4,))
        signature = tuple((str(dtype), shape) for shape, dtype in specs)
        module = jit._traced_frontend_specializations.get(signature)
        if module is None:
            module = jit._traced_autodiff_module(args, kwargs)
        certificate = certify_resident_frontends(
            legacy_module=jit._ensure_legacy_graph_ir(), tracer_module=module,
            signature=specs, graph_consumers=declaration.graph_consumers)
        jit.last_frontend_differential = certificate
    else:
        certificate = jit.frontend_differential(*args, _permitted_effect_ops=("tessera.flash_attn",), **kwargs)
        signature = tuple((str(v.dtype), tuple(v.shape)) for v in ordered)
        module = jit._traced_frontend_specializations.get(signature)
        if module is None:
            module = jit._traced_autodiff_module(args, kwargs)
    if not matches(module, "nvidia_sm120", jit.differentiation_request):
        raise TesseraJitError("certified source attention reverse differs from its declared profile")
    projected = replace(module, module_attrs={**module.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"',
        "tessera.native.source_attention_reverse_request": "unit"})
    source = projected.to_mlir(target="nvidia_sm120", canonical=True)
    key = (declaration.family, source)
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
        artifact = runtime_artifact(owner)
        launched = rt.launch(artifact, values)
        if not launched.get("ok"):
            raise TesseraJitError("native source attention reverse failed: " + str(launched.get("reason")))
        execution = {**owner.receipt, **{name: launched[name] for name in (
            "compiler_path", "execution_kind", "execution_mode", "physical_attestation", "artifact_hash")}}
        execution.update(family=declaration.family, implementation="composite_family_plugin",
                         residual_policy="recompute_all", graph_consumer="tessera.rmsnorm",
                         schedule_consumer=declaration.schedule_consumer,
                         tile_consumer=declaration.tile_consumer,
                         target_consumer=declaration.target_consumers["nvidia_sm120"],
                         source_graph_ir_digest=owner.manifest()["plan"]["source_graph_sha256"],
                         frontend_certificate_digest=certificate.digest)
        return _record_execution_certificate(
            declaration=declaration, source=module.functions[0].body[0], target="nvidia_sm120",
            ordered_inputs=ordered, out_cotangents=cotangents,
            result=NativeVJPResult(tuple(launched["output"]), execution, artifact),
            frontend_certificate=certificate)
