"""Public bounded JVP orchestration over verified native attention packages."""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

from .matmul_pipeline import normalize_target_kind


def validate_bounds(target, bounds):
    from .jit import TesseraJitError
    if normalize_target_kind(target) != "nvidia_sm120":
        raise TesseraJitError("bounded attention JVP requires exact nvidia_sm120")
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2 or
            any(type(value) is not int or value <= 0 for value in bounds)):
        raise TesseraJitError("attention sequence bounds require positive Sq/Sk capacities")
    return tuple(bounds)


def _primal_specs(fn, ordered):
    if any(hasattr(value, "__cuda_array_interface__") for value in ordered):
        if not all(hasattr(value, "__cuda_array_interface__") for value in ordered):
            raise ValueError("attention JVP requires all resident roots or all host roots")
        return fn._resident_frontend_specs(ordered)
    import numpy as np
    arrays = tuple(np.asarray(value) for value in ordered)
    return tuple((tuple(value.shape), value.dtype) for value in arrays)


class BoundedAttentionJVPCall:
    def __init__(self, fn, args, kwargs, bounds, *, specs=None):
        from .native_attention_jvp_runtime import PreparedAttentionJVP
        from .scheduled_matmul import find_tessera_opt
        from .frontend_authority import certify_resident_frontends
        from dataclasses import replace
        if specs is None:
            ordered = fn._ordered_inputs(args, kwargs)
            if ordered is None:
                raise ValueError("bounded attention JVP input binding disagrees")
            specs = _primal_specs(fn, ordered)
        traced = fn._specialized_autodiff_module(args, kwargs)
        traced = replace(traced, module_attrs={**traced.module_attrs,
            "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
        self.certificate = certify_resident_frontends(
            legacy_module=fn._ensure_legacy_graph_ir(), tracer_module=traced,
            signature=specs, graph_consumers=("tessera.flash_attn", "tessera.attention"))
        fn.last_frontend_differential = self.certificate
        compiler = find_tessera_opt()
        if compiler is None:
            raise ValueError("bounded attention JVP requires the selected native compiler")
        llvm = os.environ.get("TESSERA_NATIVE_STORAGE_LLVM_BIN",
                              os.environ.get("TESSERA_LLVM_BIN"))
        if not llvm:
            raise ValueError("bounded attention JVP requires the selected LLVM tools")
        program = fn.compile_native_attention_jvp(
            *args, compiler=compiler, llvm_bin=llvm, sequence_bounds=bounds, **kwargs)
        count = len(program.input_indices)
        names = [f"primal_{index}" for index in range(count)]
        names += [f"tangent_{program.input_indices[role]}" for role in program.active]
        self.metadata = dict(program_json=program.to_json(),
                             program_digest=program.program_digest, arg_names=names)
        self.owner = PreparedAttentionJVP(self.metadata)
        self.program = self.owner.program
        from .attention_shape_contract import attention_buffer_shapes
        physical = attention_buffer_shapes(self.owner.symbolic_dims,
            bias=self.owner.biased, bias_shape=self.owner.symbolic_bias_shape)
        frontend = {index: tuple(physical[role])
                    for role, index in enumerate(self.program.input_indices)}
        self.fixed_primal_signature = tuple(
            (str(dtype), frontend[index]) for index, dtype in enumerate(
                self.owner.dtypes[:self.owner.count]))
        policy = self.program.pair.forward.descriptor.provenance
        self._receipt_base = dict(
            compiler_path="nvidia_sm120_jvp_compiled",
            execution_kind="native_gpu", execution_mode="cuda_runtime",
            evidence_target="nvidia_sm120", family="attention_checkpoint",
            frontend_authority="tracer", frontend_certificate=dict(self.certificate.contract),
            program_digest=self.metadata["program_digest"],
            forward_graph_ir_digest=policy["graph_ir_digest"],
            forward_schedule_ir_digest=policy["schedule_ir_digest"],
            forward_tile_ir_digest=policy["tile_ir_digest"],
            tangent_binding_digest=self.program.tangent.binding_digest,
        )
        self.receipt: dict[str, Any] = {}

    def matches_primal_signature(self, specs):
        from .attention_shape_contract import DYNAMIC_DIM, attention_buffer_shapes
        if len(specs) != len(self.fixed_primal_signature):
            return False
        for (shape, dtype), (fixed_dtype, symbolic) in zip(
                specs, self.fixed_primal_signature, strict=True):
            if str(dtype) != fixed_dtype or len(shape) != len(symbolic):
                return False
            if any(fixed != DYNAMIC_DIM and actual != fixed
                   for actual, fixed in zip(shape, symbolic, strict=True)):
                return False
        q, k, v = (specs[index][0] for index in self.program.input_indices[:3])
        dims = (q[0], q[1], k[1], q[2], k[2], q[3], v[3])
        # Match physical bias policy and Q/K/V relationships without applying
        # capacity guards here. The selected owner must reject an out-of-bounds
        # sequence before compilation, rather than creating another owner.
        try:
            physical = attention_buffer_shapes(dims, bias=self.owner.biased,
                bias_shape=self.owner.symbolic_bias_shape)
        except ValueError:
            return False
        return all(tuple(specs[index][0]) == physical[role]
                   for role, index in enumerate(self.program.input_indices))

    def invoke(self, values):
        outputs = self.owner.invoke(self.metadata, values)
        self.receipt = dict(self._receipt_base,
            host_preparation=("native_ordered_resident_snapshot" if any(
                hasattr(value, "__cuda_array_interface__") for value in values)
                else "native_bounded_capacity_owner"))
        return outputs

    def close(self):
        self.owner.close()


def execute(fn, args, kwargs, ordered, tangents, bounds):
    from .jit import TesseraJitError
    # Include immutable source semantics and a real Thread object. Integer thread
    # IDs can be reused; a new lifetime must not inherit another CUDA owner.
    graph = fn._ensure_legacy_graph_ir().to_mlir(target="nvidia_sm120", canonical=True)
    base_key = ("native_bounded_attention_jvp", bounds,
           tuple(fn.differentiation_request.wrt_indices),
           hashlib.sha256(graph.encode()).hexdigest(), threading.current_thread())
    if fn._native_prepared_jvp_pid is None:
        fn._native_prepared_jvp_pid = os.getpid()
    try:
        specs = _primal_specs(fn, ordered)
        with fn._native_prepared_jvp_lock:
            call = None
            for candidate_key in reversed(fn._native_prepared_jvp_calls):
                if candidate_key[:-1] != base_key:
                    continue
                candidate = fn._native_prepared_jvp_calls[candidate_key]
                if candidate.matches_primal_signature(specs):
                    key, call = candidate_key, candidate
                    break
            if call is None:
                call = BoundedAttentionJVPCall(fn, args, kwargs, bounds, specs=specs)
                key = (*base_key, call.fixed_primal_signature)
                if len(fn._native_prepared_jvp_calls) >= 8:
                    retired = next(iter(fn._native_prepared_jvp_calls))
                    fn._native_prepared_jvp_calls.pop(retired).close()
            else:
                fn._native_prepared_jvp_calls.pop(key)
            fn._native_prepared_jvp_calls[key] = call
            outputs = call.invoke((*ordered, *tangents))
            fn.last_jvp_execution = dict(call.receipt)
            return outputs
    except (ValueError, RuntimeError) as error:
        raise TesseraJitError(str(error)) from error
