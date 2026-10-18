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


class BoundedAttentionJVPCall:
    def __init__(self, fn, args, kwargs, bounds):
        from .native_attention_jvp_runtime import PreparedAttentionJVP
        from .scheduled_matmul import find_tessera_opt
        from .frontend_authority import certify_resident_frontends
        from dataclasses import replace
        import numpy as np
        ordered = fn._ordered_inputs(args, kwargs)
        if ordered is None:
            raise ValueError("bounded attention JVP input binding disagrees")
        if any(hasattr(value, "__cuda_array_interface__") for value in ordered):
            if not all(hasattr(value, "__cuda_array_interface__") for value in ordered):
                raise ValueError("attention JVP requires all resident roots or all host roots")
            specs = fn._resident_frontend_specs(ordered)
        else:
            arrays = tuple(np.asarray(value) for value in ordered)
            specs = tuple((tuple(value.shape), value.dtype) for value in arrays)
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
    key = ("native_bounded_attention_jvp", bounds,
           tuple(fn.differentiation_request.wrt_indices),
           hashlib.sha256(graph.encode()).hexdigest(), threading.current_thread())
    if fn._native_prepared_jvp_pid is None:
        fn._native_prepared_jvp_pid = os.getpid()
    try:
        with fn._native_prepared_jvp_lock:
            call = fn._native_prepared_jvp_calls.get(key)
            if call is None:
                call = BoundedAttentionJVPCall(fn, args, kwargs, bounds)
                if len(fn._native_prepared_jvp_calls) >= 8:
                    retired = next(iter(fn._native_prepared_jvp_calls))
                    fn._native_prepared_jvp_calls.pop(retired).close()
                fn._native_prepared_jvp_calls[key] = call
            outputs = call.invoke((*ordered, *tangents))
            fn.last_jvp_execution = dict(call.receipt)
            return outputs
    except (ValueError, RuntimeError) as error:
        raise TesseraJitError(str(error)) from error
