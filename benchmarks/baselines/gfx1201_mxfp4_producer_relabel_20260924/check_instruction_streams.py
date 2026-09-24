"""Instruction-stream digests of every recorded folded/packed/TN4 variant.

Run at the parent revision and at the relabel; identical streams prove the
relabel did not change any timed kernel (whole payloads carry build-specific
bytes and differ even between two builds of one revision).
"""
import json
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.inspect_gfx1201_folded_prefill import selected_symbol_isa_evidence
from tessera.compiler.rocm_mxfp4_folded import package_mxfp4_folded_prefill, prepare_folded_weights
from tessera.compiler.rocm_mxfp4_packed_folded import (
    package_mxfp4_packed_folded_prefill, prepare_packed_folded_payload)
from tessera.compiler.rocm_mxfp4_tn4_experiment import package_folded_tn4_experiment


def stream(pkg):
    ev = selected_symbol_isa_evidence(pkg.image.payload, pkg.descriptor.entry_symbol)
    return [pkg.image.pipeline_name, ev["instruction_stream_sha256"]]


PACKED = {  # exactly the flag sets benchmark_gfx1201_mxfp4_packed_folded.py builds
    "table": dict(integer_decode=False),
    "integer": dict(integer_decode=True),
    "batched_b": dict(integer_decode=True, batched_loads=True),
    "batched_a": dict(integer_decode=True, batched_a_loads=True),
    "batched_ab": dict(integer_decode=True, batched_loads=True, batched_a_loads=True),
    "batched_b_pair_scale": dict(integer_decode=True, batched_loads=True, reuse_pair_scales=True),
    "permute": dict(integer_decode=False, batched_loads=True, permute_decode=True),
    "vector_pair": dict(integer_decode=False, permute_decode=True, vector_pair_loads=True),
    "a_base": dict(integer_decode=False, batched_loads=True, permute_decode=True, a_base_hoist=True),
    "a_offset32": dict(integer_decode=False, batched_loads=True, permute_decode=True, a_offset32=True),
}
out = {}
for m, n, k in ((256, 5120, 8704), (1024, 17408, 5120), (1024, 5120, 8704)):
    case = base.Case("prefill", m, n, k)
    inputs = base._logical_inputs(case)
    tag = f"{m}x{n}x{k}"
    folded = prepare_folded_weights(inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True)
    out[f"{tag}/folded"] = stream(package_mxfp4_folded_prefill(m, n, k, folded, allow_approximate=True))
    out[f"{tag}/safe_epilogue"] = stream(package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
        entry="tessera_mxfp4_folded_safe_epilogue", safe_epilogue_scales=inputs["a_scale"]))
    out[f"{tag}/tn4"] = stream(package_folded_tn4_experiment(m, n, k, folded))
    payload = prepare_packed_folded_payload(inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True)
    for label, flags in PACKED.items():
        try:
            out[f"{tag}/packed_{label}"] = stream(package_mxfp4_packed_folded_prefill(m, payload, **flags))
        except Exception as exc:
            out[f"{tag}/packed_{label}"] = ["refused", f"{type(exc).__name__}: {str(exc)[:70]}"]
print(json.dumps(out, indent=1, sort_keys=True))
