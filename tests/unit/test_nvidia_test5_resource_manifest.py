from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[2] / "benchmarks/nvidia/build_test5_resource_manifest.py"
SPEC = importlib.util.spec_from_file_location("test5_resources", PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


def test_manifest_maps_multi_kernel_staged_route_and_spill_evidence():
    payload = {"rows": [
        {"kernel": "mm_f32(...) ", "resource_fingerprint": "sha256:mm"},
        {"kernel": "scale_mask(...) ", "resource_fingerprint": "sha256:scale"},
        {"kernel": "softmax(...) ", "resource_fingerprint": "sha256:soft"},
        {"kernel": "tsr_flash_bwd(...) ", "resource_fingerprint": "sha256:bwd",
         "spills_detected": True},
    ]}
    out = mod.build([payload])
    assert len(out["routes"]["staged_paged_attention"]) == 3
    assert out["details"]["generated_atomic_vjp"][0]["spills_detected"] is True
