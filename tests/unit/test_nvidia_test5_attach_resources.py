from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[2] / "benchmarks/nvidia/attach_test5_resources.py"
SPEC = importlib.util.spec_from_file_location("attach_test5", PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


def test_serving_mode_gets_route_resources_and_fingerprint():
    payload = {"runs": [{"mode": "fused_paged_attention"}]}
    resource = {"resource_fingerprint": "sha256:r",
                "spill_evidence_complete": True, "spills_detected": False}
    out = mod.attach(payload, {"details": {"fused_paged_attention": [resource]}},
                     "sha256:compiler")
    row = out["runs"][0]
    assert row["selected_route"] == "fused_paged_attention"
    assert row["resource_evidence_complete"] is True
    assert row["compiler_fingerprint"] == "sha256:compiler"
