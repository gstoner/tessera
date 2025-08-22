from .spec import Spec
from typing import Any, Dict

def _mock_choose_config(spec: Spec, target: str) -> Dict[str, Any]:
    # Extremely simplified: pick a 'config' based on inductive bias and constraints
    arch = "resnet" if any("cnn" in s for s in spec.inductive_bias) else "vit"
    if any("group-equivariance:so2" in s for s in spec.inductive_bias):
        arch += "+so2"
    precision = "bf16"
    if any(c[0]=="energy" for c in spec.constraints):
        precision = "int8-mixed"  # pretend to favor int8 when energy is constrained
    schedule = {"BM":128,"BN":256,"BK":64,"stages":3}
    return {"arch": arch, "precision": precision, "schedule": schedule, "target": target}

def solve(spec: Spec, *, data=None, metrics=None, search=None, training=None, target: str = "nvidia:a100:sm80"):
    cfg = _mock_choose_config(spec, target)
    report = {
        "ok": True,
        "notes": "Prototype solver; attach real IR builders & autotuner here.",
        "constraints": spec.constraints,
        "capabilities": spec.capabilities,
        "chosen": cfg,
    }
    artifact = {
        "graph_ir": "<mock-graph-ir>",
        "schedule_ir": "<mock-schedule-ir>",
        "tile_ir": "<mock-tile-ir>",
        "report": report,
    }
    print("[LSL] task:", spec.task)
    print("[LSL] chosen config:", cfg)
    return artifact
