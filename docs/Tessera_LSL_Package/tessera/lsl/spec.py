from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

@dataclass
class Spec:
    task: str
    description: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    inductive_bias: List[str] = field(default_factory=list)
    constraints: List[Tuple[str, Any]] = field(default_factory=list)
    capabilities: List[Tuple[str, Any]] = field(default_factory=list)

def learning_objective(task: str, description: str = "", *, inputs=None, outputs=None, inductive_bias=None) -> Spec:
    return Spec(task=task, description=description,
                inputs=dict(inputs or {}), outputs=dict(outputs or {}),
                inductive_bias=list(inductive_bias or []))

class constraint:
    @staticmethod
    def latency(s: str):   return ("latency", s)
    @staticmethod
    def accuracy(s: str):  return ("accuracy", s)
    @staticmethod
    def memory(s: str):    return ("memory", s)
    @staticmethod
    def energy(s: str):    return ("energy", s)
    @staticmethod
    def size(s: str):      return ("size", s)

class capability:
    @staticmethod
    def few_shot(enabled=True, k=16, adapters=None, lora_rank=None):
        return ("few_shot", dict(enabled=enabled, k=k, adapters=adapters, lora_rank=lora_rank))
    @staticmethod
    def uncertainty(outputs=("mean","std")):
        return ("uncertainty", list(outputs))
    @staticmethod
    def interpretability(methods=("grad_cam",)):
        return ("interpretability", list(methods))
    @staticmethod
    def robustness(**kw):
        return ("robustness", kw)

# In-place add: constraints vs. capabilities
def _iadd(self: Spec, item):
    key = item[0]
    if key in {"latency","accuracy","memory","energy","size"}:
        self.constraints.append(item)
    else:
        self.capabilities.append(item)
    return self

Spec.__iadd__ = _iadd
