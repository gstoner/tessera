
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Tensor:
    shape: Tuple[int, ...]
    dtype: str = "fp32"
    device: int = -1

@dataclass
class Node:
    op: str
    inputs: List[Tensor]
    attrs: dict

class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
    def add(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(shape=a.shape, dtype=a.dtype, device=a.device)
        self.nodes.append(Node("add", [a,b], {}))
        return out
