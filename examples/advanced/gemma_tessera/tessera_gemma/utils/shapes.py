from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ShapeSpec:
    dims: List[str]

def check_shape(name: str, actual: Tuple[int, ...], spec: ShapeSpec, symbols: Dict[str,int]) -> None:
    if len(actual) != len(spec.dims):
        raise ValueError(f"{name}: rank {len(actual)} != spec rank {len(spec.dims)}")
    for i,(a, s) in enumerate(zip(actual, spec.dims)):
        if s == "?":
            continue
        if s.isdigit():
            if a != int(s):
                raise ValueError(f"{name}: dim {i} = {a}, expected {s}")
        else:
            if s in symbols:
                if symbols[s] != a:
                    raise ValueError(f"{name}: symbolic {s} mismatch: {a} vs {symbols[s]}")
            else:
                symbols[s] = a
