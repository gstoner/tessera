from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class Region:
    id: int
    x0: int; y0: int; x1: int; y1: int  # inclusive bounds

@dataclass
class LayoutPlan:
    wafer_w: int
    wafer_h: int
    regions: List[Region]
    colors: Dict[int, int]  # region_id -> color (for routing classes)

def plan_layout(grid_w: int, grid_h: int, num_regions_x: int, num_regions_y: int) -> LayoutPlan:
    """
    Simple rectangular partitioner over a wafer grid.
    Returns regions + a naive 'color' per region (row-major index).
    """
    regions = []
    stepx = grid_w // num_regions_x
    stepy = grid_h // num_regions_y
    rid = 0
    for by in range(num_regions_y):
        for bx in range(num_regions_x):
            x0 = bx * stepx
            y0 = by * stepy
            x1 = (bx + 1) * stepx - 1
            y1 = (by + 1) * stepy - 1
            regions.append(Region(rid, x0, y0, x1, y1))
            rid += 1
    colors = {r.id: r.id % 8 for r in regions}  # 8 "routing classes" as a starting point
    return LayoutPlan(grid_w, grid_h, regions, colors)

def to_json(plan: LayoutPlan) -> str:
    import json
    return json.dumps({
        "wafer": {"w": plan.wafer_w, "h": plan.wafer_h},
        "regions": [asdict(r) for r in plan.regions],
        "colors": plan.colors,
    }, indent=2)
