from typing import Dict, Any
from .csl_codegen import emit_csl
from .layout import plan_layout, to_json

def compile_to_cerebras(tessera_module: Dict[str, Any], execution_mode: str = "pipeline"):
    """
    Accepts a simplified Tessera module description (dict form for the scaffold),
    returns artifacts: CSL source(s) and a layout plan JSON string.
    """
    kind = tessera_module.get("kind", "gemm")
    name = tessera_module.get("name", f"{kind}_kernel")
    params = tessera_module.get("params", {})
    grid = tessera_module.get("grid", (64, 64))
    regions = tessera_module.get("regions", (4, 4))
    # Emit CSL
    csl = emit_csl(kind, name, params)
    # Emit layout (wafer partition)
    plan = plan_layout(grid[0], grid[1], regions[0], regions[1])
    layout_json = to_json(plan)
    # Pack artifacts
    artifacts = {
        "name": name,
        "kind": kind,
        "execution_mode": execution_mode,
        "csl": csl,
        "layout.json": layout_json,
    }
    return artifacts
