"""Typed textual frontend for the gfx1201 MXFP8 E4M3/E8M0 K32 contract.

All scheduling, fragment construction and native lowering belong to MLIR.
Python binds checked native artifacts; MLIR owns all lowering.
"""
from __future__ import annotations

import re
from .rocm_fp8_blockscale import (BlockScaleShape, BlockScaleProgram,
                                 package_blockscale, schedule_blockscale_panel)
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

MXFP8_CONTRACTS = {
    "kn": "rocm_mxfp8_e4m3_e8m0_k32_v1",
    "nk": "rocm_mxfp8_e4m3_e8m0_k32_nk_v1",
}
MXFP8_PACKAGE_ABIS = {
    (layout, output): ("tessera.rocm.mxfp8_e4m3_e8m0_k32."
                      + ("a_bnk_sa_sb_o_m_n_k." if layout == "nk" else "a_b_sa_sb_o_m_n_k.")
                      + output + ".wide_scale.v1")
    for layout in ("kn", "nk") for output in ("f32", "bf16")
}



def mxfp8_schedule_is_supported(*, layout, staging, block_m, block_n,
                                macro_k, warps, pipeline_depth):
    """Checked physical profiles, never a Python schedule selector."""
    values = (block_m, block_n, macro_k, warps, pipeline_depth)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        return False
    if macro_k not in (32,64) or pipeline_depth != 1 or layout not in {"kn", "nk"}:
        return False
    if staging == "global":
        return macro_k == 32 and block_m == block_n == 16 and warps == 1
    return (staging == "lds" and layout == "nk" and block_m == 128
            and block_n in {64, 128} and warps == 8)


def author_mxfp8_graph(shape: BlockScaleShape, *, entry: str = "mxfp8",
                      schedule_policy: str = "auto") -> str:
    """State FP8 operands and byte scales without changing operand semantics."""
    if shape.scale_k != 32 or shape.scale_n != 1:
        raise ValueError("MXFP8 requires scale_k=32 and scale_n=1")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry):
        raise ValueError(f"invalid entry symbol {entry!r}")
    if schedule_policy not in {"auto", "seed", "lds", "lds_k64"}:
        raise ValueError("MXFP8 schedule policy must be auto, seed, lds, or lds_k64")
    if schedule_policy in ("lds","lds_k64") and shape.weight_layout != "nk":
        raise ValueError("MXFP8 LDS schedule requires NK storage")
    if schedule_policy == "lds_k64" and (shape.k < 64 or shape.k % 64):
        raise ValueError("MXFP8 lds_k64 requires whole K64 slabs")
    intent = (', tessera.rocm.mxfp8_schedule = "' + schedule_policy + '"'
              if schedule_policy != "auto" else "")
    m, n, k, groups = shape.m, shape.n, shape.k, shape.groups
    weight = f"{n}x{k}" if shape.weight_layout == "nk" else f"{k}x{n}"
    transpose = ", transposeB = true" if shape.weight_layout == "nk" else ""
    output = "bf16" if shape.output == "bf16" else "f32"
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"{intent}}} {{
  func.func @{entry}(%a: tensor<{m}x{k}xf8E4M3FN>, %b: tensor<{weight}xf8E4M3FN>,
                    %sa: tensor<{m}x{groups}xi8>, %sb: tensor<{groups}x{n}xi8>) -> tensor<{m}x{n}x{output}> {{
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {{
      numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
      scale_layout = {{granularity = "block", block = [1, 32], format = "e8m0"}}{transpose}
    }} : (tensor<{m}x{k}xf8E4M3FN>, tensor<{weight}xf8E4M3FN>,
          tensor<{m}x{groups}xi8>, tensor<{groups}x{n}xi8>) -> tensor<{m}x{n}x{output}>
    return %0 : tensor<{m}x{n}x{output}>
  }}
}}
'''


def lower_mxfp8(shape: BlockScaleShape, *, entry: str = "mxfp8", tessera_opt=None,
                schedule_policy: str = "auto"):
    graph = author_mxfp8_graph(shape, entry=entry, schedule_policy=schedule_policy)
    tool = tessera_opt or find_tessera_opt()
    if tool is None:
        raise RuntimeError("tessera-opt is required for native MXFP8 lowering")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    panel = schedule_blockscale_panel(schedule)
    expected_staging = "global" if schedule_policy == "seed" else "lds"
    if schedule_policy != "auto" and panel.staging != expected_staging:
        raise RuntimeError("native compiler did not honor the explicit MXFP8 schedule policy")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    return BlockScaleProgram(shape, entry, graph, schedule, tile)


def package_mxfp8(program: BlockScaleProgram, *, project_image_identity: bool = True):
    """Bind E8M0 byte scales to the verified native contract and its distinct ABI."""
    return package_blockscale(program, scale_format="e8m0",
                              project_image_identity=project_image_identity)


def compile_mxfp8(shape: BlockScaleShape, *, entry: str = "mxfp8",
                  project_image_identity: bool = True, schedule_policy: str = "auto"):
    return package_mxfp8(lower_mxfp8(shape, entry=entry, schedule_policy=schedule_policy),
                        project_image_identity=project_image_identity)
