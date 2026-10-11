"""Production MLIR attention result storage and schedule seal checks."""
import os
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import tessera as ts


def attention(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


def run_tool(tool, source, *options):
    result = subprocess.run([str(tool), *options], input=source, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout


@pytest.fixture
def tools():
    core = Path(os.environ.get("TESSERA_OPT", "missing"))
    backend = Path(os.environ.get("TESSERA_NVIDIA_OPT", "missing"))
    if not core.is_file() or not backend.is_file():
        pytest.skip("requires production core/NVIDIA MLIR tools")
    return core, backend


@pytest.mark.parametrize("dtype,storage", [("float16", "f16"), ("bfloat16", "bf16")])
def test_native_half_result_survives_graph_schedule_tile_and_target(tools, dtype, storage):
    import ml_dtypes
    actual_dtype = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
    values = tuple(np.zeros(shape, actual_dtype) for shape in ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    fn = ts.jit(target="nvidia_sm120")(attention)
    module, _ = fn._trace_frontend_capture(values, {})
    op = module.functions[0].body[0]
    op.kwargs.update(scale=.5, causal=True, window_left=-1, window_right=-1,
                     softcap=0., dropout_p=0., dropout_seed=0)
    module.module_attrs["tessera.target"] = '"nvidia_sm120"'
    module.module_attrs["tessera.arch"] = '"sm_120"'
    module.module_attrs["tessera.launch_bindings"] = json.dumps(
        [arg.name for arg in module.functions[0].args] + [op.result])
    graph = module.to_mlir(target="nvidia_sm120", canonical=True)
    core, backend = tools
    semantic = run_tool(core, graph, "--tessera-tile-ir-lowering=tile-q=64 tile-kv=64 sm=90")
    assert "tessera_attn.lse_accumulate" in semantic
    assert "arith.truncf" in semantic and f"to tensor<3x3x{storage}>" in semantic
    assert semantic.index("tessera_attn.lse_accumulate") < semantic.index("arith.truncf")
    schedule = run_tool(core, graph, "--tessera-graph-to-schedule")
    tile = run_tool(core, schedule, "--tessera-schedule-to-tile")
    target = run_tool(backend, tile, "--tessera-lower-to-nvidia-sm120")
    assert f'output_storage = "{storage}"' in schedule
    assert f'output_storage = "{storage}"' in tile
    assert f"to {storage}" in target and "arith.truncf" in target
    tampered = schedule.replace(f'output_storage = "{storage}"', 'output_storage = "f32"')
    with pytest.raises(RuntimeError, match="altered"):
        run_tool(core, tampered, "--tessera-schedule-to-tile")
    other = "bf16" if storage == "f16" else "f16"
    tampered = tile.replace(f'output_storage = "{storage}"', f'output_storage = "{other}"')
    with pytest.raises(RuntimeError, match="matching input storage"):
        run_tool(backend, tampered, "--tessera-lower-to-nvidia-sm120")
