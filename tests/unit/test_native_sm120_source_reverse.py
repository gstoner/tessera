"""Original frontend source -> paired native AD -> reverse SSA member export.

Host-free: compiler artifact and contract proof only; numerical execution is
covered separately by the owning tests/device/nvidia suite.
"""
import copy
import json

import numpy as np
import pytest
import tessera as ts

from benchmarks.nvidia.active_source_jvp_reference import product, product_reversed
from tessera.compiler.native_sm120_source_reverse import export_native_source_reverse, validate_plan

pytestmark = pytest.mark.compiler_route


def source(storage="f16", permuted=False, wrt="a", shape=(17, 19, 35)):
    import ml_dtypes
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    m, n, k = shape
    roots = (np.zeros((m, k), dtype), np.zeros((k, n), dtype))
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=(wrt,))(
        product_reversed if permuted else product)
    if permuted:
        roots = roots[::-1]
    graph = fn._specialized_autodiff_module(roots, {})
    graph.module_attrs.update({"tessera.arch": '"sm_120"', "tessera.target": '"nvidia_sm120"',
                               "tessera.native.sm120_reverse_request": "unit"})
    return graph.to_mlir(target="nvidia_sm120", canonical=True)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_source_reverse_outline_lowers_original_ad_members(storage, permuted):
    program = export_native_source_reverse(source(storage, permuted))
    plan = program.manifest
    assert plan["execution_state"] == "artifact_only"
    assert plan["role_indices"] == ([1, 0, 2] if permuted else [0, 1, 2])
    assert [b["first_write"] for b in plan["buffers"]] == [-1, -1, -1, 0, 1]
    assert plan["buffers"][3]["last_read"] == 1 and plan["outputs"] == [4]
    members = program.lower_members()
    assert len(members) == 2
    assert "transposeB = true" in members[0][0]
    assert "tile.scalar_matmul_kernel" in members[0][2]
    assert "fma" in members[0][4] and "mma.sync" not in members[0][4]
    assert "schema = 6" in members[1][2]
    assert "tile.row_region_kernel" in members[1][2]
    assert "nvvm" in members[1][4] and "tile." not in members[1][4]


def test_source_reverse_refuses_rhs_activity_in_this_profile():
    with pytest.raises(RuntimeError):
        export_native_source_reverse(source(wrt="b"))


@pytest.mark.parametrize("fault", ["seed_dtype", "scratch_lifetime", "edge", "boolean_id"])
def test_source_reverse_plan_rejects_contract_mutation(fault):
    plan = export_native_source_reverse(source()).manifest
    changed = copy.deepcopy(plan)
    if fault == "seed_dtype":
        changed["buffers"][2]["storage"] = "f16"
    elif fault == "scratch_lifetime":
        changed["buffers"][3]["last_read"] = 0
    elif fault == "edge":
        changed["steps"][1]["inputs"] = [0, 2]
    else:
        changed["buffers"][0]["id"] = False
    with pytest.raises(ValueError):
        validate_plan(json.dumps(changed))


@pytest.mark.parametrize("extra", [
    'activation = "relu"', 'transposeA = true',
    'numeric_policy = {accum = "fp32", execution_mode = "approximate"}',
])
def test_source_reverse_refuses_forward_policy_substitution(extra):
    text = source().replace('output_dtype = "fp32",', 'output_dtype = "fp32", ' + extra + ',')
    assert text != source()
    with pytest.raises(RuntimeError):
        export_native_source_reverse(text)


@pytest.mark.parametrize("edited", [False, True])
def test_source_reverse_rejects_reused_outline_instead_of_stale_plan(edited):
    program = export_native_source_reverse(source())
    text = program.outlined_ir
    if edited:
        text = text.replace("eps = 1.000000e-05", "eps = 2.000000e-05", 1)
        assert text != program.outlined_ir
    with pytest.raises(ValueError, match="exact canonical input"):
        export_native_source_reverse(text)


def test_live_and_restored_source_identity_use_the_same_canonical_witness():
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    program = export_native_source_reverse(source())
    canonical = run_tessera_opt(program.tool, source(), "--verify-each")
    assert program.source_ir.strip() == canonical.strip()
    assert program.source_ir == program.manifest["source_graph_ir"]
