"""Native SSA partitioning of a normalization producer into saved-LSE attention.

Host-free: compiler member/call/lifetime artifacts only. Device numerical and
ownership proof belongs to a separately gated owning-SM120 suite.
"""
import base64
import json
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

pytestmark = pytest.mark.compiler_route


def prefix_attention(q, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def prefix_attention_permuted(v, q, k):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def prefix_attention_swapped_outputs(q, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    o, lse = ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")
    return lse, o


def source(storage="f16", permuted=False, target="nvidia_sm120", arch="sm_120",
           swapped_outputs=False):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    roots = (np.zeros((1, 2, 3, 4), dtype), np.zeros((1, 1, 5, 4), dtype),
             np.zeros((1, 1, 5, 3), dtype))
    function = (prefix_attention_permuted if permuted else
                prefix_attention_swapped_outputs if swapped_outputs else prefix_attention)
    fn = ts.jit(target=target)(function)
    graph = fn._traced_autodiff_module((roots[2], roots[0], roots[1]) if permuted else roots, {})
    graph.module_attrs.update({"tessera.target": '"' + target + '"', "tessera.arch": '"' + arch + '"'})
    return graph.to_mlir(target=target, canonical=True)


def run(tool, text, flag):
    result = subprocess.run([str(tool), flag, "-"], input=text, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_native_source_outline_preserves_attention_edges_and_output_roles(
        compiler_toolchain, storage, permuted):
    import re
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    outlined = run(tool, source(storage, permuted), "--tessera-outline-source-program")
    encoded = re.findall(r'tessera.native.source_program_json = "([^"]+)"', outlined)
    assert len(encoded) == 1
    plan = json.loads(base64.b64decode(encoded[0], validate=True))
    assert plan["schema"] == "tessera.native.source_program.v1"
    assert plan["kind"] == "norm_attention_saved"
    assert plan["execution_state"] == "artifact_only"
    roles = [1, 2, 0] if permuted else [0, 1, 2]
    assert plan["role_indices"] == roles
    assert [step["inputs"] for step in plan["steps"]] == [[roles[0]], [3, roles[1], roles[2]]]
    assert [step["outputs"] for step in plan["steps"]] == [[3], [4, 5]]
    assert plan["outputs"] == [4, 5]
    assert plan["buffers"][3]["ownership"] == "private_scratch"
    assert plan["buffers"][3]["first_write"] == 0 and plan["buffers"][3]["last_read"] == 1
    assert [b["storage"] for b in plan["buffers"][3:]] == [storage, storage, "f32"]
    assert plan["buffers"][4]["shape"] == [1, 2, 3, 3]
    assert plan["buffers"][5]["shape"] == [1, 2, 3]
    assert outlined.count("call @") == 2
    assert len(plan["member_graphs"]) == 2
    assert 'accum = "fp32"' in plan["member_graphs"][0]
    assert 'lse_checkpoint = "saved"' in plan["member_graphs"][1]
    assert outlined.count("tessera.rmsnorm") == 1
    assert outlined.count("tessera.flash_attn") == 1
    for index, graph in enumerate(plan["member_graphs"]):
        schedule = run(tool, graph, "--tessera-graph-to-schedule")
        tile = run(tool, schedule, "--tessera-schedule-to-tile")
        target = run(tool, tile, "--lower-tile-to-nvidia=sm=120")
        native = run(tool, target, "--lower-tessera-nvidia-to-nvvm")
        assert "schedule." in schedule and "tile." in tile
        assert ("tile.norm_kernel" if index == 0 else "tile.attention_kernel") in tile
        if index == 1:
            assert 'lse_checkpoint = "saved"' in tile
            assert 'output_storage = "' + storage + '"' in tile
        assert "tessera.rmsnorm" not in native and "tessera.flash_attn" not in native


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_native_source_program_preserves_swapped_o_lse_results(compiler_toolchain, storage):
    import re
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    text = source(storage, swapped_outputs=True)
    outlined = run(tool, text, "--tessera-source-program-outline")
    match = re.search(r'tessera.native.source_program_json = "([^"]+)"', outlined)
    assert match is not None
    plan = json.loads(base64.b64decode(match[1], validate=True))
    assert plan["outputs"] == [5, 4]
    assert plan["steps"][1]["outputs"] == [4, 5]
    assert plan["buffers"][4]["storage"] == storage
    assert plan["buffers"][5]["storage"] == "f32"


@pytest.mark.parametrize("target,arch", [
    ("rocm_gfx1151", "gfx1151"), ("rocm_gfx1201", "gfx1201"),
    ("apple_gpu", "apple7"), ("x86_avx512", "x86_avx512"),
])
def test_native_source_outline_refuses_unowned_target(compiler_toolchain, target, arch):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    text = source().replace('tessera.target = "nvidia_sm120"', 'tessera.target = "' + target + '"')
    text = text.replace('tessera.arch = "sm_120"', 'tessera.arch = "' + arch + '"')
    result = subprocess.run([str(tool), "--tessera-outline-source-program", "-"],
                            input=text, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert "native source program requires its owning" in result.stderr


def test_native_source_outline_refuses_second_outline(compiler_toolchain):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    outlined = run(tool, source(), "--tessera-outline-source-program")
    result = subprocess.run([str(tool), "--tessera-outline-source-program", "-"],
                            input=outlined, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert "native source program requires its owning" in result.stderr


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_native_source_transport_lowers_retained_members(compiler_toolchain, storage):
    from tessera.compiler.native_source_program import outline_source_program
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    text = source(storage, permuted=True, swapped_outputs=False)
    program = outline_source_program(text, tool=tool)
    members = program.scheduled_members()
    assert len(members) == 2
    assert program.source_ir == text
    assert program.manifest["role_indices"] == [1, 2, 0]
    assert "tile.norm_kernel" in members[0].tile_ir
    assert "tile.attention_kernel" in members[1].tile_ir
    norm, attention = program.scheduled_artifacts()
    assert norm.rows == 6 and norm.columns == 4
    assert norm.storage == storage
    assert attention.storage == storage and attention.output_storage == storage
    assert attention.names == ("input_0", "input_1", "input_2", "output_0", "output_1")
    assert attention.dims == (1, 2, 1, 3, 5, 4, 3)


@pytest.mark.parametrize("mutation", ["private_lifetime", "root_roles", "output_order", "source_policy"])
def test_native_source_transport_rejects_changed_contract(compiler_toolchain, mutation):
    from dataclasses import replace
    import re
    from tessera.compiler.native_source_program import outline_source_program
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    program = outline_source_program(source(), tool=tool)
    plan = program.manifest
    if mutation == "private_lifetime":
        plan["buffers"][3]["last_read"] = 0
    elif mutation == "root_roles":
        plan["role_indices"] = [1, 0, 2]
    elif mutation == "output_order":
        plan["outputs"] = [5, 4]
    else:
        plan["member_graphs"][0] = plan["member_graphs"][0].replace(
            'accum = "fp32"', 'accum = "fp16"')
    encoded = base64.b64encode(json.dumps(plan).encode()).decode()
    changed = re.sub(r'(tessera.native.source_program_json = ")[^"]+',
                     lambda match: match[1] + encoded, program.outlined_ir)
    with pytest.raises(ValueError, match="retained source replay"):
        replace(program, outlined_ir=changed).scheduled_members()


@pytest.mark.parametrize("mutation", ["private_lifetime", "private_shape", "output_storage",
                                     "role_alias", "source_digest", "output_alias"])
def test_portable_plan_rejects_invalid_ownership_and_source_binding(compiler_toolchain, mutation):
    from tessera.compiler.native_source_program import outline_source_program
    from tessera.compiler.portable_source_attention import validate_plan
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    plan = outline_source_program(source(), tool=tool).manifest
    validate_plan(plan)
    if mutation == "private_lifetime":
        plan["buffers"][3]["last_read"] = 0
    elif mutation == "private_shape":
        plan["buffers"][3]["shape"][-1] += 1
    elif mutation == "output_storage":
        plan["buffers"][5]["storage"] = "f16"
    elif mutation == "role_alias":
        plan["role_indices"][1] = plan["role_indices"][0]
    elif mutation == "source_digest":
        plan["source_graph_ir"] += "\n"
    else:
        plan["outputs"] = [4, 4]
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("location", ["module", "function"])
def test_primal_outline_does_not_drop_pending_autodiff_request(compiler_toolchain, mode, location):
    tool = compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    text = source()
    if location == "module":
        text = text.replace("module attributes {",
                            'module attributes {tessera.autodiff = "' + mode + '", ', 1)
    else:
        import re
        text, count = re.subn(
            r'(func.func @[^\n]+? attributes \{)',
            lambda match: match[1] + 'tessera.autodiff = "' + mode + '", ', text, count=1)
        assert count == 1
    result = subprocess.run([str(tool), "--tessera-outline-source-program", "-"],
                            input=text, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0, "primal outline silently dropped a pending AD request"
    assert "native source program requires its owning" in result.stderr


def prefix_attention_keyword_only(q, *, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def prefix_attention_all_keyword_only(*, q, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def prefix_attention_positional_only(q, /, *, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("convention", ["keyword_only", "all_keyword_only", "positional_only"])
def test_source_attention_capture_preserves_python_parameter_kinds(storage, convention):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    q, k, v = (np.zeros(shape, dtype) for shape in
               ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    function = {"keyword_only": prefix_attention_keyword_only,
                "all_keyword_only": prefix_attention_all_keyword_only,
                "positional_only": prefix_attention_positional_only}[convention]
    args = () if convention == "all_keyword_only" else (q,)
    kwargs = {"q": q, "k": k, "v": v} if not args else {"k": k, "v": v}
    fn = ts.jit(target="nvidia_sm120")(function)
    graph = fn._trace_frontend_capture(args, kwargs)[0]
    assert [op.op_name for op in graph.functions[0].body] == [
        "tessera.rmsnorm", "tessera.flash_attn"]
    assert [tuple(map(int, arg.ir_type.shape)) for arg in graph.functions[0].args] == [
        q.shape, k.shape, v.shape]
    assert len(graph.functions[0].result_types) == 2
