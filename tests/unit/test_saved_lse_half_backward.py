"""Original half Graph AD must retain native typed checkpoint ancestry."""
import subprocess

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def source(storage="f16", *, bias=False, saved=False, output=None,
           dims=(1, 2, 1, 3, 5, 4, 3), causal=True):
    output = output or storage
    b, hq, hkv, sq, sk, d, dv = dims
    q = f"tensor<{b}x{hq}x{sq}x{d}x{storage}>"
    k = f"tensor<{b}x{hkv}x{sk}x{d}x{storage}>"
    v = f"tensor<{b}x{hkv}x{sk}x{dv}x{storage}>"
    o = f"tensor<{b}x{hq}x{sq}x{dv}x{output}>"
    lse = f"tensor<{b}x{hq}x{sq}xf32>"
    bt = f"tensor<1x{hq}x1x{sk}xf32>"
    args = f"%q: {q}, %k: {k}, %v: {v}" + (f", %bias: {bt}" if bias else "")
    types = f"{q}, {k}, {v}" + (f", {bt}" if bias else "")
    results = f"({o}, {lse})" if saved else o
    policy = ', lse_checkpoint = "saved"' if saved else ""
    return f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
      func.func @attention({args}) -> {results} attributes {{tessera.autodiff = "reverse"}} {{
        {"%o, %lse" if saved else "%o"} = tessera.flash_attn %q, %k, %v{", %bias" if bias else ""} {{head_dim = {d} : i64, causal = {str(causal).lower()}, dropout_p = 0.0 : f64{policy}, operandSegmentSizes = array<i32: 1, 1, 1, {int(bias)}>}} : ({types}) -> {results}
        return {"%o, %lse" if saved else "%o"} : {o + ", " + lse if saved else o}
      }}
    }}'''


@pytest.fixture
def compiler():
    result = find_tessera_opt()
    if result is None:
        pytest.skip("production compiler required")
    return result


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("wide_output", [False, True])
def test_original_half_reverse_graph_reaches_native_schedule_tile(compiler, storage, bias, saved, wide_output):
    original = source(storage, bias=bias, saved=saved, output="f32" if wide_output else storage)
    for role in ("forward", "backward"):
        graph = run_tessera_opt(compiler, original, "--tessera-autodiff-paired=checkpoint-product="+role)
        schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
        tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
        assert f'storage = "{storage}"' in tile
        assert 'row_lse_storage = "f32"' in tile
        assert f"x{storage}>" in graph
        if role == "backward":
            assert 'row_correction = "probability_dot_v_f32_v1"' in tile
            assert 'tile.attention_backward_kernel' in tile
            assert f'gradient_storage = "{storage}"' in tile
            assert 'lse_cotangent = true' in tile if saved else 'lse_cotangent = true' not in tile


@pytest.mark.parametrize("target,arch", [("rocm", "gfx1201"), ("apple_gpu", "apple7"), ("x86", "zen5-avx512")])
def test_half_reverse_keeps_sibling_physical_gate(compiler, target, arch):
    text = source().replace('"nvidia_sm120"', f'"{target}"').replace('"sm_120"', f'"{arch}"')
    result = subprocess.run([str(compiler), "--tessera-autodiff-paired"], input=text,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


def test_half_tangent_not_admitted_by_reverse_storage_change(compiler):
    text = source(saved=True).replace('"reverse"', '"forward"')
    result = subprocess.run([str(compiler), "--tessera-autodiff-forward"], input=text,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


@pytest.mark.parametrize("mutation", ["missing", "unknown", "output_storage"])
def test_half_row_correction_policy_is_verified_before_target_lowering(compiler, mutation):
    graph = run_tessera_opt(compiler, source(saved=True, bias=True),
                           "--tessera-autodiff-paired=checkpoint-product=backward")
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    # Mutate only the launch-level op, preserving the sealed contract witness.
    lines = tile.splitlines()
    index = next(i for i, line in enumerate(lines) if "tile.attention_backward_kernel" in line)
    if mutation == "missing":
        lines[index] = lines[index].replace(', row_correction = "probability_dot_v_f32_v1"', "")
    elif mutation == "unknown":
        lines[index] = lines[index].replace('row_correction = "probability_dot_v_f32_v1"', 'row_correction = "rounded_output"')
    else:
        lines[index] = lines[index].replace('output_storage = "f16"', 'output_storage = "bf16"')
    result = subprocess.run([str(compiler)], input="\n".join(lines),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "row correction" in result.stderr


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("saved", [False, True])
def test_half_reverse_projection_retains_bounded_and_compact_native_roles(compiler, storage, saved):
    original = source(storage, saved=saved, bias=True).replace(
        'tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 7, 11>')
    original = original.replace('tessera.autodiff = "reverse"',
                                'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [0, 3]')
    graph = run_tessera_opt(compiler, original,
        "--tessera-autodiff-paired=checkpoint-product=backward prune-checkpoint-gradients=true compact-checkpoint-gradients=true")
    schedule = run_tessera_opt(compiler, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    assert 'shape_policy = "bounded_sequences_v1"' in tile
    assert 'gradient_output = "compact_v1"' in tile
    assert 'row_correction = "probability_dot_v_f32_v1"' in tile
    assert f'storage = "{storage}"' in tile
