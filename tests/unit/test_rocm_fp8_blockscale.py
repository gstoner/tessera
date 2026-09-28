"""Host-free contract tests for gfx1201 block-scaled FP8 W8A8 (ROCM-FP8-BLOCKSCALE-1).

The device rows live in tests/device/rocm/test_fp8_blockscale_w8a8.py; these
check the pieces that must hold on any host: the logical Graph op the author
emits, the Target-IR check that refuses every drifted semantic field, the
fp64 oracle itself, and the pipeline knob's validation.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from tessera.compiler.rocm_fp8_blockscale import (
    FP8_W8A8_BLOCKSCALE_CONTRACT,
    FP8_W8A8_BLOCKSCALE_NK_CONTRACT,
    GFX_FP8_W8A8_BLOCKSCALE_ABI,
    GFX_FP8_W8A8_BLOCKSCALE_BF16_ABI,
    GFX_FP8_W8A8_BLOCKSCALE_NK_ABI,
    GFX_FP8_W8A8_BLOCKSCALE_NK_BF16_ABI,
    PACKAGE_ABIS,
    BlockScaleShape,
    author_blockscale_graph,
    blockscale_reference,
    check_blockscale_target_ir,
    lower_blockscale,
)
from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline
from tessera.compiler.scheduled_matmul import find_tessera_opt


def test_shape_refuses_partial_groups_and_unaligned_groups():
    with pytest.raises(ValueError, match="whole number of scale groups"):
        BlockScaleShape(32, 64, 192, 128, 128)
    with pytest.raises(ValueError, match="16-wide"):
        BlockScaleShape(32, 64, 192, 24, 128)
    with pytest.raises(ValueError, match="weight_layout"):
        BlockScaleShape(32, 64, 256, 128, 128, "tn")
    with pytest.raises(ValueError, match="output"):
        BlockScaleShape(32, 64, 256, 128, 128, "nk", "f16")
    shape = BlockScaleShape(40, 200, 256, 128, 128)
    assert (shape.groups, shape.n_groups) == (2, 2)


def test_author_states_the_logical_op_and_never_the_contract():
    kn = author_blockscale_graph(BlockScaleShape(40, 72, 256, 128, 128))
    nk = author_blockscale_graph(BlockScaleShape(40, 72, 256, 128, 128, "nk"))
    for text in (kn, nk):
        assert "tessera.scaled_matmul" in text
        # The physical contract is derived by Graph->Schedule, never authored.
        assert "physical_contract" not in text
        assert 'block = [128, 128], format = "fp32"' in text
        assert "tensor<40x2xf32>" in text and "tensor<2x1xf32>" in text
    assert "tensor<256x72xf8E4M3FN>" in kn and "transposeB" not in kn
    assert "tensor<72x256xf8E4M3FN>" in nk and "transposeB = true" in nk
    assert "-> tensor<40x72xf32>" in kn


def test_bf16_output_is_the_graph_result_type_and_its_own_abi():
    text = author_blockscale_graph(BlockScaleShape(40, 72, 256, 128, 128, "nk", "bf16"))
    # The accumulator stays fp32 (numeric_policy); only the result storage narrows.
    assert "-> tensor<40x72xbf16>" in text and 'accum = "fp32"' in text
    assert PACKAGE_ABIS == {
        ("kn", "f32"): GFX_FP8_W8A8_BLOCKSCALE_ABI,
        ("nk", "f32"): GFX_FP8_W8A8_BLOCKSCALE_NK_ABI,
        ("kn", "bf16"): GFX_FP8_W8A8_BLOCKSCALE_BF16_ABI,
        ("nk", "bf16"): GFX_FP8_W8A8_BLOCKSCALE_NK_BF16_ABI,
    }
    assert len(set(PACKAGE_ABIS.values())) == 4


def _pair(layout: str = "kn", output: str = "f32", *, staging: str = "global",
          warps: int = 1, depth: int = 1, block: tuple[int, int] = (32, 32)) -> tuple[str, str]:
    contract, pointer = (
        (FP8_W8A8_BLOCKSCALE_CONTRACT, "a_b_lhs_scale_rhs_scale_d_m_n_k")
        if layout == "kn" else
        (FP8_W8A8_BLOCKSCALE_NK_CONTRACT, "a_bnk_lhs_scale_rhs_scale_d_m_n_k"))
    abi = PACKAGE_ABIS[(layout, output)]
    tile = (f'tile.scaled_matmul_kernel %a {{partial_accumulator = {{combine = "scale_outer_product_then_add", '
            f'cross_step_motion = "forbid", init = "zero", instruction_steps = 8 : i64, '
            f'schedule_scope = "scale_group", scope = "scale_group"}}, physical_contract = "{contract}", '
            f'tessera.scale_block_n = 128 : i64, tessera.schedule_hash = "h0"}}')
    target = (f'tessera_rocm.scaled_wmma_gemm {{abi = "{pointer}", block_m = {block[0]} : i64, '
              f'block_n = {block[1]} : i64, '
              f'instruction_k = 16 : i64, k = 256 : i64, k_step_schedule = "isolated_scale_group", '
              f'm = 64 : i64, macro_k = 128 : i64, n = 96 : i64, name = "w", numeric_policy = '
              f'{{accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}}, output = "{output}", '
              f'package_abi = "{abi}", partial_combine = "scale_outer_product_then_add", '
              f'physical_contract = "{contract}", pipeline_depth = {depth} : i64, scale_format = "fp32", '
              f'scale_k = 128 : i64, scale_n = 128 : i64, staging = "{staging}", '
              f'tessera.schedule_hash = "h0", warps = {warps} : i64}}')
    return tile, target


@pytest.mark.parametrize("layout", ["kn", "nk"])
def test_target_check_accepts_the_bound_directive(layout):
    tile, target = _pair(layout)
    checked = check_blockscale_target_ir(BlockScaleShape(64, 96, 256, 128, 128, layout), tile, target)
    assert (checked["block_m"], checked["block_n"], checked["macro_k"]) == (32, 32, 128)
    assert (checked["staging"], checked["warps"], checked["pipeline_depth"]) == ("global", 1, 1)


def test_target_check_binds_the_lds_body_and_the_bf16_output():
    tile, target = _pair("nk", "bf16", staging="lds", warps=8, block=(128, 128))
    checked = check_blockscale_target_ir(
        BlockScaleShape(64, 96, 256, 128, 128, "nk", "bf16"), tile, target)
    assert (checked["staging"], checked["warps"], checked["block_m"]) == ("lds", 8, 128)


@pytest.mark.parametrize(("layout", "staging", "warps", "depth", "block"), [
    ("kn", "lds", 8, 1, (128, 128)),     # the LDS body reads the [N, K] weight only
    ("nk", "lds", 6, 1, (128, 128)),     # 4 wave rows of 32 do not divide 6 waves
    ("nk", "lds", 32, 1, (128, 128)),    # more than 16 waves
    ("nk", "global", 8, 1, (32, 32)),    # the register panel is one wave
    ("nk", "lds", 8, 3, (128, 128)),     # single- or double-buffered only
    ("nk", "smem", 8, 1, (128, 128)),    # an unknown staging
])
def test_target_check_refuses_an_inconsistent_physical_schedule(layout, staging, warps, depth, block):
    tile, target = _pair(layout, staging=staging, warps=warps, depth=depth, block=block)
    with pytest.raises(ValueError, match="staging|warps|LDS|pipeline_depth|one wave"):
        check_blockscale_target_ir(BlockScaleShape(64, 96, 256, 128, 128, layout), tile, target)


@pytest.mark.parametrize(("old", "new"), [
    ('scale_format = "fp32"', 'scale_format = "e8m0"'),
    ('execution_mode = "exact_per_block"', 'execution_mode = "folded_row_reference_explicit_approximate"'),
    ("scale_k = 128 : i64", "scale_k = 64 : i64"),
    ("scale_n = 128 : i64", "scale_n = 1 : i64"),
    ('output = "f32"', 'output = "bf16"'),
    ('partial_combine = "scale_outer_product_then_add"', 'partial_combine = "row_reference_after_full_k"'),
    ("macro_k = 128 : i64", "macro_k = 96 : i64"),
    ("block_m = 32 : i64", "block_m = 24 : i64"),
    ('storage = "e4m3"', 'storage = "e5m2"'),
    ('tessera.schedule_hash = "h0"', 'tessera.schedule_hash = "h1"'),
    ("m = 64 : i64", "m = 65 : i64"),
])
def test_target_check_refuses_every_drifted_semantic_field(old, new):
    tile, target = _pair()
    assert old in target
    with pytest.raises(ValueError):
        check_blockscale_target_ir(BlockScaleShape(64, 96, 256, 128, 128), tile, target.replace(old, new, 1))


def test_target_check_refuses_the_other_layouts_directive():
    tile, target = _pair("nk")
    with pytest.raises(ValueError, match="abi|physical_contract|package_abi"):
        check_blockscale_target_ir(BlockScaleShape(64, 96, 256, 128, 128, "kn"), tile, target)


def test_target_check_refuses_an_unbound_logical_directive():
    tile, target = _pair()
    unbound = re.sub(r'package_abi = "[^"]+"', 'package_abi = "unbound"', target)
    with pytest.raises(ValueError, match="package_abi"):
        check_blockscale_target_ir(BlockScaleShape(64, 96, 256, 128, 128), tile, unbound)


def test_target_check_refuses_a_tile_carrier_that_dropped_isolation():
    tile, target = _pair()
    with pytest.raises(ValueError, match="init"):
        check_blockscale_target_ir(BlockScaleShape(64, 96, 256, 128, 128),
                                   tile.replace('init = "zero"', 'init = "carry"'), target)


def test_oracle_is_per_group_outer_product_scaling():
    rng = np.random.default_rng(7)
    m, n, k, sk, sn = 5, 40, 64, 32, 16
    a = rng.integers(-3, 4, (m, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, n)).astype(np.float32)
    sa = rng.uniform(0.25, 4.0, (m, k // sk)).astype(np.float32)
    sb = rng.uniform(0.25, 4.0, (k // sk, (n + sn - 1) // sn)).astype(np.float32)
    got = blockscale_reference(a, b, sa, sb, scale_k=sk, scale_n=sn)
    # Independent spelling: dequantize both operands blockwise, then multiply.
    a_dq = a.astype(np.float64) * np.repeat(sa, sk, axis=1)
    b_dq = b.astype(np.float64) * np.repeat(np.repeat(sb, sk, axis=0), sn, axis=1)[:, :n]
    np.testing.assert_allclose(got, a_dq @ b_dq, rtol=1e-12)
    # And it is not a single rescale of the full-K product.
    single = (a.astype(np.float64) @ b) * sa[:, :1] * sb[0, np.arange(n) // sn]
    assert np.abs(single - got).max() > 1.0


def test_oracle_refuses_mismatched_scale_extents():
    a = np.zeros((4, 64), np.float32)
    b = np.zeros((64, 8), np.float32)
    with pytest.raises(ValueError):
        blockscale_reference(a, b, np.zeros((4, 3), np.float32), np.zeros((2, 1), np.float32),
                             scale_k=32, scale_n=8)


def test_pipeline_knob_is_validated_and_reaches_the_pass_string():
    base = ROCMExecutablePipeline(family="matmul", arch="gfx1201")
    assert "scale-group-panels=-1" in base.pass_pipeline()
    tuned = ROCMExecutablePipeline(family="matmul", arch="gfx1201", scale_group_panels=4)
    assert "scale-group-panels=4" in tuned.pass_pipeline()
    assert tuned.cache_key() != base.cache_key()
    for bad in (-2, 17, 1.5, True):
        with pytest.raises(ValueError, match="scale_group_panels"):
            ROCMExecutablePipeline(family="matmul", arch="gfx1201", scale_group_panels=bad)


def test_runtime_admits_every_abi_as_proved_gfx1201_launches():
    from tessera import runtime as rt

    proved = rt._gfx1201_proved_scheduled_abis()
    assert set(PACKAGE_ABIS.values()) <= proved


def test_lds_body_knobs_are_validated_and_reach_the_pass_string():
    base = ROCMExecutablePipeline(family="matmul", arch="gfx1201")
    text = base.pass_pipeline()
    for knob in ("blockscale-stage-k=-1", "blockscale-lds-pad-bytes=-1",
                 "blockscale-prefetch=-1"):
        assert knob in text
    for field, bad in (("blockscale_stage_k", 1024), ("blockscale_lds_pad_bytes", 65),
                       ("blockscale_prefetch", 3), ("blockscale_prefetch", True)):
        with pytest.raises(ValueError, match=field):
            ROCMExecutablePipeline(family="matmul", arch="gfx1201", **{field: bad})


@pytest.mark.parametrize(("layout", "contract"), [
    ("kn", FP8_W8A8_BLOCKSCALE_CONTRACT), ("nk", FP8_W8A8_BLOCKSCALE_NK_CONTRACT)])
def test_graph_to_tile_derives_the_named_contract(layout, contract):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("tessera-opt is not built on this host")
    program = lower_blockscale(BlockScaleShape(64, 96, 256, 128, 128, layout), tessera_opt=tool)
    carrier = next(line for line in program.tile_ir.splitlines() if "tile.scaled_matmul_kernel" in line)
    assert f'physical_contract = "{contract}"' in carrier
    assert "tessera.scale_block_n = 128" in carrier
    assert 'scale_fmt = "fp32"' in carrier and "scale_k = 128" in carrier
    assert 'execution_mode = "exact_per_block"' in carrier
    # 64x96 is whole 32s but only 6 tiles at 32x32 -- under the measured
    # 256-tile floor -- so the half-height panel doubles the grid.
    assert "tessera.macro_tile_m = 16" in carrier and "tessera.macro_tile_n = 32" in carrier


@pytest.mark.parametrize(("m", "n", "panel"), [
    (512, 512, (32, 32)),    # 256 tiles at 32x32: the full panel
    (256, 1024, (32, 32)),   # 32 workgroups even at 128x64: too few for the LDS body
    (32, 4096, (16, 32)),    # 128 tiles: half height doubles the grid (M < 128)
    (48, 6144, (16, 32)),    # ragged M under 32 rows would run the edge path
    (512, 520, (16, 16)),    # N not a whole 32 takes 16 columns
])
def test_w8a8_register_panel_rule(m, n, panel):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("tessera-opt is not built on this host")
    program = lower_blockscale(BlockScaleShape(m, n, 256, 128, 128, "nk"), tessera_opt=tool)
    carrier = next(line for line in program.tile_ir.splitlines() if "tile.scaled_matmul_kernel" in line)
    assert f"tessera.macro_tile_m = {panel[0]}" in carrier
    assert f"tessera.macro_tile_n = {panel[1]}" in carrier
    assert 'staging = "global"' in carrier and "warps = 1" in carrier


@pytest.mark.parametrize(("m", "n", "layout", "macro"), [
    (1024, 4096, "nk", (128, 128)),   # 256 workgroups at 128x128
    (256, 4096, "nk", (128, 128)),    # exactly 64 at 128x128
    (128, 4096, "nk", (128, 64)),     # 32 at 128x128, 64 at 128x64
    (1000, 2048, "nk", (128, 128)),   # ragged M follows the whole-M rule (8 x 16)
    (1024, 2048, "nk", (128, 128)),   # the whole-block neighbour
    (300, 2048, "nk", (128, 64)),     # 3 x 16 = 48 at 128x128: 128x64 covers the CUs
    (1024, 4096, "kn", None),         # [K, N] keeps the register panel
    (64, 8192, "nk", None),           # below one 128-row block
])
def test_w8a8_lds_body_rule(m, n, layout, macro):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("tessera-opt is not built on this host")
    program = lower_blockscale(BlockScaleShape(m, n, 256, 128, 128, layout), tessera_opt=tool)
    carrier = next(line for line in program.tile_ir.splitlines() if "tile.scaled_matmul_kernel" in line)
    scheduled = next(line for line in program.schedule_ir.splitlines() if "schedule.matmul" in line)
    if macro is None:
        assert 'staging = "global"' in carrier and "warps = 1" in carrier
        assert "staging" not in scheduled
        return
    assert 'staging = "lds"' in carrier and "warps = 8" in carrier
    assert "tessera.pipeline_depth = 1" in carrier
    assert f"tessera.macro_tile_m = {macro[0]}" in carrier
    assert f"tessera.macro_tile_n = {macro[1]}" in carrier
    # Schedule IR states it too, and the digest moves with it.
    assert 'staging = "lds"' in scheduled


# ── FOUNDATION-BATCH-2-2026-09-27: the CU count has one authority ──────────


def _cpp_compute_units() -> dict[str, int]:
    """The `measuredComputeUnits` table as PMPasses.cpp states it."""
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2]
              / "src/compiler/programming_model/lib/PMPasses.cpp").read_text()
    body = source.split("static std::optional<int64_t> measuredComputeUnits(", 1)[1]
    body = body.split("\n}\n", 1)[0]
    table = dict(re.findall(r'arch == "(gfx\d+)"\)\s*return (\d+);', body))
    assert table, "measuredComputeUnits has no entries (or its shape changed)"
    return {arch: int(units) for arch, units in table.items()}


def test_cpp_compute_units_mirror_rocm_target():
    """The C++ rule's CU denominator is `rocm_target.compute_units`, entry for
    entry: every C++ entry equals the Python authority, and every part the
    Python table has measured is present in C++ (a missing one would silently
    keep the register panel there)."""
    from tessera.compiler.rocm_target import AMDArch, compute_units
    cpp = _cpp_compute_units()
    python = {f"gfx{arch.value}": compute_units(arch) for arch in AMDArch
              if compute_units(arch) is not None}
    assert cpp == python
    assert cpp["gfx1201"] == 64


def test_compute_units_is_twice_the_measured_wgps_and_none_when_unmeasured():
    from tessera.compiler.rocm_target import (
        AMDArch, WorkgroupProcessorMode, compute_units, dispatch_slots)
    assert compute_units(AMDArch.GFX_1201) == 2 * dispatch_slots(
        AMDArch.GFX_1201, WorkgroupProcessorMode.WGP)
    assert compute_units(AMDArch.GFX_1151) == 40
    assert compute_units(AMDArch.GFX_90A) is None


@pytest.mark.parametrize(("m", "n", "layout", "expected"), [
    (1024, 4096, "nk", ("lds", 128, 128, 8)),
    (256, 4096, "nk", ("lds", 128, 128, 8)),
    (128, 4096, "nk", ("lds", 128, 64, 8)),
    (1024, 1024, "nk", ("lds", 128, 128, 8)),   # 8 x 8 = 64 workgroups
    (1000, 24576, "nk", ("lds", 128, 128, 8)),  # ragged M: the whole-M rule
    (200, 2048, "nk", ("lds", 128, 64, 8)),     # 2 x 16 = 32 at 128x128
    (1024, 4096, "kn", ("global", 32, 32, 1)),
    (64, 8192, "nk", ("global", 32, 32, 1)),
    (256, 1024, "nk", ("global", 32, 32, 1)),   # 32 workgroups even at 128x64
    (48, 6144, "nk", ("global", 16, 32, 1)),
    (512, 520, "nk", ("global", 16, 16, 1)),
])
def test_panel_oracle_states_the_rule(m, n, layout, expected):
    from tessera.compiler.rocm_fp8_blockscale import BlockScalePanel, blockscale_panel_oracle
    assert blockscale_panel_oracle(BlockScaleShape(m, n, 256, 128, 128, layout)) == \
        BlockScalePanel(*expected)


@pytest.mark.parametrize(("m", "n", "k", "expected"), [
    (96, 8192, 1024, ("global", 32, 32, 1)),
    (97, 1024, 1024, ("lds", 128, 64, 8)),
    (100, 2048, 4096, ("lds", 128, 64, 8)),
    (127, 24576, 1536, ("lds", 128, 64, 8)),
    (128, 8192, 1024, ("lds", 128, 128, 8)),
    (100, 2048, 512, ("global", 16, 32, 1)),
])
def test_gfx1201_sub128_ragged_panel_matches_native_schedule(m, n, k, expected):
    from tessera.compiler.rocm_fp8_blockscale import (
        BlockScalePanel, blockscale_panel_oracle, verify_blockscale_schedule)

    shape = BlockScaleShape(m, n, k, 128, 128, "nk")
    assert blockscale_panel_oracle(shape) == BlockScalePanel(*expected)
    verify_blockscale_schedule(shape, _schedule_ir(shape))


@pytest.mark.parametrize(("m", "n", "k", "expected"), [
    (192, 8192, 1024, ("lds", 128, 64, 8)),
    (200, 8192, 1024, ("lds", 128, 64, 8)),
    (200, 8320, 1024, ("lds", 128, 64, 8)),
    (255, 10240, 1024, ("lds", 128, 64, 8)),
    (200, 8193, 1024, ("lds", 128, 128, 8)),
    (200, 6144, 1024, ("lds", 128, 128, 8)),
    (200, 8192, 1536, ("lds", 128, 128, 8)),
    (200, 8192, 2048, ("lds", 128, 128, 8)),
    (256, 8192, 1024, ("lds", 128, 128, 8)),
    (300, 8192, 1024, ("lds", 128, 128, 8)),
])
def test_gfx1201_short_k_m200_panel_matches_native_schedule(m, n, k, expected):
    from tessera.compiler.rocm_fp8_blockscale import (
        BlockScalePanel, blockscale_panel_oracle, verify_blockscale_schedule)

    shape = BlockScaleShape(m, n, k, 128, 128, "nk")
    assert blockscale_panel_oracle(shape) == BlockScalePanel(*expected)
    verify_blockscale_schedule(shape, _schedule_ir(shape))


def test_panel_oracle_keeps_the_register_panel_for_an_unmeasured_arch():
    """No measured CU count, no occupancy verdict: the LDS body is not offered
    (the C++ rule warns ROCM_FP8_BLOCKSCALE_LDS_NOT_APPLIED in that case)."""
    from tessera.compiler.rocm_fp8_blockscale import blockscale_panel_oracle
    shape = BlockScaleShape(1024, 4096, 256, 128, 128, "nk")
    assert blockscale_panel_oracle(shape, arch="gfx1250").staging == "global"
    assert blockscale_panel_oracle(shape, arch="gfx9999").staging == "global"
    ragged = BlockScaleShape(100, 2048, 1024, 128, 128, "nk")
    assert blockscale_panel_oracle(ragged, arch="gfx1151").staging == "global"


def _schedule_ir(shape: BlockScaleShape) -> str:
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("tessera-opt is not built on this host")
    return run_tessera_opt(tool, author_blockscale_graph(shape), "--tessera-graph-to-schedule")


@pytest.mark.parametrize("m", [64, 128, 200, 256, 300, 512, 1000, 1024, 1500, 2048])
@pytest.mark.parametrize("n", [1024, 2048, 3072, 4096, 24576])
def test_native_schedule_and_panel_oracle_agree(m, n):
    """The differential half of Decision #31: the C++ Schedule is the
    authority, and the Python oracle reproduces it on a grid spanning every
    branch of the rule (both sides of the CU threshold, ragged and whole M)."""
    from tessera.compiler.rocm_fp8_blockscale import verify_blockscale_schedule
    for layout in ("nk", "kn"):
        shape = BlockScaleShape(m, n, 256, 128, 128, layout)
        verify_blockscale_schedule(shape, _schedule_ir(shape))


def test_projection_refuses_when_the_oracle_and_the_schedule_diverge(monkeypatch):
    from tessera.compiler import rocm_fp8_blockscale as w8a8
    shape = BlockScaleShape(1024, 4096, 256, 128, 128, "nk")
    if find_tessera_opt() is None:
        pytest.skip("tessera-opt is not built on this host")
    monkeypatch.setattr(w8a8, "blockscale_panel_oracle",
                        lambda *a, **k: w8a8.BlockScalePanel("global", 32, 32, 1))
    with pytest.raises(ValueError, match="oracle disagrees with the native Schedule"):
        w8a8.lower_blockscale(shape)
