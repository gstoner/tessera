"""Owning-device proof for gfx1201 block-scaled FP8 W8A8 (ROCM-FP8-BLOCKSCALE-1).

Every row compiles the logical ``tessera.scaled_matmul`` through Graph ->
Schedule -> Tile -> Target -> HSACO, asserts the emitted instruction and the
isolated-group structure, launches it, and compares against an fp64 oracle of
the block-scaled math. Numbers alone cannot distinguish "scaled per group" from
"scaled once" when the scales are uniform, so the scales here vary by an order
of magnitude per group and per block, and one row checks the result is far
from what a dropped-group-scale kernel would return.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import (
    PACKAGE_ABIS,
    WEIGHT_LAYOUTS,
    BlockScaleProgram,
    BlockScaleShape,
    blockscale_reference,
    lower_blockscale,
    package_blockscale,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt

pytestmark = [
    pytest.mark.hardware_rocm,
    pytest.mark.skipif(
        os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
        reason="explicit gfx1201 owning-device gate",
    ),
]


def _inputs(shape: BlockScaleShape, *, exact: bool, seed: int):
    rng = np.random.default_rng(seed)
    m, n, k = shape.m, shape.n, shape.k
    if exact:
        # Small integers are exact in e4m3 and their products/sums exact in
        # fp32; power-of-two scales keep every scaled partial exact too, so the
        # device result must EQUAL the fp64 oracle.
        a = rng.integers(-3, 4, size=(m, k)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
        b = rng.integers(-3, 4, size=(k, n)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
        a_scale = np.exp2(rng.integers(-2, 3, size=(m, shape.groups))).astype(np.float32)
        b_scale = np.exp2(rng.integers(-2, 3, size=(shape.groups, shape.n_groups))).astype(np.float32)
    else:
        a = (rng.standard_normal((m, k)) * 2.0).astype(ml_dtypes.float8_e4m3fn)
        b = (rng.standard_normal((k, n)) * 2.0).astype(ml_dtypes.float8_e4m3fn)
        # Per-group and per-block scales spread over ~two decades.
        a_scale = np.exp(rng.uniform(-2.5, 2.5, size=(m, shape.groups))).astype(np.float32)
        b_scale = np.exp(rng.uniform(-2.5, 2.5, size=(shape.groups, shape.n_groups))).astype(np.float32)
    return a, b, a_scale, b_scale


def _launch(package, a, b, a_scale, b_scale, shape: BlockScaleShape) -> np.ndarray:
    output = np.zeros((shape.m, shape.n),
                      ml_dtypes.bfloat16 if shape.output == "bf16" else np.float32)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    # The logical B is [K, N]; the `nk` contract takes the same matrix stored
    # as the weight [N, K].
    weight = np.ascontiguousarray(b.T) if shape.weight_layout == "nk" else b
    result = rt.launch(artifact, {
        "buffers": {"a": a, "b": weight, "a_scale": a_scale, "b_scale": b_scale, "o": output},
        "scalars": {"M": shape.m, "N": shape.n, "K": shape.k},
    })
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    return output


def _generated_kernel(tile_ir: str, *, k_unroll: int, group_panels: int) -> str:
    """The generator's typed kernel, before the fragment ops are lowered."""
    tool = find_tessera_opt()
    assert tool is not None
    option = f" scale-group-panels={group_panels}" if group_panels >= 0 else ""
    done = subprocess.run(
        [str(tool), "-", f"--generate-wmma-gemm-kernel=via-tile=true k-unroll={k_unroll}{option}"],
        input=tile_ir, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _assert_structure(package, shape: BlockScaleShape, *, k_unroll: int,
                      group_panels: int) -> None:
    from tests._support import rocm_isa

    rocm_isa.assert_selected(
        package.image.payload,
        chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8",
        what="block-scaled FP8 W8A8",
    )
    text = _generated_kernel(package.tile_ir, k_unroll=k_unroll, group_panels=group_panels)
    block_m, block_n = package.descriptor.provenance["macro_tile"]
    macro_k = int(package.descriptor.provenance["macro_k"])
    fragments = (block_m // 16) * (block_n // 16)
    groups_per_iteration = macro_k * k_unroll // shape.scale_k
    # Fast and edge paths each emit one join per fragment per group in the
    # loop body, plus one group's worth in the remainder loop when an
    # iteration holds more than one group.
    per_path = fragments * (groups_per_iteration + (1 if groups_per_iteration > 1 else 0))
    joins = len(re.findall(r"tile\.fragment_scaled_accumulate", text))
    assert joins == 2 * per_path, (joins, per_path)
    # Each group's partial starts from its own zero: one zero per fragment per
    # group site, plus the running accumulators' initial zeros.
    zeros = len(re.findall(r"tile\.fragment_zero", text))
    assert zeros == 2 * per_path + fragments, (zeros, per_path)
    assert f"scale_n = {shape.scale_n}" in text
    # Each group site issues `step` panels straight-line (the rest of the
    # group walks the inner loop), and every panel is one MMA per fragment.
    group = shape.scale_k // 16
    step = group if group_panels == 0 else min(2 if group_panels < 0 else group_panels, group)
    mmas = len(re.findall(r"tile\.mma ", text))
    assert mmas == 2 * per_path * step, (mmas, per_path, step)
    # The weight layout reaches the fragment source: [N, K] is read K-major.
    col_major_views = len(re.findall(r'tile\.memory = #tile\.memory_layout<space = "gmem", '
                                     r'order = "col_major"', text))
    assert (col_major_views > 0) == (shape.weight_layout == "nk"), col_major_views


SHAPES = [
    # (shape, k_unroll, scale_group_panels) -- ragged M/N, one to four groups,
    # N blocks of 128, 16 and 1, a 2-group iteration with its one-group
    # remainder, the whole-group straight-line body (0) and an inner loop of
    # one panel (1), in both named weight layouts.
    (BlockScaleShape(40, 72, 256, 128, 128), 1, -1),
    (BlockScaleShape(17, 19, 64, 32, 16), 1, -1),
    (BlockScaleShape(65, 130, 384, 128, 128), 1, 0),
    (BlockScaleShape(64, 96, 192, 64, 1), 1, 1),
    (BlockScaleShape(128, 256, 512, 128, 128), 1, -1),
    (BlockScaleShape(96, 160, 384, 128, 128), 2, -1),
    (BlockScaleShape(40, 72, 256, 128, 128, "nk"), 1, -1),
    (BlockScaleShape(17, 19, 64, 32, 16, "nk"), 1, 1),
    (BlockScaleShape(65, 130, 384, 128, 128, "nk"), 1, 0),
    (BlockScaleShape(128, 256, 512, 128, 128, "nk"), 1, -1),
    (BlockScaleShape(96, 160, 384, 128, 128, "nk"), 2, 4),
]


def _id(value):
    if isinstance(value, BlockScaleShape):
        return (f"{value.weight_layout}_{value.m}x{value.n}x{value.k}"
                f"_g{value.scale_k}_n{value.scale_n}")
    return str(value)


@pytest.mark.parametrize("exact", [True, False], ids=["exact", "random"])
@pytest.mark.parametrize(("shape", "k_unroll", "group_panels"), SHAPES, ids=_id)
def test_blockscale_w8a8_matches_fp64_oracle_on_gfx1201(shape, k_unroll, group_panels, exact):
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_blockscale(lower_blockscale(shape), k_unroll=k_unroll,
                                 scale_group_panels=group_panels)
    assert package.descriptor.abi_id == WEIGHT_LAYOUTS[shape.weight_layout][1]
    assert package.image.architecture == "gfx1201"
    _assert_structure(package, shape, k_unroll=k_unroll, group_panels=group_panels)
    a, b, a_scale, b_scale = _inputs(shape, exact=exact, seed=shape.m * 7 + shape.k)
    got = _launch(package, a, b, a_scale, b_scale, shape)
    want = blockscale_reference(a.astype(np.float32), b.astype(np.float32), a_scale, b_scale,
                                scale_k=shape.scale_k, scale_n=shape.scale_n)
    if exact:
        np.testing.assert_array_equal(got, want.astype(np.float32))
    else:
        # fp32 accumulation of at most K products per element: bound the error
        # by the magnitude of the terms, not the (cancelling) result.
        magnitude = blockscale_reference(
            np.abs(a.astype(np.float32)), np.abs(b.astype(np.float32)),
            a_scale, b_scale, scale_k=shape.scale_k, scale_n=shape.scale_n)
        err = np.abs(got.astype(np.float64) - want)
        assert np.all(err <= 4 * shape.k * np.finfo(np.float32).eps * magnitude + 1e-30), (
            float((err / (magnitude + 1e-30)).max()))


# ---------------------------------------------------------------------------
# The LDS-staged multi-wave body (GFX1201-PERF-2026-09-27) and the bf16 store.
# ---------------------------------------------------------------------------
def _with_schedule(program: BlockScaleProgram, *, staging: str, warps: int,
                   macro: tuple[int, int], depth: int = 1) -> BlockScaleProgram:
    """Rewrite the carrier's physical schedule (performance keys only) -- the
    same seam the recorded sweep uses. The semantic contract is untouched."""
    tile = program.tile_ir
    for pattern, value in ((r'staging = "\w+"', f'staging = "{staging}"'),
                           (r"(?<![\w.])warps = \d+", f"warps = {warps}"),
                           (r"tessera\.pipeline_depth = \d+", f"tessera.pipeline_depth = {depth}"),
                           (r"tessera\.macro_tile_m = \d+", f"tessera.macro_tile_m = {macro[0]}"),
                           (r"tessera\.macro_tile_n = \d+", f"tessera.macro_tile_n = {macro[1]}")):
        tile, count = re.subn(pattern, value, tile)
        assert count == 1, pattern
    return BlockScaleProgram(program.shape, program.entry, program.graph_ir,
                             program.schedule_ir, tile)


def _register_reference(shape: BlockScaleShape, a, b, a_scale, b_scale) -> np.ndarray:
    """The one-wave register panel on the same inputs (the body the LDS one
    must reproduce bit for bit)."""
    panel = (32, 32) if shape.m % 32 == 0 and shape.n % 32 == 0 else (16, 16)
    program = _with_schedule(lower_blockscale(shape), staging="global", warps=1, macro=panel)
    package = package_blockscale(program)
    assert package.descriptor.provenance["staging"] == "global"
    return _launch(package, a, b, a_scale, b_scale, shape)


def _assert_lds_isa(package, *, spill_free: bool = True) -> None:
    from tests._support import rocm_isa

    rocm_isa.assert_selected(
        package.image.payload, chip="gfx1201", pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8", what="LDS-staged block-scaled FP8 W8A8")
    text = rocm_isa.disassemble(package.image.payload, chip="gfx1201")
    ops = rocm_isa.mnemonics(text, r"[a-z_0-9]+")
    # The slab is staged with 128-bit global loads and 128-bit LDS stores,
    # the fragments come from LDS, the waves meet at barriers, and the body
    # the Schedule selects does not spill.
    assert ops["global_load_b128"] > 0 and ops["ds_store_b128"] > 0, ops
    assert ops["ds_load_b64"] + ops["ds_load_2addr_b64"] > 0, ops
    assert ops["s_barrier_signal"] > 0, ops
    assert spill_free == (not any(name.startswith("scratch_") for name in ops)), ops
    # LDS-only barriers: no L0 invalidate rides them.
    assert ops["global_inv"] == 0, ops


LDS_SHAPES = [
    # (shape, expected macro tile) -- Schedule-selected: 128x128 at >= 64
    # workgroups (ragged M included since FOUNDATION-BATCH-2-2026-09-27),
    # else 128x64; ragged M, ragged N and both; both output storages.
    (BlockScaleShape(256, 4096, 256, 128, 128, "nk"), (128, 128)),
    (BlockScaleShape(128, 4096, 384, 128, 128, "nk"), (128, 64)),
    (BlockScaleShape(200, 4096, 256, 128, 128, "nk"), (128, 128)),
    (BlockScaleShape(256, 4000, 256, 128, 128, "nk"), (128, 128)),
    (BlockScaleShape(256, 4096, 512, 128, 128, "nk", "bf16"), (128, 128)),
    (BlockScaleShape(200, 4000, 256, 128, 128, "nk", "bf16"), (128, 128)),
    (BlockScaleShape(200, 2048, 384, 128, 128, "nk"), (128, 64)),
    (BlockScaleShape(1000, 4096, 256, 128, 128, "nk"), (128, 128)),
    (BlockScaleShape(300, 4000, 256, 128, 128, "nk", "bf16"), (128, 128)),
]


def _lds_id(value):
    if isinstance(value, BlockScaleShape):
        return f"{value.m}x{value.n}x{value.k}_{value.output}"
    return f"{value[0]}x{value[1]}"


@pytest.mark.parametrize("exact", [True, False], ids=["exact", "random"])
@pytest.mark.parametrize(("shape", "macro"), LDS_SHAPES, ids=_lds_id)
def test_blockscale_w8a8_lds_body_on_gfx1201(shape, macro, exact):
    """The Schedule-selected LDS body: the expected workgroup, the fp64 oracle
    (bit-equal on exact inputs), and bitwise the register panel's result."""
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_blockscale(lower_blockscale(shape))
    prov = package.descriptor.provenance
    assert (prov["staging"], prov["warps"], tuple(prov["macro_tile"])) == ("lds", 8, macro)
    assert prov["workgroup"] == [256, 1, 1]
    assert package.descriptor.abi_id == PACKAGE_ABIS[(shape.weight_layout, shape.output)]
    _assert_lds_isa(package)
    a, b, a_scale, b_scale = _inputs(shape, exact=exact, seed=shape.m * 11 + shape.n)
    got = _launch(package, a, b, a_scale, b_scale, shape)
    want = blockscale_reference(a.astype(np.float32), b.astype(np.float32), a_scale, b_scale,
                                scale_k=shape.scale_k, scale_n=shape.scale_n)
    reference = _register_reference(shape, a, b, a_scale, b_scale)
    # Same partial order and the same join: the two bodies agree bit for bit.
    np.testing.assert_array_equal(got.view(np.uint16 if shape.output == "bf16" else np.uint32),
                                  reference.view(np.uint16 if shape.output == "bf16" else np.uint32))
    if shape.output == "bf16":
        f32_shape = BlockScaleShape(shape.m, shape.n, shape.k, shape.scale_k, shape.scale_n,
                                    shape.weight_layout, "f32")
        f32 = _launch(package_blockscale(lower_blockscale(f32_shape)), a, b, a_scale, b_scale,
                      f32_shape)
        # The bf16 store is the fp32 result rounded once, to nearest-even.
        np.testing.assert_array_equal(got.view(np.uint16),
                                      f32.astype(ml_dtypes.bfloat16).view(np.uint16))
        return
    if exact:
        np.testing.assert_array_equal(got, want.astype(np.float32))
    else:
        magnitude = blockscale_reference(
            np.abs(a.astype(np.float32)), np.abs(b.astype(np.float32)),
            a_scale, b_scale, scale_k=shape.scale_k, scale_n=shape.scale_n)
        err = np.abs(got.astype(np.float64) - want)
        assert np.all(err <= 4 * shape.k * np.finfo(np.float32).eps * magnitude + 1e-30)


@pytest.mark.parametrize(("macro", "warps", "depth", "knobs", "spill_free"), [
    # The register-staged next slab beside the 32x64 wave panel's 128
    # accumulators spilled at this ragged shape until the bounded store
    # stopped holding a per-element row across the loop
    # (FOUNDATION-BATCH-2-2026-09-27); it still measured 1.18-1.29x slower at
    # whole M and is not selected. (Double-buffering 128x128 needs 72 KiB of
    # LDS and is refused below.)
    ((128, 128), 8, 1, {"blockscale_prefetch": 1}, True),         # register next slab
    ((128, 64), 8, 2, {}, True),                                  # double-buffered, 32x32 waves
    ((128, 64), 8, 1, {"blockscale_stage_k": 64}, True),          # two slabs per group
    ((128, 64), 8, 1, {"blockscale_lds_pad_bytes": 0}, True),     # unpadded rows
    ((64, 64), 4, 1, {"blockscale_lds_pad_bytes": 32}, True),     # four waves
    ((256, 128), 16, 1, {}, True),                                # sixteen waves
], ids=["f1", "d2w32", "s64", "p0", "w4p32", "w16"])
def test_blockscale_w8a8_lds_knobs_compute_the_same_bits_on_gfx1201(macro, warps, depth, knobs,
                                                                     spill_free):
    """Every performance key of the LDS body -- staging schedule, slab K,
    padding, wave grid -- changes the kernel and never the result."""
    assert rt._rocm_live_arch() == "gfx1201"
    shape = BlockScaleShape(200, 520, 512, 128, 128, "nk")
    a, b, a_scale, b_scale = _inputs(shape, exact=False, seed=99)
    program = _with_schedule(lower_blockscale(shape), staging="lds", warps=warps, macro=macro,
                             depth=depth)
    package = package_blockscale(program, **knobs)
    assert package.descriptor.provenance["workgroup"] == [32 * warps, 1, 1]
    _assert_lds_isa(package, spill_free=spill_free)
    got = _launch(package, a, b, a_scale, b_scale, shape)
    reference = _register_reference(shape, a, b, a_scale, b_scale)
    np.testing.assert_array_equal(got.view(np.uint32), reference.view(np.uint32))


def test_blockscale_w8a8_lds_body_refuses_what_it_cannot_emit_on_gfx1201():
    """Refused by name, never answered with the register panel: a
    double-buffered 128x128 tile (72 KiB of LDS) and the [K, N] weight."""
    shape = BlockScaleShape(256, 4096, 256, 128, 128, "nk")
    program = _with_schedule(lower_blockscale(shape), staging="lds", warps=8,
                             macro=(128, 128), depth=2)
    with pytest.raises(RuntimeError, match="73728 LDS bytes exceed the 64 KiB"):
        package_blockscale(program)
    kn = BlockScaleShape(256, 4096, 256, 128, 128, "kn")
    program = _with_schedule(lower_blockscale(kn), staging="lds", warps=8, macro=(128, 128))
    with pytest.raises(RuntimeError, match=r"\[N, K\] weight"):
        package_blockscale(program)


@pytest.mark.parametrize("shape", [
    BlockScaleShape(40, 72, 256, 128, 128, "kn", "bf16"),
    BlockScaleShape(64, 96, 384, 128, 128, "nk", "bf16"),
], ids=_id)
def test_blockscale_w8a8_register_panel_bf16_store_on_gfx1201(shape):
    """The register panel's bf16 store is its own fp32 result rounded once."""
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_blockscale(lower_blockscale(shape))
    assert package.descriptor.provenance["staging"] == "global"
    assert package.descriptor.abi_id == PACKAGE_ABIS[(shape.weight_layout, "bf16")]
    a, b, a_scale, b_scale = _inputs(shape, exact=False, seed=5)
    got = _launch(package, a, b, a_scale, b_scale, shape)
    f32_shape = BlockScaleShape(shape.m, shape.n, shape.k, shape.scale_k, shape.scale_n,
                                shape.weight_layout, "f32")
    f32 = _launch(package_blockscale(lower_blockscale(f32_shape)), a, b, a_scale, b_scale,
                  f32_shape)
    np.testing.assert_array_equal(got.view(np.uint16), f32.astype(ml_dtypes.bfloat16).view(np.uint16))


def test_blockscale_w8a8_is_not_a_single_rescale_on_gfx1201():
    """The result must be far from any kernel that applied one group's scale
    to the whole K sum -- the failure mode an unisolated partial produces."""
    assert rt._rocm_live_arch() == "gfx1201"
    shape = BlockScaleShape(64, 128, 512, 128, 128)
    package = package_blockscale(lower_blockscale(shape))
    a, b, a_scale, b_scale = _inputs(shape, exact=False, seed=4242)
    got = _launch(package, a, b, a_scale, b_scale, shape)
    want = blockscale_reference(a.astype(np.float32), b.astype(np.float32), a_scale, b_scale,
                                scale_k=shape.scale_k, scale_n=shape.scale_n)
    full = a.astype(np.float64) @ b.astype(np.float64)
    column_block = np.arange(shape.n) // shape.scale_n
    wrong = full * a_scale[:, :1].astype(np.float64) * b_scale[0, column_block][None, :]
    scale = np.abs(want).max()
    assert np.abs(got - want).max() <= 1e-4 * scale
    assert np.abs(wrong - want).max() > 0.1 * scale


def _vgprs(payload: bytes) -> tuple[int, int]:
    """(VGPR count, VGPR spill count) of the one kernel in an HSACO."""
    import tempfile
    from pathlib import Path

    import shutil

    from tests._support import rocm_isa
    # The disassembler's own directory first; an assertions toolchain may
    # ship llvm-objdump without llvm-readelf, so fall back along the same
    # fleet list, then PATH. Failing (not skipping) when none is found.
    siblings = [Path(rocm_isa.llvm_objdump()).with_name("llvm-readelf")]
    siblings += [candidate.with_name("llvm-readelf") for candidate in rocm_isa._candidates()]
    readelf = next((path for path in siblings if path.is_file()), None)
    if readelf is None and shutil.which("llvm-readelf"):
        readelf = Path(shutil.which("llvm-readelf"))
    assert readelf is not None, f"llvm-readelf not found beside any of {siblings}"
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as image:
        image.write(payload)
        image.flush()
        notes = subprocess.run([str(readelf), "--notes", image.name], check=True,
                               capture_output=True, text=True).stdout
    vgpr = [int(v) for v in re.findall(r"\.vgpr_count:\s+(\d+)", notes)]
    spill = [int(v) for v in re.findall(r"\.vgpr_spill_count:\s+(\d+)", notes)]
    assert len(vgpr) == 1 and len(spill) == 1, notes
    return vgpr[0], spill[0]


@pytest.mark.parametrize(("n", "k"), [(24576, 1536), (8192, 1024), (4096, 7168)])
def test_ragged_m_costs_the_lds_body_no_registers(n, k):
    """FOUNDATION-BATCH-2-2026-09-27: a ragged M used to cost the 128x128 LDS
    body 13 VGPRs (251 vs 238, one wave per SIMD fewer) because the bounded
    store held a per-element row across the K loop. With the bounded store
    testing each row per lane (TileToROCM), the ragged kernel stays
    within the whole kernel's register allocation and does not spill -- the
    precondition for the Schedule giving ragged M the whole-M tile."""
    whole = package_blockscale(lower_blockscale(BlockScaleShape(1024, n, k, 128, 128, "nk")))
    ragged = package_blockscale(lower_blockscale(BlockScaleShape(1000, n, k, 128, 128, "nk")))
    for package in (whole, ragged):
        assert tuple(package.descriptor.provenance["macro_tile"]) == (128, 128)
    (whole_vgpr, whole_spill), (ragged_vgpr, ragged_spill) = (
        _vgprs(whole.image.payload), _vgprs(ragged.image.payload))
    assert whole_spill == ragged_spill == 0
    assert ragged_vgpr - whole_vgpr <= 8, (whole_vgpr, ragged_vgpr)
