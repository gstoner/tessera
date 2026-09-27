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
    WEIGHT_LAYOUTS,
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
    output = np.zeros((shape.m, shape.n), np.float32)
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
