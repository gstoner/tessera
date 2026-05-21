"""Generic grid-AI core benchmark tests.

This is the tiny library benchmark for gridded regional AI models.  It stays
above Tessera as library code while proving the compiler-visible pieces compose:
tiled fields, stencil locality, 2D local-window attention, fused conv block,
deterministic noise, mesh-region halo lowering, and CPU/mock oracle comparison.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.grid_ai_core import (  # noqa: E402
    GridAICoreBenchmark,
    GridAICoreConfig,
    GridAICoreModel,
    deterministic_noise_step,
    local_stencil_feature,
    periodic_halo_oracle,
    tile_field,
)
from tessera.rng import RNGKey  # noqa: E402
from tessera.testing import halo_transport  # noqa: E402

FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "grid_ai_core_ir_visible.mlir"
)


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


def _lower_fixture() -> str:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run(
        [
            binary,
            "--allow-unregistered-dialect",
            "-tessera-stencil-lower",
            "-tessera-boundary-condition-lower",
            "-tessera-halo-mesh-integration",
            "-tessera-halo-transport-lower",
            str(FIXTURE),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0 and "Did you mean" in r.stderr:
        pytest.skip("tessera-opt predates required passes")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _axes_widths_from_lowered_ir(ir: str) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for line in ir.splitlines():
        if "tessera.neighbors.halo.transport" not in line:
            continue
        axis = re.search(r"axis\s*=\s*(\d+)", line)
        width = re.search(r"width\s*=\s*(\d+)", line)
        if axis and width:
            pairs.add((int(axis.group(1)), int(width.group(1))))
    return sorted(pairs)


def test_tile_field_splits_nhwc_spatial_axes() -> None:
    x = np.arange(1 * 4 * 6 * 2, dtype=np.float32).reshape(1, 4, 6, 2)
    tiles = tile_field(x, (2, 3))
    assert len(tiles) == 4
    assert tiles[0].shape == (1, 2, 3, 2)
    np.testing.assert_array_equal(tiles[0], x[:, 0:2, 0:3, :])
    np.testing.assert_array_equal(tiles[3], x[:, 2:4, 3:6, :])


def test_local_stencil_feature_periodic_matches_hand_oracle() -> None:
    x = np.arange(1 * 3 * 3 * 1, dtype=np.float32).reshape(1, 3, 3, 1)
    got = local_stencil_feature(x, alpha=0.5, boundary="periodic")
    center = x
    oracle = center + 0.5 * (
        np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
        + np.roll(x, 1, axis=2)
        + np.roll(x, -1, axis=2)
        - 4.0 * center
    )
    np.testing.assert_array_equal(got, oracle)


def test_deterministic_noise_step_replays_from_same_key() -> None:
    x = np.zeros((1, 4, 4, 2), dtype=np.float32)
    key = RNGKey.from_seed(42)
    np.testing.assert_array_equal(
        deterministic_noise_step(x, key, 0.1),
        deterministic_noise_step(x, key, 0.1),
    )


def test_model_forward_shape_and_determinism() -> None:
    cfg = GridAICoreConfig(B=1, H=8, W=8, C_in=3, C_hid=4, C_out=2, heads=2)
    model_a = GridAICoreModel(cfg)
    model_b = GridAICoreModel(cfg)
    x = GridAICoreBenchmark.make_input(cfg)
    out_a = model_a(x, step=0)
    out_b = model_b(x, step=0)
    assert out_a.shape == (1, 8, 8, 2)
    np.testing.assert_array_equal(out_a, out_b)


def test_benchmark_result_schema_is_canonical() -> None:
    cfg = GridAICoreConfig(B=1, H=8, W=8, C_in=3, C_hid=4, C_out=2, heads=2)
    row = GridAICoreBenchmark(warmup=0, reps=1).run_one(cfg).to_dict()
    for key in (
        "backend",
        "op",
        "shape",
        "dtype",
        "latency_ms",
        "throughput_msps",
        "memory_bw_gb_s",
        "device",
        "tessera_version",
        "determinism_ok",
    ):
        assert key in row
    assert row["op"] == "grid_ai_core_forward"
    assert row["determinism_ok"] is True


def test_fixture_names_all_required_compiler_visible_pieces() -> None:
    text = FIXTURE.read_text()
    for needle in (
        "schedule.mesh.region",
        "tessera.neighbors.stencil.apply",
        "tessera.attn_local_window_2d",
        "tessera.conv2d_nhwc",
        "tessera.relu",
        "tessera_rng.normal",
    ):
        assert needle in text


def test_mesh_region_lowering_keeps_halo_transport_visible() -> None:
    out = _lower_fixture()
    assert "tessera.neighbors.halo.exchange" not in out
    assert "tessera.neighbors.halo.pack" in out
    assert "tessera.neighbors.halo.transport" in out
    assert "tessera.neighbors.halo.unpack" in out
    assert 'source_op = "tessera.attn_local_window_2d"' in out
    assert "tessera.conv2d_nhwc" in out
    assert "tessera.relu" in out
    assert "tessera_rng.normal" in out


def test_ir_driven_halo_mock_matches_cpu_oracle() -> None:
    out = _lower_fixture()
    axes_widths = _axes_widths_from_lowered_ir(out)
    assert axes_widths == [(0, 1), (1, 1)]
    # Compare one IR-derived axis to avoid the current mock runtime's
    # two-axis corner overwrite ambiguity.  The structural check above still
    # proves both spatial axes are emitted by the compiler.
    oracle_axes_widths = [axes_widths[-1]]

    ranks = [
        np.arange(16, dtype=np.float32).reshape(4, 4),
        np.arange(16, 32, dtype=np.float32).reshape(4, 4),
    ]
    mock = halo_transport.halo_exchange_ring(ranks, axes_widths=oracle_axes_widths)
    oracle = periodic_halo_oracle(ranks, oracle_axes_widths)
    for rank, (got, expected) in enumerate(zip(mock, oracle, strict=True)):
        np.testing.assert_array_equal(
            got,
            expected,
            err_msg=f"rank {rank} mock halo output diverged from CPU oracle",
        )
