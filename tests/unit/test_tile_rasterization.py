"""Gate for ``compiler/tile_rasterization.py`` — block rasterization order.

A rasterization order is a permutation of block ids onto the tile grid, so it is
semantics-preserving *by construction* — which means it has a total, executable,
hardware-free oracle. That is what this file exercises:

* :func:`is_bijection` over a spread of grids including ragged ones (where a
  naive ``bid % group`` silently drops and duplicates tiles).
* The **emitted C matches the Python reference** — compiled and run on the host
  clang for every block id. The device kernel and the model of it cannot drift,
  and this is provable here with no GPU (the emitted snippet is plain integer
  arithmetic, identical under CUDA and HIP).
* The default order is the byte-identical identity, so wiring the knob into an
  emitter changes nothing until a schedule sets it.

See ``docs/audit/compiler/TILESIGHT_ASSESSMENT.md`` §3.2.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from tessera.compiler.autotune_v2 import TuningConfig
from tessera.compiler.tile_rasterization import (
    RASTER_GROUP_CHOICES,
    RASTER_ORDER_CHOICES,
    RasterError,
    RasterOrder,
    emit_c,
    is_bijection,
    num_blocks,
    remap,
    suggest_group,
    working_set_tiles,
)

# Square, wide, tall, prime, and ragged-vs-group grids. The ragged ones matter
# most: grid_m not divisible by the group is where a short final panel appears.
_GRIDS = [(1, 1), (1, 8), (8, 1), (4, 4), (8, 3), (3, 8), (7, 5), (5, 7),
          (13, 6), (6, 13), (16, 16), (17, 3)]
_ORDERS = [RasterOrder(o) for o in RASTER_ORDER_CHOICES]


# --- the oracle: every order is a permutation --------------------------------

@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("group", RASTER_GROUP_CHOICES)
def test_every_order_is_a_bijection(order: RasterOrder, grid, group: int) -> None:
    """Every tile computed exactly once — the whole correctness claim."""
    grid_m, grid_n = grid
    assert is_bijection(order, grid_m, grid_n, group=group), (
        f"{order.value} group={group} is not a permutation of a "
        f"{grid_m}x{grid_n} grid")


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("grid", _GRIDS)
def test_rasterization_never_changes_the_launch_geometry(order, grid) -> None:
    """A permutation cannot change how many blocks launch — if it did, the knob
    would not be free to set."""
    grid_m, grid_n = grid
    coords = {remap(order, b, grid_m, grid_n, group=4)
              for b in range(num_blocks(grid_m, grid_n))}
    assert len(coords) == grid_m * grid_n


def test_group_of_one_degenerates_to_the_ungrouped_order() -> None:
    """Keeps the identity reachable from every point in the search space."""
    for bid in range(35):
        assert (remap(RasterOrder.GROUPED_M, bid, 5, 7, group=1)
                == remap(RasterOrder.ROW_MAJOR, bid, 5, 7))
        assert (remap(RasterOrder.GROUPED_N, bid, 5, 7, group=1)
                == remap(RasterOrder.COLUMN_MAJOR, bid, 5, 7))


def test_row_major_is_the_existing_mapping() -> None:
    """The default must reproduce what every emitter already computes, so
    enabling the knob is a no-op until a schedule sets it."""
    grid_m, grid_n = 6, 9
    for bid in range(grid_m * grid_n):
        assert remap(RasterOrder.ROW_MAJOR, bid, grid_m, grid_n) == (
            bid // grid_n, bid % grid_n)


def test_grouped_m_fills_a_panel_before_advancing() -> None:
    """The behaviour the locality win rests on: consecutive blocks stay within a
    `group`-tall band of M rather than sweeping a whole N row."""
    coords = [remap(RasterOrder.GROUPED_M, b, 4, 4, group=2) for b in range(8)]
    assert {m for m, _ in coords} == {0, 1}      # first panel: M rows 0-1 only
    assert {n for _, n in coords} == {0, 1, 2, 3}


# --- error handling ----------------------------------------------------------

def test_out_of_range_block_id_raises_rather_than_wrapping() -> None:
    with pytest.raises(RasterError, match="out of range"):
        remap(RasterOrder.ROW_MAJOR, 12, 3, 4)
    with pytest.raises(RasterError, match="out of range"):
        remap(RasterOrder.ROW_MAJOR, -1, 3, 4)


def test_invalid_grid_group_and_order_raise() -> None:
    with pytest.raises(RasterError, match="grid dims"):
        remap(RasterOrder.ROW_MAJOR, 0, 0, 4)
    with pytest.raises(RasterError, match="group"):
        remap(RasterOrder.GROUPED_M, 0, 4, 4, group=0)
    with pytest.raises(ValueError):
        remap("z_order", 0, 4, 4)          # Morton is deliberately absent
    with pytest.raises(RasterError, match="window"):
        working_set_tiles(RasterOrder.ROW_MAJOR, 4, 4, window=0)


# --- the working-set counting fact -------------------------------------------

def test_grouping_shrinks_the_concurrent_working_set() -> None:
    """The mechanism, stated as a counting fact rather than a latency claim: for
    a wave of 16 concurrent blocks, row-major touches a distinct column per block
    (16 columns + 1 row = 17 panels); grouped-M with g=4 touches 4 rows + 4
    columns = 8."""
    row = working_set_tiles(RasterOrder.ROW_MAJOR, 16, 16, window=16)
    grouped = working_set_tiles(RasterOrder.GROUPED_M, 16, 16, group=4, window=16)
    assert row == 17
    assert grouped == 8


def test_working_set_is_minimised_near_sqrt_of_the_wave() -> None:
    """The `g + w/g` shape, and the reason ``suggest_group`` exists: the curve is
    U-shaped in the group size with its floor at sqrt(window)."""
    window = 16
    by_group = {g: working_set_tiles(RasterOrder.GROUPED_M, 16, 16,
                                     group=g, window=window)
                for g in RASTER_GROUP_CHOICES}
    best = min(by_group, key=lambda g: by_group[g])
    assert best == suggest_group(window) == 4
    # ...and both ends of the sweep degrade back to the ungrouped cost.
    assert by_group[1] == by_group[16] == 17


def test_sliding_windows_understate_the_grouping_win() -> None:
    """Documents why ``aligned`` defaults to True: an unaligned window is
    dominated by the one straddle across a panel boundary, which is a phase a
    real wave dispatch never has."""
    aligned = working_set_tiles(RasterOrder.GROUPED_M, 16, 16, group=4,
                                window=16)
    sliding = working_set_tiles(RasterOrder.GROUPED_M, 16, 16, group=4,
                                window=16, aligned=False)
    assert aligned < sliding


def test_suggest_group_tracks_sqrt_of_the_wave() -> None:
    assert suggest_group(16) == 4
    assert suggest_group(64) == 8
    assert suggest_group(1) == 1
    assert suggest_group(300) in RASTER_GROUP_CHOICES


# --- emitted C matches the reference (compiled and executed on the host) -----

_CLANG = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _run_emitted(order: RasterOrder, grid_m: int, grid_n: int, group: int,
                 tmp_path: Path) -> list[tuple[int, int]]:
    """Compile the emitted snippet with the host C compiler and run it over every
    block id, returning the `(m, n)` it computes.

    The snippet is plain integer arithmetic with no vendor intrinsics, which is
    exactly why one emission can serve CUDA and HIP — and why it can be validated
    here with no GPU.
    """
    body = emit_c(order, group=group, bid="bid", grid_m="gm", grid_n="gn",
                  indent="    ")
    src = textwrap.dedent("""\
        #include <stdio.h>
        int main(void) {
            const int gm = %d, gn = %d;
            for (int bid = 0; bid < gm * gn; ++bid) {
        %s
                printf("%%d %%d\\n", mt, nt);
            }
            return 0;
        }
        """) % (grid_m, grid_n, textwrap.indent(body, "    "))
    csrc = tmp_path / "raster.c"
    csrc.write_text(src)
    exe = tmp_path / "raster.bin"
    compile_proc = subprocess.run(
        [_CLANG, "-std=c11", "-O1", "-Werror", str(csrc), "-o", str(exe)],
        capture_output=True, text=True)
    assert compile_proc.returncode == 0, (
        f"emitted snippet did not compile:\n{compile_proc.stderr}\n--- source ---\n{src}")
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return [(int(a), int(b))
            for a, b in (line.split() for line in run.stdout.split("\n") if line)]


@pytest.mark.skipif(_CLANG is None, reason="no host C compiler available")
@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("grid,group", [((8, 3), 4), ((13, 6), 4), ((7, 5), 2),
                                        ((16, 16), 8), ((5, 7), 1)])
def test_emitted_c_matches_the_python_reference(order, grid, group, tmp_path) -> None:
    """The emitter and the model of it must agree for every block id — otherwise
    the bijection proof above is proving something the device does not run."""
    grid_m, grid_n = grid
    expected = [remap(order, b, grid_m, grid_n, group=group)
                for b in range(grid_m * grid_n)]
    assert _run_emitted(order, grid_m, grid_n, group, tmp_path) == expected


@pytest.mark.skipif(_CLANG is None, reason="no host C compiler available")
@pytest.mark.parametrize("order", [RasterOrder.GROUPED_M, RasterOrder.GROUPED_N,
                                   RasterOrder.ROW_MAJOR])
def test_two_emissions_in_one_scope_compile(order, tmp_path) -> None:
    """Regression: the grouped forms used to declare bare `const int _pp/_pn/...`
    in the enclosing scope, so a kernel mapping two grids hit `error:
    redefinition of '_pp'` — and any kernel already owning those names collided.
    Temporaries are now block-scoped behind a `_tsr_raster_` prefix."""
    body = (emit_c(order, group=4, out_m="m1", out_n="n1", indent="    ")
            + emit_c(order, group=4, out_m="m2", out_n="n2", indent="    "))
    src = ("int f(int bid, int gm, int gn) {\n" + body
           + "    return m1 + n1 + m2 + n2;\n}\n")
    csrc = tmp_path / "twice.c"
    csrc.write_text(src)
    proc = subprocess.run(
        [_CLANG, "-std=c11", "-O1", "-Wall", "-Wshadow", "-Werror", "-c",
         str(csrc), "-o", str(tmp_path / "twice.o")],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stderr}\n--- source ---\n{src}"


@pytest.mark.skipif(_CLANG is None, reason="no host C compiler available")
def test_emission_does_not_collide_with_caller_variables(tmp_path) -> None:
    """The surrounding kernel may already own a variable named like one of our
    temporaries; block scoping must keep that legal even under -Wshadow."""
    body = emit_c(RasterOrder.GROUPED_M, group=4, indent="    ")
    src = ("int f(int bid, int gm, int gn) {\n"
           "    const int _pp = 7, _pn = 8, _pr = 9, _pw = 10, _pfm = 11;\n"
           + body
           + "    return mt + nt + _pp + _pn + _pr + _pw + _pfm;\n}\n")
    csrc = tmp_path / "collide.c"
    csrc.write_text(src)
    proc = subprocess.run(
        [_CLANG, "-std=c11", "-O1", "-Wall", "-Wshadow", "-Werror", "-c",
         str(csrc), "-o", str(tmp_path / "collide.o")],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stderr}\n--- source ---\n{src}"


@pytest.mark.skipif(_CLANG is None, reason="no host C compiler available")
def test_emitted_row_major_is_the_plain_divide_modulo(tmp_path) -> None:
    """The default emission must stay the exact expression the emitters already
    contain, so turning the knob on cannot perturb a verified kernel."""
    text = emit_c(RasterOrder.ROW_MAJOR, bid="bid", grid_m="gm", grid_n="gn")
    assert "bid / gn" in text and "bid % gn" in text
    assert _run_emitted(RasterOrder.ROW_MAJOR, 4, 5, 1, tmp_path) == [
        remap(RasterOrder.ROW_MAJOR, b, 4, 5) for b in range(20)]


def test_emit_rejects_a_bad_group() -> None:
    with pytest.raises(RasterError, match="group"):
        emit_c(RasterOrder.GROUPED_M, group=0)


# --- knob plumbing -----------------------------------------------------------

def test_tuning_config_defaults_to_the_identity_order() -> None:
    cfg = TuningConfig(128, 128, 32)
    assert cfg.raster_order == RasterOrder.ROW_MAJOR.value
    assert cfg.raster_group == 1
    assert cfg.to_dict()["raster_order"] == "row_major"
    assert 'raster_order = "row_major"' in cfg.to_ir_attr()


def test_tuning_config_carries_a_swizzle() -> None:
    cfg = TuningConfig(128, 128, 32, raster_order="grouped_m", raster_group=8)
    assert cfg.to_dict()["raster_group"] == 8
    assert "raster_group = 8" in cfg.to_ir_attr()


def test_tuning_config_rejects_an_unknown_order() -> None:
    with pytest.raises(ValueError, match="raster_order"):
        TuningConfig(128, 128, 32, raster_order="zorder")
    with pytest.raises(ValueError, match="raster_group"):
        TuningConfig(128, 128, 32, raster_group=0)


def _matmul_graph():
    from tessera.compiler.graph_ir import (
        TENSOR_OPAQUE, GraphIRFunction, GraphIRModule, IRArg, IROp)

    return GraphIRModule(functions=[
        GraphIRFunction(
            "main",
            args=[IRArg("A", TENSOR_OPAQUE), IRArg("B", TENSOR_OPAQUE)],
            body=[IROp("C", "tessera.matmul", ["%A", "%B"],
                       ["tensor<*x?>", "tensor<*x?>"], "tensor<*x?>")],
        )
    ])


def _schedule_ops(**config):
    from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir

    module = lower_graph_to_schedule_ir(
        _matmul_graph(), schedule_config=config or None)
    return [op for fn in module.functions for op in fn.body]


def test_schedule_ir_exposes_the_knob_and_the_selection() -> None:
    ops = _schedule_ops(raster_order="grouped_m", raster_group=8)
    knobs = {op.attrs["name"] for op in ops if op.op_name == "schedule.knob"}
    assert {"raster_order", "raster_group"} <= knobs

    tile = next(op for op in ops if op.op_name == "schedule.tile")
    assert tile.attrs["raster_order"] == "grouped_m"
    assert tile.attrs["raster_group"] == 8


def test_tuning_cache_round_trips_a_swizzle(tmp_path) -> None:
    """A tuned swizzle must survive the SQLite cache. If the columns were
    missing, the config would reload as the identity and the swizzle would look
    like it simply did not help — a silent degrade of exactly the kind the
    fallback logs exist to catch."""
    import sqlite3

    from tessera.compiler.autotune_v2 import (
        BayesianAutotuner, GEMMWorkload, TuningResult)

    db = tmp_path / "tuning.db"
    workload = GEMMWorkload(M=256, N=256, K=256)
    tuner = BayesianAutotuner(workload)
    cfg = TuningConfig(128, 128, 32, raster_order="grouped_m", raster_group=8)
    tuner._results.append(
        TuningResult(config=cfg, latency_ms=1.0, tflops=33.5, trial_id=0))
    tuner.save_to_cache(str(db))

    reloaded = BayesianAutotuner(workload)
    assert reloaded.warm_start_from_cache(str(db)) == 1
    got = reloaded._results[0].config
    assert got.raster_order == "grouped_m"
    assert got.raster_group == 8

    # A pre-existing cache written before the columns existed must migrate to the
    # identity — which is what those rows actually measured.
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute(
            "CREATE TABLE tuning_results (M INT, N INT, K INT, dtype TEXT,"
            " tile_m INT, tile_n INT, tile_k INT, num_warps INT, num_stages INT,"
            " latency_ms REAL, tflops REAL, sampled_at REAL, trial_id INT)")
        conn.execute(
            "INSERT INTO tuning_results VALUES (256,256,256,'bf16',"
            "128,128,32,4,2,1.0,33.5,0.0,0)")
    migrated = BayesianAutotuner(workload)
    assert migrated.warm_start_from_cache(str(legacy)) == 1
    assert migrated._results[0].config.raster_order == "row_major"
    assert migrated._results[0].config.raster_group == 1
    assert migrated.warm_start_skipped == []


def test_unknown_raster_order_in_cache_is_reported_not_silently_dropped(
        tmp_path) -> None:
    """Regression: a row naming an order this build does not know raised
    ValueError inside TuningConfig, which the generic `skip corrupted rows`
    handler swallowed — a valid measured result vanished with no diagnostic,
    reintroducing the silent degrade the raster columns exist to prevent.

    It must still be skipped (the latency was measured *with* that order, so it
    cannot be relabelled row_major), but the loss has to be visible."""
    import sqlite3

    from tessera.compiler.autotune_v2 import (
        BayesianAutotuner, GEMMWorkload, _ensure_cache_schema)

    db = tmp_path / "future.db"
    with sqlite3.connect(db) as conn:
        _ensure_cache_schema(conn)
        conn.execute(
            "INSERT INTO tuning_results (M,N,K,dtype,arch,layout,movement_json,"
            "tile_m,tile_n,tile_k,num_warps,num_stages,latency_ms,tflops,"
            "sampled_at,trial_id,status,reason,method,raster_order,raster_group)"
            " VALUES (256,256,256,'bf16','generic','row_major',"
            "'{\"overlap\": \"compute\", \"prefetch\": \"auto\"}',"
            "128,128,32,4,2,1.0,33.5,0.0,0,'ok','','measured','z_order',4)")

    tuner = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256))
    assert tuner.warm_start_from_cache(str(db)) == 0
    skipped = tuner.warm_start_skipped
    assert len(skipped) == 1
    assert "z_order" in skipped[0]
    assert "128x128x32" in skipped[0]


def test_corrupted_cache_row_is_also_reported(tmp_path) -> None:
    """The generic skip path must name its rows too — a warm start that returns
    fewer results than the cache holds should never be indistinguishable from a
    cold one."""
    import sqlite3

    from tessera.compiler.autotune_v2 import (
        BayesianAutotuner, GEMMWorkload, _ensure_cache_schema)

    db = tmp_path / "corrupt.db"
    with sqlite3.connect(db) as conn:
        _ensure_cache_schema(conn)
        # num_warps=3 is not a legal warp count.
        conn.execute(
            "INSERT INTO tuning_results (M,N,K,dtype,arch,layout,movement_json,"
            "tile_m,tile_n,tile_k,num_warps,num_stages,latency_ms,tflops,"
            "sampled_at,trial_id,status,reason,method,raster_order,raster_group)"
            " VALUES (256,256,256,'bf16','generic','row_major',"
            "'{\"overlap\": \"compute\", \"prefetch\": \"auto\"}',"
            "128,128,32,3,2,1.0,33.5,0.0,0,'ok','','measured','row_major',1)")

    tuner = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256))
    assert tuner.warm_start_from_cache(str(db)) == 0
    assert len(tuner.warm_start_skipped) == 1
    assert "num_warps" in tuner.warm_start_skipped[0]


def test_raster_axes_are_carried_but_not_swept() -> None:
    """The analytical model may prune but cannot promote a raster choice.

    ROCM-CALIB-1 rejected the prior locality metric on its home architecture.
    Enumeration therefore remains owned by exact-device sweeps until each
    backend records a correlation/retain verdict for the new model.
    """
    from tessera.compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload

    tuner = BayesianAutotuner(GEMMWorkload(M=256, N=256, K=256))
    orders = {c.raster_order for c in tuner.legal_candidates()}
    groups = {c.raster_group for c in tuner.legal_candidates()}
    assert orders == {"row_major"}, (
        "the candidate generator now promotes raster_order without an "
        "architecture-owned exact-device retain verdict")
    assert groups == {1}


def test_schedule_ir_defaults_to_row_major_and_rejects_an_unknown_order() -> None:
    tile = next(op for op in _schedule_ops() if op.op_name == "schedule.tile")
    assert tile.attrs["raster_order"] == "row_major"
    assert tile.attrs["raster_group"] == 1

    with pytest.raises(ValueError, match="raster_order"):
        _schedule_ops(raster_order="zorder")


@pytest.mark.parametrize("group", [0, -5])
def test_schedule_ir_rejects_an_out_of_range_group(group: int) -> None:
    """Regression: `raster_order` was validated but `raster_group` was not, so 0
    and negatives sailed into `schedule.tile` attrs — where `TuningConfig` would
    have rejected the same value, and where 0 divides by zero in `_grouped`."""
    with pytest.raises(ValueError, match="raster_group"):
        _schedule_ops(raster_group=group)


def test_schedule_ir_accepts_a_group_off_the_sweep_grid() -> None:
    """RASTER_GROUP_CHOICES are sweep hints, not a legality bound — a measured
    optimum may sit between them."""
    tile = next(op for op in _schedule_ops(raster_order="grouped_m",
                                           raster_group=6)
                if op.op_name == "schedule.tile")
    assert tile.attrs["raster_group"] == 6
