"""Unit tests for tessera.compiler.rocm_occupancy.

Covers the RDNA occupancy composition: the quantised VGPR ladder (RDNA4 ISA
3.3.2.1), the per-wave cap clamp, the LDS pool-vs-request-cap distinction
(ISA 12.1), WGP/CU mode sensitivity, and the fail-closed behaviour on an arch
whose constants this fleet has not established.

Two tests here deliberately assert *unresolved* facts rather than preferred
ones -- the contested VGPR granule and the ``_MAX_WAVES`` unit discrepancy.
Both are device-resolvable and both would otherwise be invisible; an unwired
model's only defence against silent wrongness is a test that fails when the
model changes underneath it (Decision #29a).
"""

from __future__ import annotations

import pytest

from tessera.compiler.target_perf import Provenance
from tessera.compiler.rocm_occupancy import (
    MAX_WORKGROUPS_PER_WGP,
    MAX_WORKITEMS_PER_WORKGROUP,
    VGPR_GRANULE_CONTESTED,
    WAVE32,
    WAVE64,
    OccupancyRung,
    TesseraOccupancyError,
    allocate_vgprs,
    estimate_occupancy,
    granule_is_contested,
    headroom_to_next_rung,
    lds_pool_bytes,
    lds_alloc_granule,
    lds_request_cap_bytes,
    per_wave_vgpr_cap,
    occupancy_rungs,
    simds_per_slot,
    vgpr_alloc_granule,
    vgpr_regs_per_simd,
    wave_slots_per_simd,
    wave_slots_provenance,
)
from tessera.compiler.rocm_target import AMDArch, WorkgroupProcessorMode

RDNA4 = AMDArch.GFX_1201
RDNA35 = AMDArch.GFX_1151
#: An arch with no occupancy constants established on this fleet.
UNESTABLISHED = AMDArch.GFX_942

#: ``(vgpr_file_per_simd, wave32_granule, wave_slots_per_simd)`` measured on
#: Tajasarus 2026-09-20 -- unique solution over 12 observations.  See
#: ``benchmarks/baselines/gfx1201_vgpr_granule_20260920/``.
MEASURED_GFX1201 = (1536, 24, 16)


# ── ISA 3.3.2.1: allocation granule is derived, not tabled ──────────────────


def test_granule_follows_the_isa_rule_for_a_1536_register_file():
    """ "Devices that have 1536 VGPRs per SIMD allocate in blocks of 24 for
    wave32 and 12 for wave64." (RDNA4 ISA 3.3.2.1)"""
    assert vgpr_regs_per_simd(RDNA4) == 1536
    assert vgpr_alloc_granule(RDNA4, wave_size=WAVE32) == 24
    assert vgpr_alloc_granule(RDNA4, wave_size=WAVE64) == 12


def test_granule_is_none_for_an_arch_with_no_established_file_size():
    assert vgpr_regs_per_simd(UNESTABLISHED) is None
    assert vgpr_alloc_granule(UNESTABLISHED) is None


def test_bad_wave_size_is_refused():
    with pytest.raises(TesseraOccupancyError, match="wave_size"):
        vgpr_alloc_granule(RDNA4, wave_size=48)


# ── Allocation rounding and the per-wave cap ────────────────────────────────


def test_allocation_rounds_up_to_the_granule():
    alloc = allocate_vgprs(121, arch=RDNA4)
    assert (alloc.granule, alloc.blocks, alloc.allocated) == (24, 6, 144)
    assert alloc.wasted == 23
    assert alloc.waves_per_simd == 1536 // 144 == 10


def test_allocation_clamps_at_the_per_wave_cap_rather_than_rounding_past_it():
    """Regression: 256 is not a multiple of 24, so an unclamped ceil gives 264
    and reports 5 waves/SIMD.  The hardware caps a wave at 256 addressable
    registers, and 6 waves at 256 VGPRs is what the ROCm queue recorded."""
    alloc = allocate_vgprs(256, arch=RDNA4)
    assert alloc.allocated == 256
    assert alloc.waves_per_simd == 6


def test_the_recorded_256_vgpr_data_point_holds_under_either_candidate_granule():
    """1536/256 == 6 regardless of granule, which is why that data point pins
    the file size without presupposing the contested granule."""
    for granule in (16, 24):
        assert allocate_vgprs(256, arch=RDNA4, granule=granule).waves_per_simd == 6


def test_a_request_above_the_per_wave_cap_is_refused():
    with pytest.raises(TesseraOccupancyError, match="per-wave cap"):
        allocate_vgprs(257, arch=RDNA4)


def test_a_wave_may_not_be_created_with_zero_vgprs():
    """ISA 3.3.2.1, stated explicitly."""
    with pytest.raises(TesseraOccupancyError, match="at least 1 VGPR"):
        allocate_vgprs(0, arch=RDNA4)


def test_allocation_fails_closed_on_an_unestablished_arch():
    with pytest.raises(TesseraOccupancyError) as exc:
        allocate_vgprs(64, arch=UNESTABLISHED)
    # The refusal must name both the gap and how to close it.
    assert "VGPR file size per SIMD" in str(exc.value)
    assert "kernel-resource-usage" in str(exc.value)


# ── The ladder ──────────────────────────────────────────────────────────────


def test_rungs_are_strictly_descending_and_deduplicated():
    rungs = occupancy_rungs(RDNA4)
    waves = [r.waves_per_simd for r in rungs]
    budgets = [r.max_vgprs for r in rungs]
    assert waves == sorted(waves, reverse=True)
    assert len(set(waves)) == len(waves)
    assert budgets == sorted(budgets)


def test_every_rung_budget_actually_reaches_its_wave_count():
    """The ladder is only useful if ``max_vgprs`` is the true boundary: the
    budget reaches the rung and one more register falls off it."""
    for rung in occupancy_rungs(RDNA4):
        at = allocate_vgprs(rung.max_vgprs, arch=RDNA4)
        assert at.waves_per_simd == rung.waves_per_simd
        if rung.max_vgprs < 256:
            beyond = allocate_vgprs(rung.max_vgprs + 1, arch=RDNA4)
            assert beyond.waves_per_simd < rung.waves_per_simd


def test_shedding_registers_between_rungs_buys_nothing():
    """The property that makes a continuous register-pressure heuristic wrong."""
    assert allocate_vgprs(121, arch=RDNA4).waves_per_simd == (
        allocate_vgprs(144, arch=RDNA4).waves_per_simd
    )


# ── Headroom: the actionable output ─────────────────────────────────────────


def test_121_vgprs_is_one_register_above_a_rung_under_the_isa_granule():
    """The finding this model exists to surface.  Under granule 24 the 12-wave
    rung ends at 120, so a 121-VGPR kernel misses a 20% occupancy step by one
    register -- which a spill count cannot show."""
    shed, gained = headroom_to_next_rung(121, arch=RDNA4, granule=24)
    assert (shed, gained) == (1, 2)


def test_headroom_under_the_contested_granule_disagrees_and_that_is_the_point():
    """Granule 16 puts 121 already at 12 waves; the two candidates differ on
    the same kernel, which is exactly why the constant must be measured rather
    than chosen."""
    assert allocate_vgprs(121, arch=RDNA4, granule=16).waves_per_simd == 12
    assert allocate_vgprs(121, arch=RDNA4, granule=24).waves_per_simd == 10


def test_headroom_is_none_at_the_top_rung():
    top = occupancy_rungs(RDNA4)[0]
    assert headroom_to_next_rung(top.max_vgprs, arch=RDNA4) is None


def test_headroom_is_always_at_least_one_register():
    for vgprs in range(1, 257):
        result = headroom_to_next_rung(vgprs, arch=RDNA4)
        if result is not None:
            assert result[0] >= 1
            assert result[1] >= 1


# ── LDS: pool is not the request cap (ISA 12.1) ─────────────────────────────


def test_wgp_pool_is_twice_the_per_workgroup_request_cap_on_rdna():
    """ "128kB of memory per work-group processor [...] One work-group can
    request up to 64kB."  Dividing the cap instead of the pool halves every
    WGP-mode LDS occupancy figure."""
    cap = lds_request_cap_bytes(RDNA4)
    assert cap == 65536
    assert lds_pool_bytes(RDNA4, WorkgroupProcessorMode.WGP) == 2 * cap
    assert lds_pool_bytes(RDNA4, WorkgroupProcessorMode.CU) == cap


def test_an_lds_request_above_the_per_workgroup_cap_is_refused():
    with pytest.raises(TesseraOccupancyError, match="per-work-group cap"):
        estimate_occupancy(
            arch=RDNA4,
            mode=WorkgroupProcessorMode.WGP,
            vgprs=64,
            lds_bytes=65536 + 1,
            workgroup_threads=256,
        )


# ── Mode sensitivity ────────────────────────────────────────────────────────


def test_simds_per_slot_differs_by_mode_on_rdna():
    assert simds_per_slot(RDNA4, WorkgroupProcessorMode.WGP) == 4
    assert simds_per_slot(RDNA4, WorkgroupProcessorMode.CU) == 2


def test_mode_changes_an_lds_bound_verdict():
    """A verdict computed against the wrong mode is wrong by 2x -- the same
    failure ``dispatch_slots`` refuses to allow by defaulting the mode."""
    kwargs = dict(arch=RDNA4, vgprs=32, lds_bytes=32768, workgroup_threads=256)
    wgp = estimate_occupancy(mode=WorkgroupProcessorMode.WGP, **kwargs)
    cu = estimate_occupancy(mode=WorkgroupProcessorMode.CU, **kwargs)
    assert wgp.groups_per_slot == 2 * cu.groups_per_slot


# ── Composition ─────────────────────────────────────────────────────────────


def test_the_binding_limiter_is_named():
    heavy_regs = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=200,
        lds_bytes=1024,
        workgroup_threads=256,
    )
    assert heavy_regs.limiter == "vgpr"
    assert heavy_regs.waves_per_simd == heavy_regs.waves_by_vgpr

    heavy_lds = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.CU,
        vgprs=24,
        lds_bytes=65536,
        workgroup_threads=64,
    )
    assert "lds" in heavy_lds.limiter


def test_occupancy_never_exceeds_the_wave_slot_ceiling():
    slots = wave_slots_per_simd(RDNA4)
    tiny = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=8,
        lds_bytes=0,
        workgroup_threads=32,
    )
    assert tiny.waves_per_simd == slots
    assert tiny.utilization == 1.0


def test_a_partial_wave_is_refused_rather_than_silently_rounded():
    """A partial wave still occupies a whole slot; rounding it inside the
    occupancy model would hide a launch-shape bug."""
    with pytest.raises(TesseraOccupancyError, match="multiple of the wave size"):
        estimate_occupancy(
            arch=RDNA4,
            mode=WorkgroupProcessorMode.WGP,
            vgprs=64,
            lds_bytes=0,
            workgroup_threads=100,
        )


def test_metadata_dict_is_flat_and_carries_every_limiter():
    meta = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=121,
        lds_bytes=16384,
        workgroup_threads=256,
    ).as_metadata_dict()
    for key in (
        "waves_by_vgpr",
        "waves_by_lds",
        "waves_by_slots",
        "waves_per_simd",
        "limiter",
        "allocated_vgprs",
        "vgpr_granule",
    ):
        assert key in meta
    assert all(not isinstance(v, dict) for v in meta.values())


# ── Deliberately-unresolved facts, kept visible ─────────────────────────────


def test_both_fleet_rocm_parts_are_measured():
    """Each part was probed on its own silicon -- proof does not transfer
    between gfx1151 and gfx1201, so each needed its own run."""
    assert not granule_is_contested(RDNA4)
    assert not granule_is_contested(RDNA35)
    assert VGPR_GRANULE_CONTESTED == frozenset()
    for arch in (RDNA4, RDNA35):
        assert vgpr_regs_per_simd(arch) == 1536
        assert vgpr_alloc_granule(arch) == 24
        assert wave_slots_per_simd(arch) == 16
        assert wave_slots_provenance(arch) is Provenance.MEASURED


def test_contested_set_is_kept_as_the_gate_for_a_new_arch():
    """Empty today, but it is what an unmeasured arch passes through.  Deleting
    it would remove the only place that distinguishes an ISA-derived granule
    from a measured one."""
    assert isinstance(VGPR_GRANULE_CONTESTED, frozenset)
    assert not granule_is_contested(UNESTABLISHED)  # no granule at all
    assert vgpr_alloc_granule(UNESTABLISHED) is None


def test_max_waves_unit_is_reconciled_per_cu():
    """``_MAX_WAVES`` is per CU and now says so consistently for both families.

    It carried 16 for RDNA under the comment "Maximum waves per CU" while the
    CDNA rows were genuinely per-CU -- one column meaning two things.  An RDNA
    CU is two SIMD32s of 16 slots, so 32; a CDNA CU is four SIMDs of 8, also
    32.  This module now derives its per-SIMD figure from that one table
    instead of keeping a second copy of the constant.
    """
    from tessera.compiler.rocm_target import _MAX_WAVES, simds_per_cu

    for arch in (RDNA4, RDNA35):
        assert _MAX_WAVES[arch] == 32
        assert simds_per_cu(arch) == 2
        assert wave_slots_per_simd(arch) == _MAX_WAVES[arch] // simds_per_cu(arch)

    # CDNA was already per-CU and stays put; only the unit story changed.
    assert _MAX_WAVES[AMDArch.GFX_942] == 32
    assert simds_per_cu(AMDArch.GFX_942) == 4
    assert wave_slots_per_simd(AMDArch.GFX_942) == 8


def test_wave_slots_declines_when_cu_geometry_is_unestablished():
    """CDNA 5's CU geometry is not established on this fleet, so the derivation
    returns None rather than inventing a per-SIMD number from a per-CU one."""
    from tessera.compiler.rocm_target import simds_per_cu

    assert simds_per_cu(AMDArch.GFX_1250) is None
    assert wave_slots_per_simd(AMDArch.GFX_1250) is None
    assert wave_slots_provenance(AMDArch.GFX_1250) is Provenance.UNKNOWN


def test_waves_per_simd_property_no_longer_aliases_waves_per_cu():
    """Regression: ``ROCmTargetProfile.waves_per_simd`` used to return
    ``waves_per_cu`` unchanged, serving one number under two names that differ
    by the SIMD count."""
    from tessera.compiler.rocm_target import ROCmTargetProfile

    profile = ROCmTargetProfile(arch=RDNA4, waves_per_cu=8)
    assert profile.waves_per_simd == 4  # 8 per CU / 2 SIMDs
    assert profile.waves_per_simd != profile.waves_per_cu

    cdna = ROCmTargetProfile(arch=AMDArch.GFX_942, waves_per_cu=8)
    assert cdna.waves_per_simd == 2  # 8 per CU / 4 SIMDs


# ── The device probe's solver, exercised without a device ───────────────────
#
# The probe itself needs hipcc and the part; its *inference* does not, and the
# inference is where it could be wrong.  Covering it here means a broken solver
# fails on any box rather than only on Tajasarus.


def _probe_module():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "probe_rdna_vgpr_granule.py"
    )
    spec = importlib.util.spec_from_file_location("probe_rdna_vgpr_granule", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "truth", [MEASURED_GFX1201, (1536, 16, 16), (1024, 8, 12), (2048, 32, 20)]
)
def test_probe_recovers_a_known_device_uniquely(truth):
    """Round-trip: synthesize what a device with known constants would report,
    and confirm the solver names exactly that device."""
    probe = _probe_module()
    observations = [
        {"status": "ok", "vgprs": v, "occupancy": probe._predict(v, *truth)}
        for v in (32, 64, 100, 121, 150, 200, 256)
    ]
    assert probe.solve(observations) == [truth]


def test_the_repo_recorded_occupancy_pair_is_inconsistent_with_the_device():
    """The ROCm queue records a split-wave attention kernel at 256 VGPRs / 6
    waves and 121 VGPRs / 12 waves.  The measured gfx1201 device -- granule 24,
    rung boundary at 120 -- gives 10 waves at 121, so no device consistent with
    the recorded pair has the granule the hardware actually has.  The record is
    the wrong party, and this test keeps that visible rather than quietly
    dropping the inconsistent number."""
    probe = _probe_module()
    recorded = [
        {"status": "ok", "vgprs": 256, "occupancy": 6},
        {"status": "ok", "vgprs": 121, "occupancy": 12},
    ]
    candidates = probe.solve(recorded)
    assert candidates, "the recorded pair should be explicable by some device"
    assert MEASURED_GFX1201 not in candidates
    assert 24 not in {g for _, g, _ in candidates}


def test_probe_reports_no_solution_rather_than_forcing_a_fit():
    probe = _probe_module()
    impossible = [
        {"status": "ok", "vgprs": 64, "occupancy": 7},
        {"status": "ok", "vgprs": 64, "occupancy": 9},
    ]
    assert probe.solve(impossible) == []


def test_probe_names_a_separating_measurement_when_ambiguous():
    """An ambiguous run must say what to measure next, not just shrug."""
    probe = _probe_module()
    candidates = [(1536, 16, 16), (1536, 8, 16)]
    separators = probe.separating_register_counts(candidates)
    assert separators
    for v in separators[:5]:
        assert probe._predict(v, 1536, 16, 16) != probe._predict(v, 1536, 8, 16)


def test_probe_solver_ignores_failed_compilations():
    probe = _probe_module()
    observations = [
        {"status": "no-remark", "depth": 4},
        {
            "status": "ok",
            "vgprs": 121,
            "occupancy": probe._predict(121, *MEASURED_GFX1201),
        },
        {
            "status": "ok",
            "vgprs": 200,
            "occupancy": probe._predict(200, *MEASURED_GFX1201),
        },
        {"status": "timeout", "depth": 96},
    ]
    assert MEASURED_GFX1201 in probe.solve(observations)


def test_probe_solver_declines_when_every_compilation_failed():
    probe = _probe_module()
    assert probe.solve([{"status": "no-remark"}, {"status": "timeout"}]) == []


# ── Locked to the device ────────────────────────────────────────────────────

#: Every ``(VGPRs, occupancy)`` pair the compiler's resource-usage remark
#: reported on Tajasarus, 2026-09-20.  Source:
#: ``benchmarks/baselines/gfx1201_vgpr_granule_20260920/probe.json``.
GFX1201_DEVICE_OBSERVATIONS = (
    (14, 16),
    (24, 16),
    (31, 16),
    (39, 16),
    (47, 16),
    (62, 16),
    (78, 16),
    (111, 12),
    (132, 10),
    (164, 9),
    (196, 7),
    (228, 6),
)


@pytest.mark.parametrize("vgprs,expected", GFX1201_DEVICE_OBSERVATIONS)
def test_model_reproduces_the_gfx1201_device_measurement(vgprs, expected):
    """The model was written from the ISA before the probe ran and reproduces
    all twelve points with nothing fitted to them.  This is the test that makes
    the measurement load-bearing: change a constant and it fails here rather
    than silently shifting every RDNA4 register target.

    It runs on any host -- the numbers came from gfx1201, the arithmetic does
    not need it.
    """
    result = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=vgprs,
        lds_bytes=0,
        workgroup_threads=256,
    )
    assert result.waves_per_simd == expected


def test_the_measured_rung_boundary_is_120_not_128():
    """The concrete consequence, and the reason the granule mattered: the
    device gives 12 waves at 111 VGPRs and 10 at 132, so the 12-wave rung ends
    at 120.  A 121-VGPR kernel is one register above a 20% occupancy step."""
    rungs = {r.waves_per_simd: r.max_vgprs for r in occupancy_rungs(RDNA4)}
    assert rungs[12] == 120
    assert allocate_vgprs(120, arch=RDNA4).waves_per_simd == 12
    assert allocate_vgprs(121, arch=RDNA4).waves_per_simd == 10
    assert headroom_to_next_rung(121, arch=RDNA4) == (1, 2)


def test_a_verdict_on_the_measured_part_is_not_flagged_contested():
    result = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=121,
        lds_bytes=16384,
        workgroup_threads=256,
    )
    assert result.granule_contested is False


# ── Ceilings taken from the ISA PDF, 2026-09-20 ─────────────────────────────


def test_lds_is_allocated_in_1kib_blocks():
    """ISA 3.3.5: "allocated in blocks of 1024 bytes".  Dividing the pool by an
    unrounded request overstates residency -- a 26000 B request occupies
    26624 B, so four groups fit on a WGP, not the five naive division gives."""
    result = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=16,
        lds_bytes=26000,
        workgroup_threads=256,
    )
    assert result.lds_allocated_bytes == 26624
    assert result.groups_per_slot == 4
    assert lds_pool_bytes(RDNA4, WorkgroupProcessorMode.WGP) // 26000 == 5


def test_an_already_aligned_lds_request_is_not_padded():
    result = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=16,
        lds_bytes=32768,
        workgroup_threads=256,
    )
    assert result.lds_allocated_bytes == 32768


def test_a_workgroup_above_1024_work_items_is_refused():
    """ISA 2.3: "a maximum of 1024 work-items per work-group"."""
    with pytest.raises(TesseraOccupancyError, match="work-item limit"):
        estimate_occupancy(
            arch=RDNA4,
            mode=WorkgroupProcessorMode.WGP,
            vgprs=16,
            lds_bytes=0,
            workgroup_threads=1056,
        )


def test_workgroup_ceiling_is_currently_coincident_with_wave_slots():
    """``MAX_WORKGROUPS_PER_WGP`` is applied but cannot change any answer on a
    part with 16 wave slots per SIMD: 4 SIMDs x 16 slots = 64 waves, and a
    non-exempt group is >= 2 waves, so residency caps at 32 either way.  Kept
    because the coincidence is arithmetic rather than a law, and asserted here
    so it is never mistaken for a live constraint (Decision #29a).
    """
    slots = wave_slots_per_simd(RDNA4)
    assert slots is not None
    assert slots * simds_per_slot(RDNA4, WorkgroupProcessorMode.WGP) // 2 == (
        MAX_WORKGROUPS_PER_WGP
    )
    for threads in range(64, MAX_WORKITEMS_PER_WORKGROUP + 1, 32):
        result = estimate_occupancy(
            arch=RDNA4,
            mode=WorkgroupProcessorMode.WGP,
            vgprs=8,
            lds_bytes=0,
            workgroup_threads=threads,
        )
        uncapped = (
            result.waves_per_simd * simds_per_slot(RDNA4, WorkgroupProcessorMode.WGP)
        ) // result.waves_per_group
        assert uncapped <= MAX_WORKGROUPS_PER_WGP


#: Princess-Luna, 2026-09-20 -- gfx1151's own silicon.  Source:
#: ``benchmarks/baselines/gfx1151_vgpr_granule_20260920/probe.json``.
GFX1151_DEVICE_OBSERVATIONS = (
    (14, 16),
    (25, 16),
    (31, 16),
    (39, 16),
    (47, 16),
    (62, 16),
    (78, 16),
    (111, 12),
    (132, 10),
    (164, 9),
    (196, 7),
    (228, 6),
)


@pytest.mark.parametrize("vgprs,expected", GFX1151_DEVICE_OBSERVATIONS)
def test_model_reproduces_the_gfx1151_device_measurement(vgprs, expected):
    """gfx1151 measured independently of gfx1201.  The two parts agree, but
    agreement is the result, not the assumption."""
    result = estimate_occupancy(
        arch=RDNA35,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=vgprs,
        lds_bytes=0,
        workgroup_threads=256,
    )
    assert result.waves_per_simd == expected


# ── CDNA5 (gfx125x): a GFX12-derived part with different constants ──────────
#
# Verified against the CDNA5 ISA manual, 2026-09-20.  These exist because the
# model's RDNA-shaped constants would be silently wrong here, not because
# CDNA5 occupancy is claimed -- its VGPR file per SIMD and CU geometry are
# still unestablished, so a verdict declines.

CDNA5 = AMDArch.GFX_1250


def test_cdna5_lds_granule_is_2048_not_1024():
    """CDNA5 ISA 3.3.4: "LDS space is allocated in blocks of 2048 bytes",
    against 1024 on RDNA3.5/4.  A single module-wide constant would be wrong
    by 2x here."""
    assert lds_alloc_granule(CDNA5) == 2048
    assert lds_alloc_granule(RDNA4) == 1024
    assert lds_alloc_granule(RDNA35) == 1024


def test_cdna1_to_4_lds_granule_is_unestablished_not_inherited():
    """Absent rather than defaulted: inheriting RDNA's block size on a CDNA
    part would understate the allocation."""
    assert lds_alloc_granule(UNESTABLISHED) is None


def test_cdna5_per_wave_vgpr_cap_is_1024():
    """CDNA5 ISA 3.3.2.1: "a shader may have up to 1024 VGPRs", reached above
    255 through VGPR-MSB indexing.  A hardcoded 256 would refuse a legal
    CDNA5 wave."""
    assert per_wave_vgpr_cap(CDNA5) == 1024
    assert per_wave_vgpr_cap(RDNA4) == 256


def test_cdna5_granule_is_16_and_stated_not_derived():
    """The "devices that have 1536 VGPRs per SIMD allocate in blocks of 24"
    sentence is in the RDNA3.5 and RDNA4 manuals and **not** in CDNA5's, whose
    3.3.2.1 says only "VGPRs are allocated in blocks of 16 for wave32".

    So the granule is known for gfx125x even though its VGPR file size is not.
    A derivation keyed on file size would throw that away.
    """
    assert vgpr_alloc_granule(CDNA5) == 16
    assert vgpr_regs_per_simd(CDNA5) is None


def test_cdna5_states_no_wave64_granule():
    """CDNA5 compute waves are wave32; its 3.3.2.1 carries no wave64 form,
    unlike RDNA's "8 for wave64" / "12 for wave64"."""
    assert vgpr_alloc_granule(CDNA5, wave_size=WAVE64) is None
    assert vgpr_alloc_granule(RDNA4, wave_size=WAVE64) == 12


def test_a_known_granule_still_does_not_yield_an_occupancy_verdict():
    """A granule alone cannot give a wave count -- the file size is the
    numerator, and it is unmeasured on gfx125x."""
    with pytest.raises(TesseraOccupancyError, match="VGPR file size"):
        allocate_vgprs(64, arch=CDNA5, granule=16)


def test_cdna5_occupancy_declines_rather_than_guessing():
    """Its CU geometry and VGPR file are unestablished on this fleet."""
    assert wave_slots_per_simd(CDNA5) is None
    with pytest.raises(TesseraOccupancyError):
        allocate_vgprs(64, arch=CDNA5)


def test_a_caller_supplied_cap_overrides_the_arch_default():
    alloc = allocate_vgprs(300, arch=RDNA4, per_wave_cap=1024, granule=16)
    assert alloc.allocated == 304


# ── SIMD tier is per-arch, not inferred from "is this RDNA?" ────────────────


def test_cdna5_wgp_has_four_simds_not_two():
    """Regression for a silent 2x.  ``simds_per_slot`` used to send every
    non-RDNA arch down a CU-shaped branch returning 2, but CDNA5 ISA 2.2 says
    a work-group's waves "can run on any of the 4 SIMD32s" of its WGP.  gfx1251
    is a far larger part than gfx1201, which is exactly where a fallback shaped
    by the small part goes unnoticed."""
    assert simds_per_slot(CDNA5, WorkgroupProcessorMode.WGP) == 4
    assert simds_per_slot(RDNA4, WorkgroupProcessorMode.WGP) == 4


def test_cdna5_has_no_established_cu_tier():
    """Its manual describes only the WGP tier, so CU mode declines."""
    assert simds_per_slot(CDNA5, WorkgroupProcessorMode.CU) is None


def test_cdna1_to_4_have_no_wgp_tier():
    """A dispatch slot there is a CU of 4 SIMDs; asking for a WGP declines
    rather than borrowing RDNA's shape."""
    assert simds_per_slot(UNESTABLISHED, WorkgroupProcessorMode.WGP) is None
    assert simds_per_slot(UNESTABLISHED, WorkgroupProcessorMode.CU) == 4


def test_lds_pool_declines_where_the_manual_states_no_separate_pool():
    """CDNA5 gives a 320 kB per-work-group maximum and no distinct per-WGP
    pool.  Assuming pool == cap would silently cap gfx1251 residency at one
    work-group for any LDS-heavy kernel."""
    assert lds_request_cap_bytes(CDNA5) == 327680
    assert lds_pool_bytes(CDNA5, WorkgroupProcessorMode.WGP) is None
    assert lds_pool_bytes(UNESTABLISHED, WorkgroupProcessorMode.CU) is None


def test_occupancy_refuses_rather_than_guessing_an_unestablished_tier():
    with pytest.raises(TesseraOccupancyError, match="SIMDs per"):
        estimate_occupancy(
            arch=UNESTABLISHED,
            mode=WorkgroupProcessorMode.WGP,
            vgprs=64,
            lds_bytes=0,
            workgroup_threads=256,
        )


def test_cdna5_shares_rdna_workgroup_limits():
    """CDNA5 ISA 2.2 states the same per-WGP limits as RDNA4/3.5 ISA 2.3 --
    32 work-groups, 1024 work-items, single-wave groups exempt -- so applying
    those two constants arch-independently is verified, not assumed."""
    assert MAX_WORKGROUPS_PER_WGP == 32
    assert MAX_WORKITEMS_PER_WORKGROUP == 1024


# ── Review findings, PR #789 ────────────────────────────────────────────────


def test_wave64_registers_cost_twice_the_storage():
    """A VGPR is one dword per lane, so a wave64 register is 2x a wave32 one.
    ISA 3.3.2.1 gives the block in dwords -- "16*32 or 8*64 = 512 DWORDs", and
    24*32 = 12*64 on a 1536-register part -- so the block is the same physical
    size while the register count filling it differs by 2x.  Dividing the
    file's wave32 count by a wave64 count overstated occupancy by 2x."""
    w32 = allocate_vgprs(121, arch=RDNA4, wave_size=WAVE32)
    w64 = allocate_vgprs(121, arch=RDNA4, wave_size=WAVE64)
    assert (w32.granule, w32.allocated, w32.waves_per_simd) == (24, 144, 10)
    assert (w64.granule, w64.allocated) == (12, 132)
    assert w64.waves_per_simd == 5  # not 1536 // 132 == 11


def test_a_block_is_the_same_physical_size_in_either_wave_mode():
    """The invariant behind the fix: equal dword footprints occupy equally."""
    for vgprs32 in (24, 48, 96, 240):
        w32 = allocate_vgprs(vgprs32, arch=RDNA4, wave_size=WAVE32)
        w64 = allocate_vgprs(vgprs32 // 2, arch=RDNA4, wave_size=WAVE64)
        assert w32.allocated * WAVE32 == w64.allocated * WAVE64
        assert w32.waves_per_simd == w64.waves_per_simd


def test_wave64_rungs_are_half_the_wave32_register_budgets():
    w32 = {r.waves_per_simd: r.max_vgprs for r in occupancy_rungs(RDNA4)}
    w64 = {
        r.waves_per_simd: r.max_vgprs for r in occupancy_rungs(RDNA4, wave_size=WAVE64)
    }
    for waves in (12, 10, 8):
        assert w64[waves] * 2 == w32[waves]


def test_lds_bound_residency_does_not_lose_groups_to_per_simd_rounding():
    """Three 2-wave groups on four SIMDs is six waves.  Computing a per-SIMD
    ceiling first and multiplying back floors 6//4 to 1 and reports two
    groups, discarding a runnable one."""
    result = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=32,
        lds_bytes=40 * 1024,
        workgroup_threads=64,
    )
    pool = lds_pool_bytes(RDNA4, WorkgroupProcessorMode.WGP)
    assert pool is not None and pool // (40 * 1024) == 3
    assert result.groups_per_slot == 3
    assert result.limiter == "lds"
    # Waves distribute unevenly (2,2,1,1); the reported per-SIMD figure is the
    # busiest SIMD's, because that is the register file that binds.
    assert result.waves_per_simd == 2


def test_residency_never_exceeds_any_single_ceiling():
    """Property check across a grid: groups actually fit."""
    for threads in (32, 64, 128, 256, 512):
        for lds in (0, 4096, 16384, 40 * 1024, 64 * 1024):
            for vgprs in (24, 64, 121, 200):
                r = estimate_occupancy(
                    arch=RDNA4,
                    mode=WorkgroupProcessorMode.WGP,
                    vgprs=vgprs,
                    lds_bytes=lds,
                    workgroup_threads=threads,
                )
                waves = r.groups_per_slot * r.waves_per_group
                slots = wave_slots_per_simd(RDNA4)
                assert slots is not None
                assert waves <= slots * 4
                assert r.waves_per_simd <= min(r.waves_by_vgpr, slots)
                if lds:
                    pool = lds_pool_bytes(RDNA4, WorkgroupProcessorMode.WGP)
                    assert pool is not None
                    assert r.groups_per_slot * r.lds_allocated_bytes <= pool


# ── Closure evidence, 2026-09-21 ────────────────────────────────────────────


def test_wave_slots_match_the_hsa_runtime_report():
    """`rocminfo` on both parts reports `Max Waves Per CU: 32` and
    `SIMDs per CU: 2` -- 16 waves/SIMD, from the runtime rather than the
    compiler.  This is the second of three independent sources; the ISA
    manuals state no wave-slot count at all."""
    from tessera.compiler.rocm_target import _MAX_WAVES, simds_per_cu

    for arch in (RDNA4, RDNA35):
        assert _MAX_WAVES[arch] == 32  # rocminfo: Max Waves Per CU
        assert simds_per_cu(arch) == 2  # rocminfo: SIMDs per CU
        assert wave_slots_per_simd(arch) == 16


def test_wave_slots_match_llvm_get_max_waves_per_eu():
    """LLVM's own constant, `AMDGPUBaseInfo.cpp::getMaxWavesPerEU`:
    ``isGFX90A ? 8 : (!isGFX10Plus ? 10 : (hasGFX10_3Insts ? 16 : 20))``.
    gfx11/gfx12 take the 16 branch; gfx90a/942/950 take the 8 branch, which a
    compiler probe confirms for all three.  Third independent source."""
    llvm_waves_per_eu = {
        RDNA4: 16,
        RDNA35: 16,
        AMDArch.GFX_942: 8,
        AMDArch.GFX_90A: 8,
        AMDArch.GFX_950: 8,
    }
    for arch, expected in llvm_waves_per_eu.items():
        assert wave_slots_per_simd(arch) == expected, arch.name


def test_occupancy_is_not_claimed_to_predict_performance():
    """Guard for the finding that closed this sync key: on gfx1151, +20%
    occupancy cost -55% throughput.  Nothing in this module may present a wave
    count as a speed, and no consumer may rank on it.

    Asserted structurally rather than in prose: the result object exposes
    capacity fields only.  A `score`, `rank`, `faster`, `speedup` or `latency`
    field appearing here would mean someone had started ranking on occupancy,
    which the device measurement says is wrong.
    """
    meta = estimate_occupancy(
        arch=RDNA4,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=121,
        lds_bytes=16384,
        workgroup_threads=256,
    ).as_metadata_dict()
    forbidden = (
        "score",
        "rank",
        "faster",
        "speedup",
        "latency",
        "ms",
        "tflops",
        "throughput",
    )
    assert not [k for k in meta if any(f in k.lower() for f in forbidden)]


def test_the_measured_g6_rung_crossing_is_representable():
    """The two points the closure experiment measured, as the model sees them:
    121 VGPRs is 10 waves/SIMD, 113 is 12, and the step is real.  The device
    then showed the 12-wave build 2.2x slower -- which is why this is the last
    test that touches those numbers, and none of them rank anything."""
    assert allocate_vgprs(121, arch=RDNA35).waves_per_simd == 10
    assert allocate_vgprs(113, arch=RDNA35).waves_per_simd == 12
    assert headroom_to_next_rung(121, arch=RDNA35) == (1, 2)


# ── The input trap that produced a wrong published claim (PR #791) ──────────


def test_launch_geometry_not_max_flat_workgroup_size_decides_the_limiter():
    """Feeding the model a *maximum* where it wants the *launch* size inverts
    which limiter binds.  This is the exact error behind PR #791's retracted
    "occupancy-bound" claim.

    `fa_dkdv` carries `max_flat_workgroup_size = 256` in its metadata but
    `runtime.py` launches it with `block=32` -- one wave per work-group.  With
    256 the verdict reads `vgpr`-limited and a register change looks like it
    moves residency; with the real 32 it reads `lds`-limited at 7 groups and
    the register change moves nothing.
    """
    wrong = estimate_occupancy(
        arch=RDNA35,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=209,
        lds_bytes=17408,
        workgroup_threads=256,
    )
    right = estimate_occupancy(
        arch=RDNA35,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=209,
        lds_bytes=17408,
        workgroup_threads=32,
    )
    assert wrong.limiter == "vgpr"
    assert right.limiter == "lds"
    assert right.groups_per_slot == 7

    # And with the real geometry the register lever cannot move residency.
    shed = estimate_occupancy(
        arch=RDNA35,
        mode=WorkgroupProcessorMode.WGP,
        vgprs=192,
        lds_bytes=17408,
        workgroup_threads=32,
    )
    assert shed.groups_per_slot == right.groups_per_slot == 7
    assert shed.waves_by_vgpr > right.waves_by_vgpr  # ceiling moved...
    assert shed.limiter == "lds"  # ...but was never binding


def test_attention_kernels_are_lds_bound_on_both_parts():
    """With one wave per work-group the register ceiling sits well above the
    LDS ceiling for every attention kernel measured, on gfx1151 and gfx1201
    alike.  Measured VGPR/LDS pairs from
    `benchmarks/baselines/rocm_mlp_correction_20260921/`."""
    measured = [
        (RDNA35, 209, 17408),
        (RDNA35, 218, 9408),
        (RDNA35, 122, 9216),
        (RDNA4, 149, 17408),
        (RDNA4, 211, 9408),
        (RDNA4, 94, 9216),
    ]
    for arch, vgprs, lds in measured:
        r = estimate_occupancy(
            arch=arch,
            mode=WorkgroupProcessorMode.WGP,
            vgprs=vgprs,
            lds_bytes=lds,
            workgroup_threads=32,
        )
        assert r.limiter == "lds", (arch.name, vgprs, lds, r.limiter)
        assert r.waves_by_vgpr > r.waves_by_lds
