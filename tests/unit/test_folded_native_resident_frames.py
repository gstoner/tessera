"""Benchmark ownership failures must not leak HIP roots or modules."""
from ctypes import c_void_p
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm import ablate_gfx1201_folded_load_schedule as ablate
from benchmarks.rocm.folded_native_resident_frame import FoldedNativeFramePool
from benchmarks.rocm.record_gfx1201_folded_native_package import matching_reference_schedule
from tessera.compiler.rocm_mxfp4_folded import FoldedPrefillSchedule


class Copy:
    def __init__(self):
        self.device = [c_void_p(1234)]
        self.closed = 0

    def close(self):
        self.closed += 1


def test_shared_frames_remain_alive_until_all_engines_close():
    copies = [Copy(), Copy(), Copy()]
    pool = FoldedNativeFramePool(lambda arrays, count: copies)
    arrays = (np.zeros((2, 3), dtype=np.uint8),)
    first = pool.acquire(arrays, 3)
    second = pool.acquire(arrays, 3)
    assert first.copies is second.copies
    with pytest.raises(RuntimeError, match="live engine"):
        pool.close()
    first.close()
    first.close()
    assert pool.report["leases"] == 1
    assert all(c.closed == 0 for c in copies)
    second.close()
    pool.close()
    pool.close()
    assert all(c.closed == 1 for c in copies)
    assert pool.report["closed"] and pool.report["leases"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire(arrays, 3)


@pytest.mark.parametrize("different_count", [False, True])
def test_shared_frames_reject_other_roots_or_rotation_count(different_count):
    arrays = (np.zeros((2, 3), dtype=np.uint8),)
    pool = FoldedNativeFramePool(lambda arrays, count: [Copy()])
    lease = pool.acquire(arrays, 3)
    with pytest.raises(ValueError, match="roots or copy count"):
        pool.acquire(arrays if different_count else (arrays[0].copy(),),
                     2 if different_count else 3)
    lease.close()
    pool.close()


def test_failed_allocation_can_be_retried_without_live_lease():
    attempts = []
    def allocate(arrays, count):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("allocation failed")
        return [Copy()]
    pool = FoldedNativeFramePool(allocate)
    arrays = (np.zeros(2, dtype=np.uint8),)
    with pytest.raises(RuntimeError, match="allocation failed"):
        pool.acquire(arrays, 1)
    assert pool.report["leases"] == 0
    lease = pool.acquire(arrays, 1)
    lease.close()
    pool.close()


def test_partial_rotation_allocation_releases_prior_copies(monkeypatch):
    allocated = []
    def allocate(hip, arrays):
        if len(allocated) == 2:
            raise RuntimeError("out of memory")
        copy = Copy()
        allocated.append(copy)
        return copy
    monkeypatch.setattr(base, "_DeviceArrays", allocate)
    with pytest.raises(RuntimeError, match="out of memory"):
        base._copies(None, (np.ones(2), np.ones(2)), 3)
    assert [copy.closed for copy in allocated] == [1, 1]


def test_workgroup_comparison_permits_only_requested_mode_change():
    schedule = FoldedPrefillSchedule()
    other = replace(schedule, workgroup_mode="cu")
    assert not matching_reference_schedule(other, schedule)
    assert matching_reference_schedule(other, schedule, compare_workgroup_mode=True)
    # Geometry and input staging must remain equal even when mode comparison is enabled.
    assert not matching_reference_schedule(
        replace(other, raster_group_m=2 if schedule.raster_group_m != 2 else 1), schedule,
        compare_workgroup_mode=True)


def test_metadata_failure_releases_lease_and_unloads_module(monkeypatch):
    class Hip:
        unloaded = 0
        def hipModuleLoadData(self, *args): return 0
        def hipModuleGetFunction(self, *args): return 0
        def hipModuleUnload(self, *args): self.unloaded += 1
    hip = Hip()
    copies = [Copy()]
    pool = FoldedNativeFramePool(lambda arrays, count: copies)
    def fail(*args):
        raise RuntimeError("ISA extraction failed")
    monkeypatch.setattr(ablate, "selected_symbol_isa_evidence", fail)
    descriptor = SimpleNamespace(
        entry_symbol="entry", abi_id="abi",
        geometry=SimpleNamespace(grid=(1, 1, 1), workgroup=(256, 1, 1)),
        provenance={"kernel_argument_layout": "expanded_memref"})
    package = SimpleNamespace(image=SimpleNamespace(payload=b"image"), descriptor=descriptor)
    with pytest.raises(RuntimeError, match="ISA extraction"):
        ablate.package_engine(hip, base.Case("prefill", 256, 64, 64),
            {"a": np.zeros((256, 64)), "a_scale": np.ones(256),
             "output": np.zeros((256, 64))},
            SimpleNamespace(weight_bytes=np.zeros((64, 32)),
                            row_reference=np.ones(64)),
            package, name="native", metadata={}, copies=1, resident_pool=pool)
    assert hip.unloaded == 1 and pool.report["leases"] == 0
    pool.close()
    assert copies[0].closed == 1


def test_shared_download_destinations_keep_independent_correctness_snapshots():
    from benchmarks.rocm.record_gfx1201_folded_native_package import independent_output_snapshots
    host = np.zeros(2, dtype=np.uint16)
    def output(value):
        host[:] = value
        return host
    engines = [SimpleNamespace(name="first", output=lambda: output(1)),
               SimpleNamespace(name="second", output=lambda: output(2))]
    snapshots = independent_output_snapshots(engines)
    np.testing.assert_array_equal(snapshots["first"], [1, 1])
    np.testing.assert_array_equal(snapshots["second"], [2, 2])
    assert not np.shares_memory(snapshots["first"], snapshots["second"])
