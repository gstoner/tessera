import copy
import pytest
from tessera.compiler.profiler_cuda_window import build_cuda_window_calibration


def arguments():
    packet = dict(
        backend="nvidia", process_id=100, run_id="1"*32,
        architecture="sm_120",
        compiler_sha256="a" * 64,
        shape=[32, 2, 16, 4],
        clock="CUDA events",
        cooperative=True,
        execution="native_gpu",
        rows=[dict(chunk=8, binding_digest="b" * 64, image_sha256="c" * 64, device_event_ms=[1.0] * 7)],
    )
    kernels = [
        dict(start=i * 1000000, end=(i + 1) * 1000000, device=0, context=1, stream=0, name="product")
        for i in range(701)
    ]
    profiled = copy.deepcopy(packet)
    profiled.update(process_id=101,run_id="2"*32)
    return dict(
        clean=packet,
        profiled=profiled,
        kernels=kernels,
        source=dict(source_commit="d" * 40, worktree_dirty=False, execution_environment="bare_metal"),
        capture_device=dict(device=0, architecture="sm_120", uuid="device-uuid",process_id=101),
        capture_sha256="e" * 64,
        sample_id="sample",
    )


def test_cuda_window_calibration_recomputes_clock_and_overhead():
    args = arguments()
    report = build_cuda_window_calibration(**args)
    assert report["eligible_for_promotion"]
    assert report["maximum_clock_relative_error"] == 0
    args["profiled"]["rows"][0]["device_event_ms"] = [2.0] * 7
    report = build_cuda_window_calibration(**args)
    assert not report["eligible_for_promotion"]
    assert report["instrumentation_overhead"] == 2.0
    assert len(report["ineligibility_reasons"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["kernels"].pop(),
        lambda a: a["capture_device"].update(architecture="sm_90"),
        lambda a: a["kernels"][5].update(stream=2),
        lambda a: a["kernels"][5].update(start=-1),
        lambda a: a["clean"]["rows"][0].update(device_event_ms=[True] * 7),
        lambda a: a["profiled"]["rows"][0].update(image_sha256="changed"),
        lambda a: a["source"].update(worktree_dirty="false"),
    ],
)
def test_cuda_calibration_refuses_malformed_input(mutation):
    args = arguments()
    mutation(args)
    with pytest.raises(ValueError):
        build_cuda_window_calibration(**args)


def test_cuda_selector_rebuilds_and_refuses_reused_calibration():
    from tessera.compiler.ssd_performance import _admit_cuda_windows

    packets = []
    pairs = []
    for i in range(9):
        pair = {}
        for j, name in enumerate(("serial", "cooperative")):
            args = arguments()
            args["sample_id"] = str(2 * i + j)
            args["clean"]["run_id"] = f"{4*i+2*j:032x}"
            args["profiled"]["run_id"] = f"{4*i+2*j+1:032x}"
            args["capture_sha256"] = f"{2 * i + j:064x}"
            packet = build_cuda_window_calibration(**args)
            packets.append(packet)
            pair[name] = args["clean"]
        pairs.append(pair)
    assert _admit_cuda_windows(dict(pairs=pairs), packets, 2.0).admitted
    packets[-1]["capture_sha256"] = packets[0]["capture_sha256"]
    with pytest.raises(ValueError, match="reused"):
        _admit_cuda_windows(dict(pairs=pairs), packets, 2.0)
    packets[-1]["capture_sha256"] = f"{17:064x}"
    packets[0]["source"]["worktree_dirty"] = True
    packets[0]["eligible_for_promotion"] = True
    assert not _admit_cuda_windows(dict(pairs=pairs), packets, 2.0).admitted


def test_production_collection_preflight_requires_eligible_host():
    from benchmarks.record_ssd_calibrated_pairs import require_eligible_host
    require_eligible_host(dict(worktree_dirty=False,execution_environment='wsl2'))
    with pytest.raises(ValueError,match='unknown execution environment'):
        require_eligible_host(dict(worktree_dirty=False,execution_environment='container'))
    with pytest.raises(ValueError,match='uncommitted'):
        require_eligible_host(dict(worktree_dirty=True,execution_environment='bare_metal'))


def test_cuda_calibration_refuses_foreign_capture_process():
    args = arguments()
    args['capture_device']['process_id'] += 1
    with pytest.raises(ValueError,match='another profiled process'):
        build_cuda_window_calibration(**args)


def test_wsl2_cuda_window_is_admissible_when_its_clocks_agree():
    """Owner direction 2026-09-25: the activity-window / event agreement is
    the witness; WSL2 alone no longer refuses."""
    args = arguments()
    args["source"]["execution_environment"] = "wsl2"
    report = build_cuda_window_calibration(**args)
    assert report["eligible_for_promotion"], report["ineligibility_reasons"]
    args["profiled"]["rows"][0]["device_event_ms"] = [2.0] * 7
    assert not build_cuda_window_calibration(**args)["eligible_for_promotion"]
