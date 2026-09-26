"""Calibrate launch-inclusive CUDA event windows against Nsight GPU timelines.

This bounded adapter consumes one checked warm-up and seven 100-launch windows.
It never substitutes the sum of kernel durations for the launch-inclusive span.
Promotion also requires clean source, event/activity-window agreement within
5% and <=5% trace overhead, on bare metal or WSL2 alike (owner direction,
MASTER_AUDIT 2026-09-25 -- recorded there as an explicit NVIDIA exception,
since this witness is profiler-derived).
"""

import hashlib
import math
import re
import statistics


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def build_cuda_window_calibration(*, clean, profiled, kernels, source, capture_device, capture_sha256, sample_id):
    if (
        not isinstance(sample_id, str)
        or not sample_id
        or not isinstance(capture_sha256, str)
        or not re.fullmatch("[0-9a-f]{64}", capture_sha256)
    ):
        raise ValueError("CUDA calibration requires sample and capture identity")
    for packet in (clean,profiled):
        if (type(packet.get('process_id')) is not int or packet['process_id'] <= 0
                or not isinstance(packet.get('run_id'),str) or not re.fullmatch('[0-9a-f]{32}',packet['run_id'])):
            raise ValueError('CUDA calibration requires process identity and run nonce')
    if clean['run_id'] == profiled['run_id']:
        raise ValueError('clean and profiled CUDA runs must be distinct')
    if capture_device.get('process_id') != profiled['process_id']:
        raise ValueError('CUDA capture belongs to another profiled process')
    keys = ("backend", "architecture", "compiler_sha256", "shape", "clock", "cooperative", "execution")
    if (
        any(clean.get(k) != profiled.get(k) for k in keys)
        or clean.get("backend") != "nvidia"
        or clean.get("clock") != "CUDA events"
        or clean.get("execution") != "native_gpu"
    ):
        raise ValueError("CUDA calibration workload identity disagrees")
    if len(clean["rows"]) != 1 or len(profiled["rows"]) != 1:
        raise ValueError("CUDA calibration requires one artifact")
    if (
        type(capture_device.get("device")) is not int
        or capture_device.get("architecture") != clean["architecture"]
        or not isinstance(capture_device.get("uuid"), str)
        or not capture_device["uuid"]
    ):
        raise ValueError("CUDA capture device disagrees with measured architecture")
    a, b = clean["rows"][0], profiled["rows"][0]
    if any(a.get(k) != b.get(k) for k in ("chunk", "binding_digest", "image_sha256")):
        raise ValueError("CUDA calibration image or checkpoint policy disagrees")
    for row in (a, b):
        if len(row["device_event_ms"]) != 7 or any(not _positive(v) for v in row["device_event_ms"]):
            raise ValueError("CUDA calibration requires seven finite event windows")
    if len(kernels) != 701:
        raise ValueError("CUDA calibration requires one warm-up and 700 measured kernels")
    previous = -1
    identities = set()
    for row in kernels:
        if (
            any(type(row.get(k)) is not int for k in ("start", "end", "device", "context", "stream"))
            or row["start"] < previous
            or row["end"] <= row["start"]
            or row.get("name") != "product"
        ):
            raise ValueError("CUDA calibration requires ordered nonoverlapping product kernels")
        previous = row["end"]
        identities.add((row["device"], row["context"], row["stream"]))
    if len(identities) != 1 or next(iter(identities))[0] != capture_device["device"]:
        raise ValueError("CUDA calibration spans different device contexts or streams")
    spans, active, errors = [], [], []
    for i in range(7):
        group = kernels[1 + i * 100 : 1 + (i + 1) * 100]
        span = (group[-1]["end"] - group[0]["start"]) / 1e8
        spans.append(span)
        active.append(sum(r["end"] - r["start"] for r in group) / 1e8)
        errors.append(abs(b["device_event_ms"][i] - span) / b["device_event_ms"][i])
    if (
        not isinstance(source.get("source_commit"), str)
        or not re.fullmatch("[0-9a-f]{40}", source["source_commit"])
        or type(source.get("worktree_dirty")) is not bool
    ):
        raise ValueError("CUDA calibration requires exact source state")
    if source.get("execution_environment") not in ("bare_metal", "wsl2"):
        raise ValueError("CUDA calibration requires a known execution environment")
    overhead = statistics.median(b["device_event_ms"]) / statistics.median(a["device_event_ms"])
    reasons = []
    if max(errors) > 0.05:
        reasons.append("event and activity window disagree by more than five percent")
    if overhead > 1.05:
        reasons.append("profiler overhead exceeds five percent")
    if source["worktree_dirty"]:
        reasons.append("source tree has uncommitted changes")
    # No bare-metal requirement since 2026-09-25 (owner, MASTER_AUDIT): the
    # Nsight kernel activity window is a GPU-side clock and the CUDA event its
    # witness, so the 5% agreement and 5% overhead gates above are the
    # independent-witness method on WSL2 and bare metal alike.
    return dict(
        schema=1,
        sample_id=sample_id,
        capture_sha256=capture_sha256,
        source=dict(source),
        capture_device=dict(capture_device),
        clean=clean,
        profiled=profiled,
        kernels=kernels,
        activity_window_ms=spans,
        kernel_active_ms=active,
        maximum_clock_relative_error=max(errors),
        instrumentation_overhead=overhead,
        eligible_for_promotion=not reasons,
        ineligibility_reasons=reasons,
    )


def read_nsys_kernels(path):
    import sqlite3

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT k.start,k.end,k.deviceId,k.contextId,k.streamId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"
        ).fetchall()
    return [dict(zip(("start", "end", "device", "context", "stream", "name"), row, strict=True)) for row in rows]


def capture_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_nsys_device(path):
    import sqlite3

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT d.cudaId,g.computeMajor,g.computeMinor,d.uuid,d.pid "
            "FROM TARGET_INFO_CUDA_DEVICE d JOIN TARGET_INFO_GPU g ON g.id=d.gpuId"
        ).fetchall()
    if len(rows) != 1:
        raise ValueError("CUDA calibration requires one recorded device")
    device, major, minor, uuid, pid = rows[0]
    return dict(device=device, architecture=f"sm_{major}{minor}", uuid=uuid, process_id=pid)
