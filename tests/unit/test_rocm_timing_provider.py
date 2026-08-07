from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

from tessera.compiler.profiler_timing import validate_timing_sample


ROOT = Path(__file__).resolve().parents[2]


def test_rocm_native_probe_is_optional_and_fail_closed() -> None:
    cmake = (ROOT / "tools/profiler/CMakeLists.txt").read_text()
    source = (ROOT / "tools/profiler/src/runtime/rocm_timing_provider.hip").read_text()
    assert "TPROF_WITH_HIP" in cmake
    assert "tprof_rocm_timing" in cmake
    for phrase in (
        "wall_clock64()",
        "hipDeviceAttributeWallClockRate",
        'result.architecture == "gfx1151"',
        "hipEventElapsedTime",
        "empty_probe_kernel",
        "result.wsl",
    ):
        assert phrase in source


def test_rocm_probe_wrapper_preserves_invalid_hip_event(tmp_path: Path) -> None:
    probe = tmp_path / "fake_rocm_probe.py"
    output = tmp_path / "timing.json"
    probe.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({"
        '"schema":"tessera.rocm_clock_probe.v1",'
        '"architecture":"gfx1151","device_name":"test",'
        '"error":"","exact_gfx1151":True,"wsl":True,'
        '"repetitions":10,"blocks":80,"threads":256,'
        '"host_wall_ns":10000,"empty_host_wall_ns":1000,'
        '"hip_event_ms":0.0,"empty_hip_event_ms":0.0,'
        '"device_wall_clock_ticks":250,"device_wall_clock_ns":10000,'
        '"wall_clock_rate_khz":25000,"host_wall_valid":True,'
        '"hip_event_valid":False,"device_wall_clock_valid":True,'
        '"clock_agreement_valid":False,"eligible_for_regression":True,'
        '"eligible_for_promotion":False}))\n',
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_rocm_timing.py"),
            "--probe",
            str(probe),
            "--artifact-digest",
            "sha256:test",
            "--sample-id",
            "gfx1151-test",
            "--repetitions",
            "10",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_timing_sample(payload)
    assert payload["clocks"]["hip_event_ns"]["valid"] is False
    assert payload["clocks"]["hip_event_ns"]["value"] is None
    assert payload["clocks"]["device_wall_clock_ns"]["valid"] is True
    assert payload["clocks"]["device_wall_clock_ns"]["eligible_for_promotion"] is False
