"""Controlled Graph-to-native NVFP4 compiler comparison; no route promotion."""
import argparse
from contextlib import ExitStack
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import tempfile

import numpy as np
from tessera import runtime as rt
from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources(payload):
    with tempfile.TemporaryDirectory(prefix="nvfp4-resource-") as directory:
        path = Path(directory) / "image.hsaco"
        path.write_bytes(payload)
        notes = subprocess.run(["/opt/rocm/llvm/bin/llvm-readelf", "--notes", str(path)],
                               check=True, capture_output=True, text=True).stdout
    result = {}
    for key in ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size"):
        values = re.findall(r"\." + key + r":\s*(\d+)", notes)
        if len(values) != 1:
            raise RuntimeError(f"ambiguous/missing kernel resource {key}: {values}")
        result[key] = int(values[0])
    result["image_bytes"] = len(payload)
    return result


def bind_tools(root):
    root = Path(root).resolve()
    os.environ["TESSERA_OPT"] = str(root / "tools/tessera-opt/tessera-opt")
    os.environ["TESSERA_ROCM_OPT"] = str(root / "src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt")
    return {name: digest(os.environ[name]) for name in ("TESSERA_OPT", "TESSERA_ROCM_OPT")}


def checked(session, converted, stored, expected):
    session.run_combined()
    values = session.conversion_diagnostics()
    for name, wanted in zip(("packed", "exponents", "stats"), converted, strict=True):
        if name == "stats":
            np.testing.assert_allclose(values[name], wanted, rtol=1e-13, atol=1e-30)
        else:
            np.testing.assert_array_equal(values[name], wanted)
    storage = session.storage_diagnostics()
    for name, wanted in zip(("fragment", "plane"), stored, strict=True):
        np.testing.assert_array_equal(storage[name], wanted)
    output = session.read_output()
    np.testing.assert_allclose(output.astype("f4"), expected, rtol=.008, atol=.015625)
    return values, storage, output


def identical(left, right):
    for a, b in zip(left, right, strict=True):
        if isinstance(a, dict):
            for name in a:
                np.testing.assert_array_equal(a[name].view("u1"), b[name].view("u1"))
        else:
            np.testing.assert_array_equal(a.view("u1"), b.view("u1"))


def profile(shape, roots):
    m, n, k = shape
    args, _, converted, stored, expected = inputs_and_oracle(m, n, k)
    programs = {}
    tools = {}
    for arm, root in roots.items():
        tools[arm] = bind_tools(root)
        programs[arm] = make_function(n, k).compile_native_nvfp4_program(*args, m_bound=m).native
    images = {arm: p.receipt["component_image_digests"] for arm, p in programs.items()}
    # Consumer and storage must remain identical; only converter code changes.
    if images["control"][1:] != images["candidate"][1:]:
        raise RuntimeError("storage or consumer image changed")
    counts = {}
    samples = {arm: {stage: {mode: [] for mode in ("direct", "graph")}
                     for stage in ("converter", "combined")} for arm in programs}
    with ExitStack() as stack:
        sessions = {arm: stack.enter_context(p.native_session(*args)) for arm, p in programs.items()}
        initial = {arm: checked(s, converted, stored, expected) for arm, s in sessions.items()}
        identical(initial["control"], initial["candidate"])
        for arm, s in sessions.items():
            counts[arm] = s.frame_stats()
        for stage in ("converter", "combined"):
            for round_index in range(7):
                order = ("control", "candidate") if round_index % 2 == 0 else ("candidate", "control")
                for mode in ("direct", "graph"):
                    for arm in order:
                        session = sessions[arm]
                        checked(session, converted, stored, expected)
                        if mode == "direct":
                            value = session.measure(stage, samples=1, repeats=128)[0]
                        else:
                            window = session.measure_graph(stage, samples=1, repeats=128)[0]
                            if window["graph_nodes"] != 128 * (1 if stage == "converter" else 3):
                                raise RuntimeError("capture node count mismatch")
                            if window["host_graph_submissions"] != 1:
                                raise RuntimeError("capture submission count mismatch")
                            value = window["per_iteration_ms"]
                        samples[arm][stage][mode].append(value)
                        after = session.conversion_diagnostics()
                        for name in after:
                            np.testing.assert_array_equal(after[name].view("u1"), initial[arm][0][name].view("u1"))
                        if stage == "combined":
                            np.testing.assert_array_equal(session.read_output(), initial[arm][2])
        final = {arm: checked(s, converted, stored, expected) for arm, s in sessions.items()}
        identical(final["control"], final["candidate"])
    ratios = {stage: {mode: median(samples["control"][stage][mode]) /
                      median(samples["candidate"][stage][mode]) for mode in ("direct", "graph")}
              for stage in ("converter", "combined")}
    return {"shape_mnk": list(shape), "tools": tools, "component_images": images,
            "resources": {arm: resources(p.ingest.native.image.payload) for arm, p in programs.items()},
            "input_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in args],
            "correctness": "independent oracle; control/candidate packed, exponents, f64 stats, storage and output bitwise identical",
            "native_frames": counts, "samples_ms": samples, "control_over_candidate": ratios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires exact gfx1201")
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    name = c.create_string_buffer(256)
    uuid = (c.c_ubyte * 16)()
    ordinal = c.c_int()
    if (hip.hipInit(0) or hip.hipGetDevice(c.byref(ordinal)) or
        hip.hipDeviceGetName(name, 256, ordinal.value) or hip.hipDeviceGetUuid(c.byref(uuid), ordinal.value)):
        raise RuntimeError("device identity query failed")
    packet = {"schema": "tessera.nvfp4.winner_codes.compare.v1", "architecture": "gfx1201",
              "device": {"name": name.value.decode(), "uuid": bytes(uuid).hex(), "ordinal": ordinal.value},
              "recorder_sha256": digest(__file__), "selector_promotion": False,
              "runtime_sha256": digest(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
              "timing_domain": "HIP events; direct includes host submission gaps; captured window one submission; seven AB/BA rounds, 128 repetitions",
              "profiles": [profile(shape, {"control": options.control, "candidate": options.candidate})
                           for shape in ((256,64,1024), (256,512,1024), (256,1024,4096))]}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
