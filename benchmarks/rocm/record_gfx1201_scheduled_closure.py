"""Run and record the complete gfx1201 scheduled-package closure gate.

The ordinary unit lane intentionally skips exact-device rows.  This recorder
turns that expected skip state into a bounded owning-host gate: it requires the
selected RX 9070 XT, an explicitly selected current ``tessera-opt``, all 90
hardware/compiler-dependent cases, and the five adjacent host-contract cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from tessera import runtime as rt  # noqa: E402
from tessera.compiler import rocm_native  # noqa: E402
from tessera.compiler.rocm_exact_device_proofs import (  # noqa: E402
    GFX1201_SCHEDULED_SUITE_PROOF,
)
from tessera.compiler.rocm_target import (  # noqa: E402
    TESSERA_TARGET_HIP,
    TESSERA_TARGET_ROCM,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt  # noqa: E402


EXPECTED_DEVICE = "AMD Radeon RX 9070 XT"
_PROOF_BUILD_RE = re.compile(
    r"^llvm(?P<llvm>\d+\.\d+\.\d+)\+rocm(?P<rocm>\d+\.\d+)\+gfx1201$"
)


DEVICE_CASE_FAMILIES = {
    "test_gfx1201_public_matmul_row_uses_exact_executor_on_owning_device": 1,
    "test_gfx1201_scheduled_package_executes": 2,
    "test_gfx1201_scheduled_matmul_package_executes_fused_epilogue": 15,
    "test_gfx1201_scheduled_matmul_package_executes_fp8": 6,
    "test_gfx1201_scheduled_matmul_package_executes": 3,
    "test_gfx1201_scheduled_attention_package_executes": 8,
    "test_gfx1201_driver_uses_adjacent_scheduled_artifacts": 2,
    "test_gfx1201_backward_program": 4,
    "test_gfx1201_public_attention_native_backward": 4,
    "test_gfx1201_reusable_attention_owner": 4,
    "test_gfx1201_dynamic_matmul_reuses_one_image": 1,
    "test_gfx1201_external_reader_orders_reuse_and_retirement": 4,
    "test_gfx1201_scheduled_paged_kv_package_executes": 4,
    "test_gfx1201_scheduled_matmul_package_executes_integer_storage": 6,
    "test_gfx1201_scheduled_matmul_package_executes_bf16": 4,
    "test_gfx1201_scheduled_matmul_package_executes_the_selected_panel": 3,
    "test_gfx1201_low_precision_takes_the_selected_panel_and_executes": 4,
    "test_gfx1201_double_k_int4_emits_its_instruction_and_is_exact": 2,
    "test_gfx1201_b_fragment_uses_the_transpose_load_only_where_derived": 5,
    "test_gfx1201_macro_k_block_walks_the_whole_contraction": 4,
    "test_gfx1201_mixed_fp8_pairs_select_their_instruction_and_execute": 4,
}

HOST_CONTRACT_FAMILIES = {
    "test_gfx1201_unary_projection_rejects_stale_metadata": 2,
    "test_gfx1201_cached_launcher_keeps_architecture_and_family_gate": 2,
    "test_public_attention_rejects_unknown_rocm_architecture": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def _version(output: str, pattern: str, label: str) -> str:
    match = re.search(pattern, output, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot parse {label} version")
    return match.group(1)


def _same_release(observed: str, required: str) -> bool:
    return observed == required or observed.startswith(required + ".")


def _validate_build_versions(
    proof_build: str,
    *,
    compiler_output: str,
    hipcc_output: str,
    rocm_release: str,
) -> dict[str, str]:
    build = _PROOF_BUILD_RE.fullmatch(proof_build)
    if build is None:
        raise RuntimeError(f"unparseable gfx1201 proof build: {proof_build}")
    llvm = _version(compiler_output, r"\bLLVM version\s+(\d+\.\d+\.\d+)", "LLVM")
    hip = _version(hipcc_output, r"\bHIP version:\s*(\d+\.\d+(?:\.\d+)?)", "HIP")
    required = {
        "LLVM": build.group("llvm"),
        "ROCm": build.group("rocm"),
        "HIP": TESSERA_TARGET_HIP,
    }
    observed = {"LLVM": llvm, "ROCm": rocm_release, "HIP": hip}
    for label, expected in required.items():
        if not _same_release(observed[label], expected):
            raise RuntimeError(
                f"gfx1201 proof build requires {label} {expected}; "
                f"observed {observed[label]}"
            )
    if build.group("rocm") != TESSERA_TARGET_ROCM:
        raise RuntimeError(
            f"proof build ROCm {build.group('rocm')} disagrees with "
            f"Tessera target ROCm {TESSERA_TARGET_ROCM}"
        )
    return {
        "llvm": llvm,
        "rocm": rocm_release,
        "hip": hip,
        "compiler_output": compiler_output,
        "hipcc_output": hipcc_output,
    }


def _rocm_release() -> str:
    roots = []
    if configured := os.environ.get("ROCM_PATH"):
        roots.append(Path(configured))
    roots.extend((Path("/opt/rocm/core"), Path("/opt/rocm")))
    seen: set[Path] = set()
    for root in roots:
        for candidate in (root, root.resolve()):
            marker = candidate / ".info" / "version"
            if marker in seen:
                continue
            seen.add(marker)
            if marker.is_file():
                return _version(marker.read_text(), r"^(\d+\.\d+(?:\.\d+)?)", "ROCm")
    raise RuntimeError("cannot locate the ROCm release marker under ROCM_PATH")


def _selected_hip_device_name() -> str:
    hip = rt._load_hip_for_launch()
    if hip is None:
        raise RuntimeError("cannot load HIP to identify the selected gfx1201 device")
    get_device = hip.hipGetDevice
    get_name = hip.hipDeviceGetName
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_device.restype = ctypes.c_int
    get_name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    get_name.restype = ctypes.c_int
    device = ctypes.c_int()
    name = ctypes.create_string_buffer(256)
    if get_device(ctypes.byref(device)) != 0:
        raise RuntimeError("hipGetDevice failed while identifying the proof device")
    if get_name(name, len(name), device.value) != 0:
        raise RuntimeError("hipDeviceGetName failed while identifying the proof device")
    value = name.value.decode("utf-8", errors="strict").strip()
    if not value:
        raise RuntimeError("selected HIP device returned an empty model name")
    return value


def _compiler_source_root(tool: Path) -> Path:
    for directory in tool.resolve().parents:
        cache = directory / "CMakeCache.txt"
        if not cache.is_file():
            continue
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                source = Path(line.split("=", 1)[1]).resolve()
                if source.is_dir():
                    return source
    raise RuntimeError(f"cannot bind {tool} to its CMake source checkout")


def _case_family(name: str) -> str:
    return name.split("[", 1)[0]


def _parse_report(path: Path) -> tuple[dict[str, int], dict[str, int], dict[str, object]]:
    suite = ET.parse(path).getroot().find("testsuite")
    if suite is None:
        raise RuntimeError("pytest JUnit report has no testsuite")
    names = [_case_family(case.attrib["name"]) for case in suite.findall("testcase")]
    counts = Counter(names)
    expected = DEVICE_CASE_FAMILIES | HOST_CONTRACT_FAMILIES
    unknown = sorted(set(counts) - set(expected))
    missing = sorted(set(expected) - set(counts))
    if unknown or missing:
        raise RuntimeError(f"gfx1201 closure inventory drift: unknown={unknown}, missing={missing}")
    for family, count in expected.items():
        if counts[family] != count:
            raise RuntimeError(
                f"gfx1201 closure count drift for {family}: {counts[family]} != {count}"
            )
    device = {name: counts[name] for name in DEVICE_CASE_FAMILIES}
    host = {name: counts[name] for name in HOST_CONTRACT_FAMILIES}
    summary: dict[str, object] = {
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
        "seconds": float(suite.attrib["time"]),
        "hostname": suite.attrib.get("hostname", ""),
        "timestamp": suite.attrib.get("timestamp", ""),
    }
    return device, host, summary


def record(output: Path) -> None:
    proof = GFX1201_SCHEDULED_SUITE_PROOF
    if os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1":
        raise RuntimeError("set TESSERA_GFX1201_DEVICE_PROOF=1 for this owning-device gate")
    if rt._rocm_live_arch() != "gfx1201" or rt._rocm_chip() != "gfx1201":
        raise RuntimeError("gfx1201 closure evidence requires the selected gfx1201 device and compiler target")
    selected_device = _selected_hip_device_name()
    if selected_device != EXPECTED_DEVICE:
        raise RuntimeError(
            f"gfx1201 closure evidence requires {EXPECTED_DEVICE}; "
            f"selected {selected_device}"
        )
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("set TESSERA_OPT to the freshly rebuilt LLVM/MLIR 23.1.1 tessera-opt")
    stale = rocm_native.stale_generator_sources(tool)
    if stale:
        raise RuntimeError(
            f"refusing stale compiler evidence: {tool} predates {len(stale)} generator sources"
        )
    source_revision = _command_output("git", "-C", str(ROOT), "rev-parse", "HEAD")
    source_dirty = _command_output("git", "-C", str(ROOT), "status", "--porcelain")
    if source_dirty:
        raise RuntimeError(f"tested source checkout is dirty: {ROOT}")
    compiler_source = _compiler_source_root(tool)
    compiler_source_revision = _command_output(
        "git", "-C", str(compiler_source), "rev-parse", "HEAD"
    )
    compiler_source_dirty = _command_output(
        "git", "-C", str(compiler_source), "status", "--porcelain"
    )
    if compiler_source_dirty:
        raise RuntimeError(f"compiler source checkout is dirty: {compiler_source}")
    if compiler_source_revision != source_revision:
        raise RuntimeError(
            "compiler source revision does not match the tested checkout: "
            f"{compiler_source_revision} != {source_revision}"
        )

    compiler_output = _command_output(str(tool), "--version")
    hipcc_output = _command_output("hipcc", "--version")
    versions = _validate_build_versions(
        proof.proof_build,
        compiler_output=compiler_output,
        hipcc_output=hipcc_output,
        rocm_release=_rocm_release(),
    )

    with tempfile.TemporaryDirectory(prefix="gfx1201-scheduled-closure-") as tmp:
        report = Path(tmp) / "pytest.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            proof.numerical_fixture,
            "-q",
            f"--junitxml={report}",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        device_cases, host_cases, summary = _parse_report(report)
        if completed.returncode != 0:
            raise RuntimeError(f"gfx1201 closure suite failed with exit code {completed.returncode}")

    required = {
        "tests": proof.required_passed_cases,
        "failures": 0,
        "errors": 0,
        "skipped": proof.required_skipped_cases,
    }
    for key, expected in required.items():
        if summary[key] != expected:
            raise RuntimeError(f"gfx1201 closure requires {key}={expected}; observed {summary[key]}")
    if sum(device_cases.values()) != proof.device_dependent_cases:
        raise RuntimeError("gfx1201 device-dependent case total drifted")
    if sum(host_cases.values()) != proof.host_contract_cases:
        raise RuntimeError("gfx1201 host-contract case total drifted")

    fixture = ROOT / proof.numerical_fixture
    recorder = Path(__file__).resolve()
    packet = {
        "schema": "tessera.rocm.gfx1201_scheduled_closure.v1",
        "work_item": "ROCM-2",
        "sync_key": "GFX1201-SCHEDULED-SKIP-CLOSURE-2026-09-21",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "device": selected_device,
        "target": proof.target,
        "live_architecture": rt._rocm_live_arch(),
        "compiler_target": rt._rocm_chip(),
        "proof_build": proof.proof_build,
        "source_revision": source_revision,
        "fixture": proof.numerical_fixture,
        "fixture_sha256": _sha256(fixture),
        "recorder_sha256": _sha256(recorder),
        "compiler": {
            "path": str(tool.resolve()),
            "sha256": _sha256(tool),
            "version": compiler_output.splitlines()[:3],
            "stale_generator_sources": 0,
            "source_root": str(compiler_source),
            "source_revision": compiler_source_revision,
            "source_dirty": False,
        },
        "toolchain": {
            "llvm_version": versions["llvm"],
            "rocm_version": versions["rocm"],
            "hip_version": versions["hip"],
            "hipcc": hipcc_output.splitlines()[0],
            "rocm_path": os.environ.get("ROCM_PATH", ""),
        },
        "result": summary,
        "device_dependent_cases": {
            "count": sum(device_cases.values()),
            "families": device_cases,
        },
        "host_contract_cases": {
            "count": sum(host_cases.values()),
            "families": host_cases,
        },
        "policy": (
            "ordinary non-owning-host CI keeps the explicit skips; promotion is "
            "bound to this exact-device, fresh-compiler, zero-skip packet"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record(args.output)


if __name__ == "__main__":
    main()
