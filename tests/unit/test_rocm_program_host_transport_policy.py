"""Native transport boundaries: architecture, total traffic, slab caps and overrides."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def native_policy(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("rocm-host-transport")
    source = directory / "policy.cpp"
    source.write_text("""
#include "ProgramHostTransportPolicy.h"
#include <cstdlib>
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 5) return 2;
  const char *mode = argv[4][0] == '-' ? nullptr : argv[4];
  std::cout << tessera::rocm::usePinnedProgramHostTransport(
      argv[1], std::strtoull(argv[2], nullptr, 10),
      std::strtoull(argv[3], nullptr, 10), mode);
}
""")
    binary = directory / "policy"
    subprocess.run([compiler, "-std=c++17", "-I",
                    str(root / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip"),
                    str(source), "-o", str(binary)], check=True, capture_output=True,
                   env={**os.environ}, timeout=60)
    return binary


@pytest.mark.parametrize("inputs,outputs,expected", [
    (0, 0, False), (128*1024, 128*1024-1, False),
    (128*1024, 128*1024, True), (256*1024, 0, True),
    (0, 256*1024, True), (121152, 81920, False),
    (203072, 163840, True),
    (8*1024*1024, 8*1024*1024, True),
    (8*1024*1024+1, 0, False), (0, 8*1024*1024+1, False),
    (2**64-1, 2**64-1, False),
])
def test_native_total_traffic_and_bounded_slabs(native_policy, inputs, outputs, expected):
    result = subprocess.check_output(
        [str(native_policy), "gfx1201", str(inputs), str(outputs), "-"], text=True)
    assert result == str(int(expected))


@pytest.mark.parametrize("architecture", ["gfx1151", "nvidia_sm120", "apple_gpu", "x86_avx512"])
def test_native_policy_does_not_transfer_architecture_evidence(native_policy, architecture):
    result = subprocess.check_output(
        [str(native_policy), architecture, "262144", "262144", "-"], text=True)
    assert result == "0"


@pytest.mark.parametrize("mode,expected", [("0", False), ("1", True), ("invalid", False)])
def test_native_explicit_transport_override_is_preserved(native_policy, mode, expected):
    result = subprocess.check_output([str(native_policy), "gfx1201", "1", "1", mode], text=True)
    assert result == str(int(expected))
