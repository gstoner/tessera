"""Host-free: native owner-pool quota and overflow-safe admission policy."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess

import pytest

MiB = 1 << 20
GiB = 1 << 30
U64 = (1 << 64) - 1


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    folder = tmp_path_factory.mktemp("rocm-program-cache-policy")
    source = folder / "policy.cpp"
    source.write_text("""
#include "ProgramCachePolicy.h"
extern "C" uint64_t budget(const char *arch, uint64_t bytes) {
  return tessera::rocm::programIdleByteBudget(arch,bytes);
}
extern "C" bool fits(uint64_t idle,uint64_t frame,uint64_t limit) {
  return tessera::rocm::programFrameFitsIdleBudget(idle,frame,limit);
}
""")
    output = folder / "policy.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC",
                    "-I", str(root / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip"),
                    str(source), "-o", str(output)], check=True, capture_output=True)
    library = C.CDLL(str(output))
    library.budget.argtypes = [C.c_char_p, C.c_uint64]
    library.budget.restype = C.c_uint64
    library.fits.argtypes = [C.c_uint64] * 3
    library.fits.restype = C.c_bool
    return library


@pytest.mark.parametrize("arch,device,expected", [
    ("gfx1201", 0, 128 * MiB), ("gfx1201", 2 * GiB, 128 * MiB),
    ("gfx1201", 4 * GiB, 128 * MiB), ("gfx1201", 8 * GiB, 256 * MiB),
    ("gfx1201", 16 * GiB, 512 * MiB), ("gfx1201", 64 * GiB, 512 * MiB),
    ("gfx1201", U64, 512 * MiB), ("gfx1151", 64 * GiB, 128 * MiB),
    ("gfx1200", 16 * GiB, 128 * MiB), ("unknown", U64, 128 * MiB),
])
def test_device_bounded_budget_preserves_other_architectures(policy, arch, device, expected):
    assert policy.budget(arch.encode(), device) == expected


@pytest.mark.parametrize("idle,frame,limit,expected", [
    (0, 400 * MiB, 512 * MiB, True),
    (128 * MiB, 400 * MiB, 512 * MiB, False),
    (512 * MiB, 1, 256 * MiB, False),
    (512 * MiB, 0, 256 * MiB, False),
    (U64, 1, U64, False), (U64, 0, U64, True),
    (0, U64, U64, True), (1, U64, U64, False),
    (256 * MiB, 256 * MiB, 512 * MiB, True),
    (256 * MiB + 1, 256 * MiB, 512 * MiB, False),
])
def test_admission_never_wraps_when_context_budgets_differ(policy, idle, frame, limit, expected):
    assert policy.fits(idle, frame, limit) is expected
