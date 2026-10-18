"""Real HIP owner recovery after deterministic host API faults."""

import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="requires owning gfx1201 and matching native provider"
)


@pytest.fixture(scope="module")
def hip_fault_shim(tmp_path_factory):
    """Interpose only failed API statuses; all successful work uses real HIP."""
    directory = tmp_path_factory.mktemp("hip-owner-fault")
    source = directory / "fault.cpp"
    source.write_text(r"""
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <dlfcn.h>
static std::atomic<int> kind{0}, remaining{0};
extern "C" void tessera_test_fault_arm(int selected, int count) {
  remaining.store(count);
  kind.store(selected);
}
static bool fail(int selected) {
  if (kind.load() != selected) return false;
  if (remaining.fetch_sub(1) != 1) return false;
  kind.store(0);
  return true;
}
extern "C" hipError_t hipMalloc(void **pointer, size_t bytes) {
  if (fail(1)) {
    if (pointer) *pointer = nullptr;
    return hipErrorOutOfMemory;
  }
  using Function = hipError_t (*)(void **, size_t);
  static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hipMalloc"));
  return original ? original(pointer, bytes) : hipErrorSharedObjectSymbolNotFound;
}
extern "C" hipError_t hipMemcpyAsync(void *destination, const void *source,
    size_t bytes, hipMemcpyKind direction, hipStream_t stream) {
  if (direction == hipMemcpyHostToDevice && fail(2)) return hipErrorInvalidValue;
  using Function = hipError_t (*)(void *, const void *, size_t, hipMemcpyKind, hipStream_t);
  static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hipMemcpyAsync"));
  return original ? original(destination, source, bytes, direction, stream)
                  : hipErrorSharedObjectSymbolNotFound;
}
""")
    library = directory / "libhip_owner_fault.so"
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    include = next(
        (p for p in (rocm / "include", rocm / "core/include") if (p / "hip/hip_runtime_api.h").exists()), None
    )
    assert include is not None, "owning HIP headers unavailable"
    compiler = shutil.which("c++")
    assert compiler is not None, "owning host C++ compiler unavailable"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-D__HIP_PLATFORM_AMD__",
            "-I",
            str(include),
            str(source),
            "-ldl",
            "-Wl,--no-as-needed",
            "-L",
            str(rocm / "lib"),
            "-lamdhip64",
            "-Wl,-rpath," + str(rocm / "lib"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


@pytest.mark.parametrize(
    "kind,nth,cached,operation",
    [
        (1, 1, False, "prepare"),
        (1, 3, False, "prepare"),
        (2, 1, False, "prepare"),
        (2, 3, False, "prepare"),
        (2, 1, True, "prepare"),
        (2, 3, True, "prepare"),
        (2, 1, True, "full_update"),
        (2, 3, True, "full_update"),
        (2, 1, True, "partial_update"),
        (2, 2, True, "partial_update"),
    ],
)
def test_failed_submission_releases_or_quarantines_before_recovery(hip_fault_shim, kind, nth, cached, operation):
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = str(hip_fault_shim)
    environment["TESSERA_TEST_HIP_FAULT"] = f"{kind},{nth},{int(cached)},{operation}"
    child = r"""
import ctypes as C
import os
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tests.unit.test_native_nvfp4_expansion_program import source_graph
from tests.device.rocm.test_prepared_nvfp4_expansion_program import inputs_and_oracle
assert rt._rocm_live_arch() == "gfx1201"
os.environ["TESSERA_ROCM_PROGRAM_CACHE"] = "1"
package = package_native_nvfp4_expansion_program(source_graph())
roots, expected = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
provider = os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]
with package.prepare(roots, runtime_library=provider) as first:
    generation, _ = first.invoke()
    retained, = first.read(generation)
    np.testing.assert_array_equal(retained, expected)
    library = first.lib
kind, nth, cached, operation = os.environ["TESSERA_TEST_HIP_FAULT"].split(",")
kind, nth, cached = int(kind), int(nth), int(cached)
clear = library.tessera_rocm_program_cache_clear
if not cached:
    assert clear() == 0
shim = C.CDLL(None)
arm = shim.tessera_test_fault_arm
arm.argtypes = [C.c_int, C.c_int]
active = None
if operation != "prepare":
    active = package.prepare(roots, runtime_library=provider)
    generation, _ = active.invoke()
arm(kind, nth)
try:
    if operation == "prepare":
        package.prepare(roots, runtime_library=provider)
    elif operation == "full_update":
        active.update(roots)
    else:
        active.update_activations(roots[3], roots[4])
except RuntimeError as error:
    assert f"status {4 if kind == 1 else 5}" in str(error), str(error)
else:
    raise AssertionError("injected host HIP fault was not observed")
finally:
    arm(0, 0)
if active is not None:
    try:
        active.read(generation)
    except RuntimeError as error:
        assert "status 10" in str(error), str(error)
    else:
        raise AssertionError("failed update published old GPU output")
    active.close()
np.testing.assert_array_equal(retained, expected)
# The failed owner must not return as a usable idle frame.
counts = [C.c_uint64() for _ in range(4)]
stats = library.tessera_rocm_program_cache_stats
stats.argtypes = [C.POINTER(C.c_uint64)] * 4
assert stats(*[C.byref(v) for v in counts]) == 0
assert counts[2].value == counts[3].value == 0
with package.prepare(roots, runtime_library=provider) as recovered:
    generation, _ = recovered.invoke()
    np.testing.assert_array_equal(recovered.read(generation)[0], expected)
np.testing.assert_array_equal(retained, expected)
assert clear() == 0
assert stats(*[C.byref(v) for v in counts]) == 0
assert counts[2].value == counts[3].value == 0
print("real HIP failure recovery passed")
"""
    result = subprocess.run([sys.executable, "-c", child], env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "real HIP failure recovery passed" in result.stdout
