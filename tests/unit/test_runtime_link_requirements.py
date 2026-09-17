"""The runtime archive publishes what an out-of-CMake consumer must link.

`libtessera_runtime.a` carries the CUDA/HIP backend objects whenever the build
enabled them. A consumer inside the build gets `libcudart` / `libamdhip64` from
the target's usage requirements; a harness compiled with its own command line
does not, and six runtime C-ABI tests failed on both ROCm boxes with a page of
`undefined reference to hipMalloc` — a missing library reported as a broken ABI.
CI never saw it: its runtime is built with neither backend, so the archive has no
device objects and the bare link succeeds.

The build now writes `tessera_runtime.consumer-link.txt` beside the archive.
These tests hold the writer and the readers together, because the failure mode if
they drift is exactly the one above: silent on a CPU-only build, a wall of
linker errors on a device build.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._support.runtime_link import runtime_consumer_link_args

REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE = REPO_ROOT / "src/runtime/CMakeLists.txt"
SIDECAR_NAME = "tessera_runtime.consumer-link.txt"
CONSUMERS = (
    "tests/unit/test_runtime_artifact_abi.py",
    "tests/unit/test_runtime_artifact_abi_g6.py",
    "tests/unit/test_runtime_profiler_trace.py",
)


def test_the_build_writes_the_sidecar_beside_the_archive():
    text = CMAKE.read_text(encoding="utf-8")
    assert SIDECAR_NAME in text
    # `file(GENERATE)` and not `file(WRITE)`: the paths come from `find_library`
    # results that are only known after configure, and the output directory is a
    # generator expression.
    assert re.search(r"file\(GENERATE\b", text), "the sidecar must be generated, not written at configure time"
    assert "$<TARGET_FILE_DIR:tessera_runtime>" in text


@pytest.mark.parametrize("backend,library", [("HIP", "amdhip64"), ("CUDA", "cudart")])
def test_each_enabled_backend_contributes_its_vendor_runtime(backend, library):
    """A backend compiled into the archive without its library in the sidecar is
    the original defect with an extra step."""
    text = CMAKE.read_text(encoding="utf-8")
    assert f"find_library(TESSERA_RUNTIME_{backend}_LIBRARY {library}" in text
    assert f"if(TESSERA_ENABLE_{backend} AND TESSERA_RUNTIME_{backend}_LIBRARY)" in text


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_every_harness_that_links_the_archive_reads_the_sidecar(consumer):
    text = (REPO_ROOT / consumer).read_text(encoding="utf-8")
    links = text.count("str(RUNTIME_LIB),")
    reads = text.count("runtime_consumer_link_args(RUNTIME_LIB)")
    assert links and reads == links, (
        f"{consumer} links the archive {links} time(s) and reads the sidecar {reads} time(s)")


def test_an_absent_sidecar_reads_as_no_extra_arguments(tmp_path):
    """A build tree from before the sidecar degrades to the old behaviour —
    correct on a CPU-only build, and the same link error as before on a device
    build — rather than to a wrong answer."""
    assert runtime_consumer_link_args(tmp_path / "libtessera_runtime.a") == ()


def test_the_sidecar_is_paths_one_per_line(tmp_path):
    (tmp_path / SIDECAR_NAME).write_text("/opt/rocm/lib/libamdhip64.so\n\n/usr/local/cuda/lib64/libcudart.so\n")
    assert runtime_consumer_link_args(tmp_path / "libtessera_runtime.a") == (
        "/opt/rocm/lib/libamdhip64.so", "/usr/local/cuda/lib64/libcudart.so")


def test_this_host_agrees_with_its_own_build_tree():
    """Whatever this host built, the sidecar and the archive must agree: a device
    backend in the archive means its library in the sidecar."""
    archive = REPO_ROOT / "build/src/runtime/libtessera_runtime.a"
    if not archive.is_file():
        pytest.skip("no built libtessera_runtime.a on this host")
    cache = REPO_ROOT / "build/CMakeCache.txt"
    if not cache.is_file():
        pytest.skip("no CMakeCache.txt to read the build's backend selection from")
    text = cache.read_text(encoding="utf-8", errors="replace")
    args = runtime_consumer_link_args(archive)
    for backend, library in (("HIP", "amdhip64"), ("CUDA", "cudart")):
        enabled = re.search(rf"^TESSERA_ENABLE_{backend}:BOOL=(ON|TRUE|1)$", text, re.MULTILINE)
        named = any(library in arg for arg in args)
        if enabled and not named:
            pytest.fail(f"this build enabled {backend} but the sidecar names no {library}: {args}")
        if named and not enabled:
            pytest.fail(f"the sidecar names {library} but this build did not enable {backend}")
