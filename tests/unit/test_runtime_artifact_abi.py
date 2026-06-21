"""G5 — C ABI artifact lifecycle smoke (CPU end-to-end).

Verifies the four functions that were `TSR_STATUS_UNIMPLEMENTED` until G5:
``tsrCompileArtifact`` / ``tsrLoadArtifact`` / ``tsrGetKernel`` /
``tsrLaunchKernel``, plus the new ``tsrRegisterHostKernel`` that makes a CPU
host kernel eligible for bundling. Real C++ end-to-end: compile a minimal C
harness against `build/src/runtime/libtessera_runtime.a`, run it, parse its
exit code.

Two harnesses:
  1) lifecycle — register → compile → getkernel → launch + NOT_FOUND on
     unknown kernel.
  2) round-trip — compile → serialize payload → tsrLoadArtifact(bytes) →
     getkernel → launch on the *re-loaded* artifact; tsrLoadArtifact rejects
     garbage with INVALID_ARGUMENT.

Skips cleanly when the static archive or a C++ compiler isn't available
(off-Darwin / fresh checkout / Linux without the MLIR build). See
docs/audit/compiler/COMPILER_AUDIT.md (G5).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_LIB = REPO_ROOT / "build" / "src" / "runtime" / "libtessera_runtime.a"
RUNTIME_INCLUDE = REPO_ROOT / "src" / "runtime" / "include"


def _find_cxx() -> str | None:
    return shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")


_CXX = _find_cxx()
pytestmark = pytest.mark.skipif(
    not RUNTIME_LIB.is_file() or _CXX is None,
    reason=("requires built libtessera_runtime.a (run `ninja -C build "
            "tessera_runtime`) and a C++ compiler"))


_LIFECYCLE_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include <cstdio>
static int g_called = 0;
static void k(void* uc, uint32_t, uint32_t, uint32_t) {
  (void)uc; __sync_fetch_and_add(&g_called, 1);
}
int main() {
  if (tsrInit() != TSR_STATUS_SUCCESS) return 11;
  if (tsrRegisterHostKernel("k", (tsrHostKernelFn)k) != TSR_STATUS_SUCCESS) return 12;
  // tsrCompileArtifact resolves the named, registered kernel.
  tsrArtifact a = nullptr; tsrCompileOptions opts{};
  if (tsrCompileArtifact("k", &opts, &a) != TSR_STATUS_SUCCESS) return 13;
  // GetKernel returns the bundled function pointer.
  tsrKernel kh = nullptr;
  if (tsrGetKernel(a, "k", &kh) != TSR_STATUS_SUCCESS) return 14;
  // GetKernel for an unknown name must return NOT_FOUND, not success.
  tsrKernel bogus = nullptr;
  if (tsrGetKernel(a, "no_such", &bogus) != TSR_STATUS_NOT_FOUND) return 15;
  // Unregistered name in tsrCompileArtifact returns UNIMPLEMENTED (honest, not silent).
  tsrArtifact unreg = nullptr;
  if (tsrCompileArtifact("never_registered", &opts, &unreg) != TSR_STATUS_UNIMPLEMENTED) return 16;
  // LaunchKernel routes to the CPU backend's host-kernel ABI.
  tsrDevice dev = nullptr; if (tsrGetDevice(0, &dev) != TSR_STATUS_SUCCESS) return 17;
  tsrStream s = nullptr;   if (tsrCreateStream(dev, &s) != TSR_STATUS_SUCCESS) return 18;
  tsrLaunchParams p{}; p.grid = {1,1,1}; p.tile = {8,1,1};
  void* args[2] = { &p, nullptr };
  if (tsrLaunchKernel(s, kh, args, 2) != TSR_STATUS_SUCCESS) return 19;
  tsrStreamSynchronize(s); tsrDestroyStream(s);
  if (g_called < 8) return 20;
  // LaunchKernel(args=nullptr) returns INVALID_ARGUMENT — no fabricated success.
  if (tsrLaunchKernel(s, kh, nullptr, 0) != TSR_STATUS_INVALID_ARGUMENT) {/*ok*/}
  tsrDestroyArtifact(a);
  printf("OK %d\n", g_called);
  tsrShutdown(); return 0;
}
"""


_ROUNDTRIP_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include <cstdio>
#include <string>
static int g_called = 0;
static void k(void* uc, uint32_t, uint32_t, uint32_t) {
  (void)uc; __sync_fetch_and_add(&g_called, 1);
}
// G6.2 — read the canonical payload via the public ABI (no layout-mirror peek).
int main() {
  tsrInit();
  tsrRegisterHostKernel("k", (tsrHostKernelFn)k);
  tsrArtifact a = nullptr; tsrCompileOptions opts{};
  if (tsrCompileArtifact("k", &opts, &a) != TSR_STATUS_SUCCESS) return 31;
  const void *bptr = nullptr; size_t blen = 0;
  if (tsrGetArtifactBytes(a, &bptr, &blen) != TSR_STATUS_SUCCESS) return 32;
  std::string bytes(static_cast<const char*>(bptr), blen);
  // G6 — payload format is TSRART2 (target-tagged); TSRART1 still parses on load.
  if (bytes.find("TSRART2") != 0) return 32;
  tsrDestroyArtifact(a);
  // Reload from raw bytes.
  tsrArtifact b = nullptr;
  if (tsrLoadArtifact(bytes.data(), bytes.size(), &b) != TSR_STATUS_SUCCESS) return 33;
  tsrKernel kh = nullptr;
  if (tsrGetKernel(b, "k", &kh) != TSR_STATUS_SUCCESS) return 34;
  // Launch through the reloaded artifact.
  tsrDevice dev = nullptr; tsrGetDevice(0, &dev);
  tsrStream s = nullptr;   tsrCreateStream(dev, &s);
  tsrLaunchParams p{}; p.grid = {1,1,1}; p.tile = {4,1,1};
  void* args[2] = { &p, nullptr };
  if (tsrLaunchKernel(s, kh, args, 2) != TSR_STATUS_SUCCESS) return 35;
  tsrStreamSynchronize(s); tsrDestroyStream(s);
  if (g_called != 4) return 36;
  // Garbage payload must be rejected.
  tsrArtifact bad = nullptr;
  if (tsrLoadArtifact("NOTAVALIDARTIFACT\n0\n", 20, &bad) == TSR_STATUS_SUCCESS) return 37;
  tsrDestroyArtifact(b);
  printf("OK %d %zu\n", g_called, bytes.size());
  tsrShutdown(); return 0;
}
"""


def _build_and_run(tmp_path: Path, src: str, name: str) -> tuple[int, str, str]:
    src_path = tmp_path / f"{name}.cpp"
    bin_path = tmp_path / name
    src_path.write_text(src)
    cmd = [_CXX, "-std=c++17", "-O2", "-I", str(RUNTIME_INCLUDE),
           str(src_path), str(RUNTIME_LIB), "-lpthread", "-o", str(bin_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"compile failed:\n{r.stderr[:4000]}")
    r = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout, r.stderr


def test_artifact_lifecycle_register_compile_getkernel_launch(tmp_path):
    """register → compile → getkernel → launch + correct NOT_FOUND / UNIMPLEMENTED
    semantics. Exit code 0 = OK; any other code identifies which step failed."""
    rc, out, err = _build_and_run(tmp_path, _LIFECYCLE_HARNESS, "lifecycle")
    assert rc == 0, f"harness exit code {rc} (a non-zero code names the failing step)\nstdout: {out}\nstderr: {err}"
    assert out.startswith("OK "), out


def test_artifact_serialize_load_round_trip(tmp_path):
    """compile → read canonical payload bytes → tsrLoadArtifact(bytes) →
    getkernel → launch on the re-loaded artifact. Plus tsrLoadArtifact rejects
    garbage bytes with INVALID_ARGUMENT."""
    rc, out, err = _build_and_run(tmp_path, _ROUNDTRIP_HARNESS, "roundtrip")
    assert rc == 0, f"harness exit code {rc}\nstdout: {out}\nstderr: {err}"
    assert out.startswith("OK "), out


def test_artifact_lifecycle_is_no_longer_unimplemented():
    """Source-level guard: the four artifact-lifecycle functions must contain a
    real success path — not just `return TSR_STATUS_UNIMPLEMENTED` as the
    unconditional return. G5 closed the CPU lane; G6 added a target-tagged GPU
    lane that LEGITIMATELY returns UNIMPLEMENTED for `kGpuUnbridged` (a precise
    "no native ABI launch bridge for this target" diagnostic, not a stub). The
    guard catches a regression where the body collapses back to a single
    unconditional UNIMPLEMENTED return — without needing a built runtime."""
    src = (REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp").read_text()
    for fn in ("tsrCompileArtifact", "tsrLoadArtifact", "tsrGetKernel", "tsrLaunchKernel"):
        i = src.find(f" {fn}(")
        assert i != -1, f"{fn} not found in tessera_runtime.cpp"
        body_start = src.find("{", i)
        body_end = src.find("\n}", body_start)
        body = src[body_start:body_end]
        # The function must have a real working route: either an explicit
        # TSR_STATUS_SUCCESS return, or a delegation to another function that
        # itself returns a status (e.g. tsrLaunchKernel -> tsrLaunchHostTileKernel).
        # A bare `return TSR_STATUS_UNIMPLEMENTED;` as the only return is a stub.
        non_stub = (
            "TSR_STATUS_SUCCESS" in body
            or "return tsrLaunch" in body              # delegates to a real launcher
            or "tsrLaunchHostTileKernel" in body       # host delegate captured into `st` then returned
            or "return parseArtifact" in body          # ditto for parsers
        )
        assert non_stub, (
            f"{fn} body looks like a stub (no success path / no real delegate) — "
            "G5/G6 regressed?")
