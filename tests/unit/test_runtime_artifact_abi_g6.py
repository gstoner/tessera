"""G6 — Target-tagged GPU artifact lifecycle on the C ABI.

The G5 tests proved the CPU host-kernel lifecycle (compile -> load ->
getkernel -> launch). They could not, on their own, catch a regression where
the GPU lane *silently succeeded* without a real native bridge. G6 extends the
C ABI artifact to carry (target, compiler_path, execution_kind) and makes
tsrKernel a tagged union over {CPU host fn, GPU artifact without native bridge}.

This test pins the contract:

  1) tsrCompileArtifact(... target="apple_gpu" ...) produces a target-tagged
     artifact whose payload is v2 (TSRART2 magic) and whose kernel slots are
     placeholder names (no host fn-pointer to launch through).
  2) tsrLoadArtifact round-trips the v2 payload bit-exactly.
  3) tsrGetKernel returns a kernel handle (success), but...
  4) tsrLaunchKernel returns TSR_STATUS_UNIMPLEMENTED with a precise
     "no native C-ABI launch bridge" reason. Honest, not a stub.
  5) v1 (legacy TSRART1) payloads still load (target defaults to "cpu").

This is the audit's missing "compiler -> C ABI -> GPU end-to-end" test
(test_apple_gpu_tile_pass_status_matches_envelope proves status agreement
between the C++ pass and the Python envelope, but not the C ABI lifecycle).
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


def _cxx() -> str | None:
    return shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")


_CXX = _cxx()
pytestmark = pytest.mark.skipif(
    not RUNTIME_LIB.is_file() or _CXX is None,
    reason=("requires built libtessera_runtime.a (run `ninja -C build "
            "tessera_runtime`) and a C++ compiler"))


_GPU_LIFECYCLE_HARNESS = r"""
// G6 — Apple GPU artifact lifecycle on the C ABI. The launch step must return
// TSR_STATUS_UNIMPLEMENTED (no native ABI launch bridge yet), not silently
// succeed. Exit code 0 = OK; any other code names the failing step.
#include "tessera/tessera_runtime.h"
#include <cstdio>
#include <cstring>
#include <string>
int main() {
  if (tsrInit() != TSR_STATUS_SUCCESS) return 11;
  // 1) Compile an Apple GPU artifact. No registered kernels needed — the GPU
  //    lane records kernel NAMES with placeholder fn-pointers; the bridge is
  //    a separate gap.
  tsrCompileOptions opts{};
  opts.target = "apple_gpu";
  opts.options_json = nullptr;
  tsrArtifact a = nullptr;
  if (tsrCompileArtifact("flash_attn,matmul,softmax", &opts, &a) != TSR_STATUS_SUCCESS) return 12;
  // 2) Payload must be v2 (TSRART2) and contain the target tag.
  //    We can't read a->payload through the opaque pointer publicly, so use
  //    a layout mirror to confirm. Must match tessera_runtime.cpp struct.
  struct artifact_layout {
    std::string payload;
    std::string target;
    std::string compiler_path;
    std::string execution_kind;
  };
  auto* layout = reinterpret_cast<artifact_layout*>(a);
  if (layout->payload.find("TSRART2") != 0) return 13;
  if (layout->target != "apple_gpu") return 14;
  if (layout->compiler_path.find("apple_gpu") == std::string::npos) return 15;
  if (layout->execution_kind != "artifact_only") return 16;
  // 3) tsrGetKernel succeeds for a registered name in the artifact.
  tsrKernel k = nullptr;
  if (tsrGetKernel(a, "flash_attn", &k) != TSR_STATUS_SUCCESS) return 17;
  // 4) tsrLaunchKernel must return UNIMPLEMENTED, not silently SUCCESS.
  tsrDevice dev = nullptr; if (tsrGetDevice(0, &dev) != TSR_STATUS_SUCCESS) return 18;
  tsrStream s = nullptr;   if (tsrCreateStream(dev, &s) != TSR_STATUS_SUCCESS) return 19;
  tsrLaunchParams p{}; p.grid = {1,1,1}; p.tile = {8,1,1};
  void* args[2] = { &p, nullptr };
  TsrStatus st = tsrLaunchKernel(s, k, args, 2);
  if (st != TSR_STATUS_UNIMPLEMENTED) {
    fprintf(stderr, "expected UNIMPLEMENTED, got %d\n", (int)st);
    return 20;
  }
  // 5) Round-trip: serialize bytes, destroy, load, get kernel — still GPU,
  //    still UNIMPLEMENTED to launch (the target/compiler_path/execution_kind
  //    travel with the bytes).
  std::string bytes = layout->payload;
  tsrDestroyKernel(k);
  tsrDestroyArtifact(a);
  tsrArtifact b = nullptr;
  if (tsrLoadArtifact(bytes.data(), bytes.size(), &b) != TSR_STATUS_SUCCESS) return 21;
  auto* layout2 = reinterpret_cast<artifact_layout*>(b);
  if (layout2->target != "apple_gpu") return 22;
  if (layout2->execution_kind != "artifact_only") return 23;
  tsrKernel kb = nullptr;
  if (tsrGetKernel(b, "matmul", &kb) != TSR_STATUS_SUCCESS) return 24;
  if (tsrLaunchKernel(s, kb, args, 2) != TSR_STATUS_UNIMPLEMENTED) return 25;
  // 6) v1 legacy payload still loads as a CPU artifact (target defaults to cpu).
  const char *v1 = "TSRART1\n0\n";
  tsrArtifact c = nullptr;
  if (tsrLoadArtifact(v1, strlen(v1), &c) != TSR_STATUS_SUCCESS) return 26;
  if (reinterpret_cast<artifact_layout*>(c)->target != "cpu") return 27;
  // Cleanup.
  tsrDestroyKernel(kb);
  tsrDestroyArtifact(c);
  tsrDestroyArtifact(b);
  tsrDestroyStream(s);
  printf("OK\n");
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


def test_gpu_artifact_lifecycle_returns_unimplemented_launch(tmp_path):
    """G6 — Apple GPU artifact: compile → payload-is-v2 → load (round-trip) →
    getkernel succeeds → tsrLaunchKernel returns UNIMPLEMENTED with a precise
    reason → v1 legacy still loads. The CPU passing test cannot mask this gap."""
    rc, out, err = _build_and_run(tmp_path, _GPU_LIFECYCLE_HARNESS, "gpu_lifecycle")
    assert rc == 0, (f"harness exit code {rc} (a non-zero code names the failing "
                     f"step)\nstdout: {out}\nstderr: {err}")
    assert out.strip() == "OK", out


def test_gpu_artifact_unimplemented_reason_is_precise():
    """Source-level guard: the GPU-unbridged TSR_STATUS_UNIMPLEMENTED branch
    in tsrLaunchKernel must carry a precise reason naming the target — not a
    generic stub message. Catches a regression where the GPU branch collapses
    to a one-liner without diagnostics."""
    src = (REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp").read_text()
    i = src.find(" tsrLaunchKernel(")
    body = src[src.find("{", i):src.find("\n}", i)]
    assert "kGpuUnbridged" in body
    assert "no native C-ABI launch bridge" in body, (
        "tsrLaunchKernel GPU branch must carry a precise 'no native C-ABI "
        "launch bridge' reason (not just return UNIMPLEMENTED silently)")


def test_g6_artifact_struct_carries_target_compiler_path_execution_kind():
    """Source-level guard: tsrArtifact_t must carry the G6 target-tagging
    fields (target, compiler_path, execution_kind). Catches a regression that
    collapses it back to the G5 CPU-only shape."""
    src = (REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp").read_text()
    # Find the struct body.
    i = src.find("struct tsrArtifact_t")
    assert i != -1
    body = src[i:src.find("};", i)]
    for field in ("std::string target", "std::string compiler_path",
                  "std::string execution_kind"):
        assert field in body, f"tsrArtifact_t missing G6 field: {field}"


def test_g6_kernel_is_tagged_union():
    """Source-level guard: tsrKernel_t carries a kind (tagged union over
    {CPU host fn, GPU unbridged}). Catches a regression to the G5 shape."""
    src = (REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp").read_text()
    assert "enum class tsrKernelKind" in src
    assert "kHostCpu" in src and "kGpuUnbridged" in src
    i = src.find("struct tsrKernel_t")
    body = src[i:src.find("};", i)]
    assert "tsrKernelKind kind" in body
