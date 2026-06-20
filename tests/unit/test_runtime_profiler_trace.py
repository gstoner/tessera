"""Runtime profiler callback ABI and trace spine coverage."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_LIB = REPO_ROOT / "build" / "src" / "runtime" / "libtessera_runtime.a"
RUNTIME_INCLUDE = REPO_ROOT / "src" / "runtime" / "include"
TPROF_INCLUDE = REPO_ROOT / "tools" / "profiler" / "include"
TPROF_SOURCES = [
    REPO_ROOT / "tools" / "profiler" / "src" / "runtime" / "tprof_runtime.cpp",
    REPO_ROOT / "tools" / "profiler" / "src" / "runtime" / "tessera_runtime_adapter.cpp",
    REPO_ROOT / "tools" / "profiler" / "src" / "runtime" / "nvtx_shim.cpp",
    REPO_ROOT / "tools" / "profiler" / "src" / "runtime" / "cupti_shim.cpp",
    REPO_ROOT / "tools" / "profiler" / "src" / "exporters" / "chrome_trace_exporter.cpp",
    REPO_ROOT / "tools" / "profiler" / "src" / "exporters" / "perfetto_exporter.cpp",
]


def _find_cxx() -> str | None:
    return shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")


_CXX = _find_cxx()


def _runtime_lib_has_profile_callback() -> bool:
    nm = shutil.which("nm")
    if nm is None or not RUNTIME_LIB.is_file():
        return False
    result = subprocess.run([nm, "-g", str(RUNTIME_LIB)], capture_output=True, text=True)
    return result.returncode == 0 and "tsrSetProfileEventCallback" in result.stdout


def test_runtime_profile_callback_abi_is_public_and_decoupled() -> None:
    types = (RUNTIME_INCLUDE / "tessera" / "tsr_types.h").read_text()
    runtime_h = (RUNTIME_INCLUDE / "tessera" / "tessera_runtime.h").read_text()
    runtime_cpp = (REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp").read_text()
    cmake = (REPO_ROOT / "src" / "runtime" / "CMakeLists.txt").read_text()

    assert "TsrProfileEventKind" in types
    assert "TSR_PROFILE_RUNTIME_API" in types
    assert "TSR_PROFILE_DEVICE_ACTIVITY" in types
    assert "tsrProfileEventFn" in types
    assert "tsrSetProfileEventCallback" in runtime_h
    assert "EmitProfileEvent" in runtime_cpp
    assert "tprof_runtime" not in cmake


_PROFILE_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include <string>
#include <vector>
#include <cstdio>

struct Ev {
  int kind;
  std::string name;
  std::string payload;
  double value;
};

static std::vector<Ev> a_events;
static std::vector<Ev> b_events;
static int g_called = 0;

static void cb_a(TsrProfileEventKind kind, const char* name,
                 const char* payload_json, double value, void*) {
  a_events.push_back(Ev{(int)kind, name ? name : "", payload_json ? payload_json : "", value});
}

static void cb_b(TsrProfileEventKind kind, const char* name,
                 const char* payload_json, double value, void*) {
  b_events.push_back(Ev{(int)kind, name ? name : "", payload_json ? payload_json : "", value});
}

static void k(void*, uint32_t, uint32_t, uint32_t) {
  __sync_fetch_and_add(&g_called, 1);
}

static bool has_event(const std::vector<Ev>& events, int kind, const char* name) {
  for (const auto& e : events) {
    if (e.kind == kind && e.name == name) return true;
  }
  return false;
}

static bool has_payload(const std::vector<Ev>& events, int kind, const char* needle) {
  for (const auto& e : events) {
    if (e.kind == kind && e.payload.find(needle) != std::string::npos) return true;
  }
  return false;
}

int main() {
  tsrEnableProfiling(0);
  if (tsrSetProfileEventCallback(cb_a, nullptr) != TSR_STATUS_SUCCESS) return 10;

  int count = 0;
  tsrGetDeviceCount(&count);
  if (!a_events.empty()) return 11;

  tsrEnableProfiling(1);
  if (tsrInit() != TSR_STATUS_SUCCESS) return 12;
  tsrDevice dev = nullptr;
  if (tsrGetDevice(0, &dev) != TSR_STATUS_SUCCESS) return 13;
  tsrStream s = nullptr;
  if (tsrCreateStream(dev, &s) != TSR_STATUS_SUCCESS) return 14;

  tsrBuffer src = nullptr, dst = nullptr;
  if (tsrMalloc(dev, 64, &src) != TSR_STATUS_SUCCESS) return 15;
  if (tsrMalloc(dev, 64, &dst) != TSR_STATUS_SUCCESS) return 16;
  if (tsrMemset(src, 7, 64) != TSR_STATUS_SUCCESS) return 17;
  if (tsrMemcpy(dst, src, 64, TSR_MEMCPY_DEVICE_TO_DEVICE) != TSR_STATUS_SUCCESS) return 18;

  if (tsrRegisterHostKernel("k", (tsrHostKernelFn)k) != TSR_STATUS_SUCCESS) return 19;
  tsrArtifact artifact = nullptr;
  tsrCompileOptions opts{};
  if (tsrCompileArtifact("k", &opts, &artifact) != TSR_STATUS_SUCCESS) return 20;
  tsrKernel kernel = nullptr;
  if (tsrGetKernel(artifact, "k", &kernel) != TSR_STATUS_SUCCESS) return 21;
  tsrLaunchParams p{};
  p.grid = {1, 1, 1};
  p.tile = {4, 1, 1};
  void* args[2] = {&p, nullptr};
  if (tsrLaunchKernel(s, kernel, args, 2) != TSR_STATUS_SUCCESS) return 22;
  if (tsrStreamSynchronize(s) != TSR_STATUS_SUCCESS) return 23;
  if (g_called != 4) return 24;

  if (!has_event(a_events, TSR_PROFILE_RUNTIME_API, "tsrCompileArtifact")) return 25;
  if (!has_event(a_events, TSR_PROFILE_RUNTIME_API, "tsrGetKernel")) return 26;
  if (!has_event(a_events, TSR_PROFILE_RUNTIME_API, "tsrLaunchKernel")) return 27;
  if (!has_event(a_events, TSR_PROFILE_DEVICE_ACTIVITY, "tsrMemcpy")) return 28;
  if (!has_event(a_events, TSR_PROFILE_DEVICE_ACTIVITY, "tsrLaunchHostTileKernel")) return 29;
  if (!has_payload(a_events, TSR_PROFILE_DEVICE_ACTIVITY, "\"duration_us\"")) return 30;
  if (!has_payload(a_events, TSR_PROFILE_DEVICE_ACTIVITY, "\"bytes\":64")) return 31;
  if (!has_payload(a_events, TSR_PROFILE_DEVICE_ACTIVITY, "\"memcpy_kind\":\"device_to_device\"")) return 32;

  size_t a_before = a_events.size();
  if (tsrSetProfileEventCallback(cb_b, nullptr) != TSR_STATUS_SUCCESS) return 33;
  tsrGetDeviceCount(&count);
  if (a_events.size() != a_before) return 34;
  if (b_events.empty()) return 35;

  size_t b_before = b_events.size();
  if (tsrSetProfileEventCallback(nullptr, nullptr) != TSR_STATUS_SUCCESS) return 36;
  tsrGetDeviceCount(&count);
  if (b_events.size() != b_before) return 37;

  tsrDestroyKernel(kernel);
  tsrDestroyArtifact(artifact);
  tsrFree(src);
  tsrFree(dst);
  tsrDestroyStream(s);
  tsrShutdown();
  std::printf("OK %zu %zu\n", a_events.size(), b_events.size());
  return 0;
}
"""


_TPROF_ADAPTER_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include "tprof/tessera_runtime_adapter.h"
#include "tprof/tprof_runtime.h"
#include <cstdio>

static void k(void*, uint32_t, uint32_t, uint32_t) {}

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  tprof::config_t cfg;
  tprof::enable(cfg);
  if (!tprof::attach_tessera_runtime_trace(true)) return 2;

  if (tsrInit() != TSR_STATUS_SUCCESS) return 3;
  tsrDevice dev = nullptr;
  if (tsrGetDevice(0, &dev) != TSR_STATUS_SUCCESS) return 4;
  tsrStream s = nullptr;
  if (tsrCreateStream(dev, &s) != TSR_STATUS_SUCCESS) return 5;
  tsrBuffer src = nullptr, dst = nullptr;
  if (tsrMalloc(dev, 32, &src) != TSR_STATUS_SUCCESS) return 6;
  if (tsrMalloc(dev, 32, &dst) != TSR_STATUS_SUCCESS) return 7;
  if (tsrMemcpy(dst, src, 32, TSR_MEMCPY_DEVICE_TO_DEVICE) != TSR_STATUS_SUCCESS) return 8;
  tsrLaunchParams p{};
  p.grid = {1, 1, 1};
  p.tile = {2, 1, 1};
  if (tsrLaunchHostTileKernel(s, &p, (tsrHostKernelFn)k, nullptr) != TSR_STATUS_SUCCESS) return 9;
  if (tsrStreamSynchronize(s) != TSR_STATUS_SUCCESS) return 10;

  tsrFree(src);
  tsrFree(dst);
  tsrDestroyStream(s);
  tsrShutdown();
  tprof::detach_tessera_runtime_trace();
  if (!tprof::export_chrome(argv[1])) return 11;
  tprof::disable();
  std::printf("OK\n");
  return 0;
}
"""


_HARDENING_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include <atomic>
#include <thread>
#include <cstdio>

// (1) Re-entrancy: a callback that re-enters the runtime. The nested emission
// must be suppressed (depth never exceeds 1) — otherwise unbounded recursion.
static thread_local int tl_depth = 0;
static std::atomic<int> g_reentrant_calls{0};
static std::atomic<bool> g_saw_recursion{false};
static void cb_reentrant(TsrProfileEventKind, const char*, const char*, double, void*) {
  if (tl_depth > 0) { g_saw_recursion.store(true); return; }
  ++tl_depth;
  g_reentrant_calls.fetch_add(1);
  int c = 0;
  tsrGetDeviceCount(&c);   // re-enters the runtime; must NOT call us again
  --tl_depth;
}

// (2) Reconfigure-from-within: a callback that resets the callback. Must not
// self-deadlock on the in-flight drain (the program returning is the proof).
static void cb_self_reconfigure(TsrProfileEventKind, const char*, const char*, double, void*) {
  tsrSetProfileEventCallback(nullptr, nullptr);
}

// (3) Concurrent emit + swap: cb_A must NEVER run after the swap to cb_B
// returns — the drain closes the copy-then-free use-after-free window.
static std::atomic<bool> g_swapped{false};
static std::atomic<int> g_a_after_swap{0};
static std::atomic<bool> g_stop{false};
static void cb_A(TsrProfileEventKind, const char*, const char*, double, void*) {
  if (g_swapped.load()) g_a_after_swap.fetch_add(1);
}
static void cb_B(TsrProfileEventKind, const char*, const char*, double, void*) {}

int main() {
  tsrEnableProfiling(1);

  if (tsrSetProfileEventCallback(cb_reentrant, nullptr) != TSR_STATUS_SUCCESS) return 2;
  int c = 0;
  tsrGetDeviceCount(&c);
  tsrSetProfileEventCallback(nullptr, nullptr);
  if (g_saw_recursion.load()) return 10;        // re-entrancy guard failed
  if (g_reentrant_calls.load() < 1) return 11;  // callback never fired

  if (tsrSetProfileEventCallback(cb_self_reconfigure, nullptr) != TSR_STATUS_SUCCESS) return 3;
  tsrGetDeviceCount(&c);                         // hangs here if drain self-deadlocks
  tsrSetProfileEventCallback(nullptr, nullptr);

  if (tsrSetProfileEventCallback(cb_A, nullptr) != TSR_STATUS_SUCCESS) return 4;
  std::thread emitter([]{ int cc = 0; while (!g_stop.load()) tsrGetDeviceCount(&cc); });
  for (int i = 0; i < 100000; ++i) { int cc = 0; tsrGetDeviceCount(&cc); }
  tsrSetProfileEventCallback(cb_B, nullptr);     // drains in-flight cb_A before returning
  g_swapped.store(true);                         // from now cb_A must never run
  for (int i = 0; i < 100000; ++i) { int cc = 0; tsrGetDeviceCount(&cc); }
  g_stop.store(true);
  emitter.join();
  tsrSetProfileEventCallback(nullptr, nullptr);
  if (g_a_after_swap.load() != 0) return 12;     // cb_A ran after swap returned -> UAF window

  std::printf("OK reentrant=%d\n", g_reentrant_calls.load());
  return 0;
}
"""


def test_runtime_profile_callback_hardening(tmp_path: Path) -> None:
    """Re-entrancy guard + callback-lifetime drain (no recursion, no deadlock when
    a callback reconfigures the callback, no use-after-free under concurrent swap)."""
    if not RUNTIME_LIB.is_file() or _CXX is None or not _runtime_lib_has_profile_callback():
        pytest.skip(
            "requires built libtessera_runtime.a with tsrSetProfileEventCallback "
            "(run `ninja -C build tessera_runtime`) and a C++ compiler"
        )
    src_path = tmp_path / "profile_hardening.cpp"
    bin_path = tmp_path / "profile_hardening"
    src_path.write_text(_HARDENING_HARNESS)
    compile_cmd = [
        _CXX, "-std=c++17", "-O2", "-I", str(RUNTIME_INCLUDE),
        str(src_path), str(RUNTIME_LIB), "-lpthread", "-o", str(bin_path),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:4000]
    # A hang (self-deadlock) trips the timeout; a non-zero exit pinpoints which
    # invariant broke (10=recursion, 11=no fire, 12=UAF window).
    run = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, f"exit {run.returncode}\nstdout:{run.stdout}\nstderr:{run.stderr}"
    assert run.stdout.startswith("OK "), run.stdout


def test_runtime_profile_callback_functional_trace_spine(tmp_path: Path) -> None:
    if not RUNTIME_LIB.is_file() or _CXX is None or not _runtime_lib_has_profile_callback():
        pytest.skip(
            "requires built libtessera_runtime.a with tsrSetProfileEventCallback "
            "(run `ninja -C build tessera_runtime`) and a C++ compiler"
        )
    src_path = tmp_path / "profile_trace.cpp"
    bin_path = tmp_path / "profile_trace"
    src_path.write_text(_PROFILE_HARNESS)
    compile_cmd = [
        _CXX,
        "-std=c++17",
        "-O2",
        "-I",
        str(RUNTIME_INCLUDE),
        str(src_path),
        str(RUNTIME_LIB),
        "-lpthread",
        "-o",
        str(bin_path),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:4000]

    run = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=30)

    assert run.returncode == 0, f"exit {run.returncode}\nstdout:{run.stdout}\nstderr:{run.stderr}"
    assert run.stdout.startswith("OK "), run.stdout


def test_tprof_adapter_maps_runtime_callback_to_trace_categories(tmp_path: Path) -> None:
    if not RUNTIME_LIB.is_file() or _CXX is None or not _runtime_lib_has_profile_callback():
        pytest.skip(
            "requires built libtessera_runtime.a with tsrSetProfileEventCallback "
            "(run `ninja -C build tessera_runtime`) and a C++ compiler"
        )

    src_path = tmp_path / "tprof_adapter.cpp"
    bin_path = tmp_path / "tprof_adapter"
    trace_path = tmp_path / "adapter.trace.json"
    src_path.write_text(_TPROF_ADAPTER_HARNESS)
    compile_cmd = [
        _CXX,
        "-std=c++17",
        "-O2",
        "-I",
        str(RUNTIME_INCLUDE),
        "-I",
        str(TPROF_INCLUDE),
        str(src_path),
        *(str(path) for path in TPROF_SOURCES),
        str(RUNTIME_LIB),
        "-lpthread",
        "-o",
        str(bin_path),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:4000]

    run = subprocess.run([str(bin_path), str(trace_path)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, f"exit {run.returncode}\nstdout:{run.stdout}\nstderr:{run.stderr}"

    trace = trace_path.read_text()
    assert '"cat": "runtime_api"' in trace
    assert '"cat": "device_activity"' in trace
    assert '"name": "tsrMemcpy"' in trace
    assert '"name": "tsrLaunchHostTileKernel"' in trace
