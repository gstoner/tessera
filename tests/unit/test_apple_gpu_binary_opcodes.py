"""The Apple GPU elementwise-binary opcode table, across all three implementations.

One opcode table (`apple_gpu_envelope._APPLE_GPU_BINARY_OPCODES`) has three
implementations that must agree:

* `runtime._apple_gpu_binary_numpy` — the declared host reference.
* `apple_gpu_runtime.mm::mpsg_binary_node` — Metal/MPSGraph, Darwin only.
* `apple_gpu_runtime_stub.cpp` — the portable C++ stub compiled on every
  non-Darwin host so the C symbol exists.

Until 2026-08-19 the stub's switch stopped at opcode 8 and its `default:` arm
assigned `out[i] = x`, so `mod`(9), `floor_div`(10), the six comparisons(11-16),
and the logical/bitwise ops(17-22) returned the LEFT operand on every non-Apple
host — CI and the Strix Halo box included. It was not a visible envelope miss:
because the symbol exists, `_apple_gpu_dispatch_mpsgraph_binary` takes the
kernel branch rather than its numpy fallback, so the wrong value came back as if
computed. Observed on CI as `tessera.floor_div` with `{"scalar": 2.0}` returning
`[1,2,4]` where the truth is `[0,1,2]`.

Testing this is awkward because the Mac loads the real Metal symbol and so never
executes the stub. The coverage here is therefore layered:

* `TestHostReference` — the dispatcher against the numpy reference for every
  opcode, on both lanes. Runs everywhere; catches dispatch-level regressions.
* `TestStubSource` — structural gates on the stub source. Run on ANY host,
  including the Mac that cannot execute it, so a newly added opcode that the
  stub does not implement fails here rather than silently on Linux.
* `TestStubCompiledBehaviour` — extracts the stub's opcode switch and error
  channel VERBATIM, compiles them with the host C++ compiler, runs them, and
  compares every opcode to the numpy reference. This is the only test that
  executes the stub's real semantics; it skips when no compiler is available.
* `TestKernelErrorFallback` — a kernel that reports failure through the
  last-error channel must route to the host reference, never return its buffer.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tessera.runtime as runtime_mod
from tessera.compiler.apple_gpu_envelope import _APPLE_GPU_BINARY_OPCODES
from tessera.runtime import (
    _apple_gpu_binary_numpy,
    _apple_gpu_dispatch_mpsgraph_binary as dispatch_binary,
)

ROOT = Path(__file__).resolve().parents[2]
STUB = (
    ROOT
    / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime_stub.cpp"
)

#: Inputs chosen to separate the implementations rather than merely exercise
#: them: a negative numerator distinguishes floor-mod from C `fmod` and
#: floor-division from truncation, and a zero distinguishes the logical ops from
#: the bitwise ones.
X = np.array([1.0, 2.0, 4.0, -3.0, 0.0, 7.0], dtype=np.float32)
Y = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 3.0], dtype=np.float32)

OP_NAMES = sorted(_APPLE_GPU_BINARY_OPCODES)

#: Opcode 6 is `silu_mul`. The AppleGPUToMPS lowering emits it directly
#: (`SiluMulToAppleGPU.cpp`), so it is absent from the Python table but must
#: still be implemented by the stub — it is covered by the compiled test below.
LOWERING_ONLY_OPCODES = {6}


def _live_metal() -> bool:
    return (
        sys.platform == "darwin"
        and runtime_mod._apple_gpu_mpsgraph_binary_f32() is not None
    )


@pytest.fixture(params=[
    "host_reference",
    pytest.param("live_kernel", marks=pytest.mark.hardware_apple_gpu),
])
def lane(request, monkeypatch):
    """Both dispatcher lanes; they must agree on every opcode."""
    if request.param == "host_reference":
        monkeypatch.setattr(runtime_mod, "_apple_gpu_mpsgraph_binary_f32", lambda: None)
    return request.param


class TestHostReference:
    @pytest.mark.parametrize("op_name", OP_NAMES)
    def test_dispatcher_matches_the_declared_reference(self, op_name, lane):
        got = dispatch_binary(op_name, [X, Y], {}, np)
        want = _apple_gpu_binary_numpy(op_name, X, Y, np)
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32),
            rtol=1e-6, atol=1e-6,
        )


class TestStubSource:
    """Structural gates that run on a host which cannot execute the stub."""

    @staticmethod
    def _binary_fn_source() -> str:
        text = STUB.read_text(encoding="utf-8")
        start = text.index(
            'extern "C" void tessera_apple_gpu_mpsgraph_binary_f32('
        )
        depth, i = 0, text.index("{", start)
        j = i
        while True:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        return text[start:j + 1]

    def test_every_declared_opcode_has_a_case(self):
        """The drift gate: adding an opcode to the Python table without
        implementing it in the stub fails here, on any host."""
        source = self._binary_fn_source()
        missing = [
            name for name, code in sorted(_APPLE_GPU_BINARY_OPCODES.items())
            if f"case {code}:" not in source
        ]
        assert not missing, (
            "opcodes declared in _APPLE_GPU_BINARY_OPCODES but not implemented "
            f"in apple_gpu_runtime_stub.cpp: {missing}"
        )

    @pytest.mark.parametrize("code", sorted(LOWERING_ONLY_OPCODES))
    def test_lowering_only_opcode_has_a_case(self, code):
        assert f"case {code}:" in self._binary_fn_source()

    def test_default_arm_does_not_return_an_operand(self):
        """The exact regression: `default: out[i] = x;` made an unimplemented
        opcode indistinguishable from a computed result."""
        source = self._binary_fn_source()
        assert "out[i] = x; break;" not in source.replace("case 0:", "")
        assert "default:" in source
        tail = source[source.index("default:"):]
        assert "stub_set_last_error" in tail, "default arm must report an error"

    def test_unsupported_opcode_is_rejected_before_writing_output(self):
        source = self._binary_fn_source()
        guard = source.index("stub_binary_opcode_supported")
        loop = source.index("for (int64_t i = 0;")
        assert guard < loop, "the opcode guard must run before the compute loop"


def _cxx() -> str | None:
    for candidate in ("c++", "clang++", "g++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _extract_verbatim() -> str:
    """Slice the stub's error channel + opcode switch out of the real file.

    Sliced, never retyped: a copy would drift from the code it claims to test.
    """
    text = STUB.read_text(encoding="utf-8")

    def balanced_end(idx: int) -> int:
        depth, j = 0, text.index("{", idx)
        while True:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1

    def find(anchor: str) -> int:
        try:
            return text.index(anchor)
        except ValueError:
            pytest.fail(
                f"apple_gpu_runtime_stub.cpp no longer contains {anchor!r}. This "
                "test compiles that region verbatim; if the stub was "
                "restructured, update the anchors here rather than dropping the "
                "coverage -- it is the only test that executes the stub."
            )

    a = find("// ── Portable last-error state")
    b = balanced_end(find('extern "C" void tessera_apple_gpu_mpsgraph_binary_f32('))
    getters_start = find(
        'extern "C" int32_t tessera_apple_gpu_last_error_kind(void) {'
    )
    getters_end = balanced_end(
        find('extern "C" void tessera_apple_gpu_clear_last_error(void) {')
    )
    return text[a:b] + "\n\n" + text[getters_start:getters_end]


class TestStubCompiledBehaviour:
    """Compile and RUN the stub's real opcode switch, on any host with a C++
    compiler. Without this the stub's semantics are only ever exercised on a
    machine none of us runs the suite on."""

    @pytest.fixture(scope="class")
    def stub_lib(self, tmp_path_factory):
        cxx = _cxx()
        if cxx is None:
            pytest.skip("no C++ compiler available to build the portable stub")
        work = tmp_path_factory.mktemp("apple_stub")
        source = work / "stub_extract.cpp"
        source.write_text(
            "#include <algorithm>\n#include <atomic>\n#include <cmath>\n"
            "#include <cstdint>\n#include <cstring>\n#include <limits>\n"
            "#include <string>\n#include <vector>\n\n" + _extract_verbatim(),
            encoding="utf-8",
        )
        lib = work / ("stub_extract" + (".dylib" if sys.platform == "darwin" else ".so"))
        result = subprocess.run(
            [cxx, "-std=c++17", "-O1", "-shared", "-fPIC",
             str(source), "-o", str(lib)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.fail(
                "the stub's extracted opcode switch did not compile:\n"
                + result.stderr[-2000:]
            )
        handle = ctypes.CDLL(str(lib))
        handle.tessera_apple_gpu_mpsgraph_binary_f32.argtypes = [
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        handle.tessera_apple_gpu_mpsgraph_binary_f32.restype = None
        handle.tessera_apple_gpu_last_error_kind.argtypes = []
        handle.tessera_apple_gpu_last_error_kind.restype = ctypes.c_int32
        handle.tessera_apple_gpu_last_error_message.argtypes = []
        handle.tessera_apple_gpu_last_error_message.restype = ctypes.c_char_p
        handle.tessera_apple_gpu_clear_last_error.argtypes = []
        handle.tessera_apple_gpu_clear_last_error.restype = None
        return handle

    @staticmethod
    def _run(handle, code: int, a: np.ndarray, b: np.ndarray):
        af = np.ascontiguousarray(a, np.float32)
        bf = np.ascontiguousarray(b, np.float32)
        out = np.zeros(af.size, np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        handle.tessera_apple_gpu_clear_last_error()
        handle.tessera_apple_gpu_mpsgraph_binary_f32(
            ctypes.c_int32(code),
            af.ctypes.data_as(fp), bf.ctypes.data_as(fp), out.ctypes.data_as(fp),
            ctypes.c_int64(af.size),
        )
        return out, int(handle.tessera_apple_gpu_last_error_kind())

    @pytest.mark.parametrize("op_name", OP_NAMES)
    def test_compiled_stub_matches_the_declared_reference(self, stub_lib, op_name):
        code = _APPLE_GPU_BINARY_OPCODES[op_name]
        got, kind = self._run(stub_lib, code, X, Y)
        assert kind == 0, f"{op_name} reported error kind {kind}"
        want = np.asarray(_apple_gpu_binary_numpy(op_name, X, Y, np), np.float32)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    def test_compiled_stub_implements_the_lowering_only_silu_mul(self, stub_lib):
        got, kind = self._run(stub_lib, 6, X, Y)
        assert kind == 0
        np.testing.assert_allclose(
            got, X * (Y / (1.0 + np.exp(-Y))), rtol=1e-6, atol=1e-6
        )

    def test_unknown_opcode_is_diagnosed_and_poisons_the_output(self, stub_lib):
        """Never `out[i] = x` again: an unimplemented opcode must report through
        the last-error channel, and must not leave a plausible-looking value."""
        unknown = max(_APPLE_GPU_BINARY_OPCODES.values()) + 1
        got, kind = self._run(stub_lib, unknown, X, Y)
        assert kind != 0, "unimplemented opcode reported success"
        assert np.all(np.isnan(got)), "unimplemented opcode produced usable-looking values"
        assert not np.allclose(got, X, equal_nan=True), "returned the left operand"
        message = stub_lib.tessera_apple_gpu_last_error_message().decode()
        assert "tessera_apple_gpu_mpsgraph_binary_f32" in message
        assert str(unknown) in message

    def test_clear_resets_the_channel(self, stub_lib):
        self._run(stub_lib, max(_APPLE_GPU_BINARY_OPCODES.values()) + 1, X, Y)
        assert int(stub_lib.tessera_apple_gpu_last_error_kind()) != 0
        stub_lib.tessera_apple_gpu_clear_last_error()
        assert int(stub_lib.tessera_apple_gpu_last_error_kind()) == 0


class TestKernelErrorFallback:
    """A kernel that reports failure must not have its buffer believed."""

    def test_reported_error_routes_to_the_host_reference(self, monkeypatch):
        def _garbage_kernel(op, a_ptr, b_ptr, out_ptr, n):
            # The dispatcher passes ctypes objects, so unwrap `.value`.
            count = n.value if hasattr(n, "value") else int(n)
            for i in range(count):
                out_ptr[i] = -12345.0

        monkeypatch.setattr(
            runtime_mod, "_apple_gpu_mpsgraph_binary_f32", lambda: _garbage_kernel
        )
        monkeypatch.setattr(runtime_mod, "_apple_gpu_arm_gpu_error", lambda: None)
        monkeypatch.setattr(
            runtime_mod, "_apple_gpu_consume_gpu_error",
            lambda: "unsupported binary opcode 23 on the non-Darwin portable stub",
        )
        got = dispatch_binary("tessera.lt", [X, Y], {}, np)
        np.testing.assert_allclose(
            np.asarray(got, np.float32),
            np.asarray(_apple_gpu_binary_numpy("tessera.lt", X, Y, np), np.float32),
        )
