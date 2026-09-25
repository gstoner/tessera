"""Spectral FFT arbiter retarget — the ts-spectral-opt `lower-*-to-target-ir`
seam pointed at the D1 candidate arbiter.

Verifies the shipped Stockham kernel is registered as an F4-gated candidate for
the ``spectral_fft`` op-kind: the real compiled CPU kernel matches ``numpy.fft``
through the arbiter, a wrong candidate is refused even at a higher tier, and the
arbiter falls back honestly to the reference when nothing applies.  The CPU lane
compiles the shipped ``TargetHooks/CPU/StockhamRadix4.cpp``; if no C++ toolchain
is present it declines and the real-kernel assertions are skipped.
"""
from __future__ import annotations

import ctypes
import numpy as np
import pytest

from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import Candidate, Tier
from tessera.compiler.emit import spectral_candidates as SC
from tessera.compiler.emit.spectral_candidates import (
    OP_SPECTRAL_FFT,
    SpectralFFTRegion,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = {k: list(v) for k, v in C._CANDIDATES.items()}
    yield
    C._CANDIDATES.clear()
    C._CANDIDATES.update(saved)


def _cpu():
    return next(c for c in C.candidates_for("cpu", OP_SPECTRAL_FFT)
               if c.name == "cpu_stockham")


def test_op_kind_registered():
    assert OP_SPECTRAL_FFT in C._OP_KIND_VERIFY


def test_cpu_stockham_matches_numpy_fft():
    cpu = _cpu()
    if not cpu.available():
        pytest.skip("no C++ toolchain to build the shipped CPU kernel")
    for n in (64, 128, 256, 512, 1024):
        for sign in (-1, 1):
            reg = SpectralFFTRegion(n, sign=sign)
            x = reg.probe_input(0)
            out, tag = cpu.run(reg, x)
            assert tag == "cpu_stockham"
            assert np.allclose(out, reg.reference(x), atol=1e-2 * max(1, n / 64))


def test_arbiter_picks_verified_cpu_kernel():
    cpu = _cpu()
    if not cpu.available():
        pytest.skip("no C++ toolchain")
    reg = SpectralFFTRegion(256, sign=-1)
    assert C.verify_candidate(cpu, reg) is True
    win = C.arbitrate(reg, OP_SPECTRAL_FFT, "cpu")
    assert win is not None and win.name == "cpu_stockham"


def test_wrong_candidate_is_f4_rejected_even_at_higher_tier():
    if not _cpu().available():
        pytest.skip("no C++ toolchain")

    class _Wrong(Candidate):
        name, tier, target, op = "wrong", Tier.HAND_TUNED, "cpu", OP_SPECTRAL_FFT

        def run(self, region, x, *a, **k):
            return np.full(region.n, 9.0, np.complex64), "wrong_tag"

    C.register_candidate(_Wrong())
    reg = SpectralFFTRegion(128, sign=-1)
    assert C.verify_candidate(_Wrong(), reg) is False
    # Higher tier but wrong → arbiter still selects the correct CPU kernel.
    win = C.arbitrate(reg, OP_SPECTRAL_FFT, "cpu")
    assert win is not None and win.name == "cpu_stockham"


def test_run_arbitrated_end_to_end():
    if not _cpu().available():
        pytest.skip("no C++ toolchain")
    reg = SpectralFFTRegion(512, sign=-1)
    x = reg.probe_input(3)
    out, tag = C.run_arbitrated(reg, OP_SPECTRAL_FFT, "cpu", x)
    assert tag == "cpu_stockham"
    assert np.allclose(out, reg.reference(x), atol=1e-1)


def test_reference_fallback_when_no_candidate():
    reg = SpectralFFTRegion(64, sign=-1)
    x = reg.probe_input(0)
    # No candidates registered for this target → honest reference fallback.
    out, tag = C.run_arbitrated(reg, OP_SPECTRAL_FFT, "no_such_target", x)
    assert tag == "reference"
    assert np.allclose(out, reg.reference(x))


def test_canonical_rocm_loader_never_falls_back_to_source_compile(monkeypatch):
    saved = SC._libs.pop("amd_prebuilt", None)
    monkeypatch.setattr(SC, "_prebuilt_amd_paths", lambda: ())
    monkeypatch.setattr(
        SC,
        "_amd_source_lib",
        lambda: (_ for _ in ()).throw(
            AssertionError("canonical ROCm FFT attempted a source build")
        ),
    )
    try:
        assert SC._amd_lib() is None
    finally:
        if saved is not None:
            SC._libs["amd_prebuilt"] = saved


class _FakeAmdPackage:
    ts_fft_plan_create_for_artifact_amd = object()
    ts_fft_plan_artifact_digest_amd = object()
    ts_fft_plan_execute_hostptr_batch_amd = object()
    ts_fft_plan_workspace_elems_amd = object()
    ts_fft_plan_destroy_amd = object()

    def __init__(self, arch: bytes):
        self._arch = arch

    @staticmethod
    def ts_fft_package_abi_amd():
        return b"tessera.rocm.fft.plan.v1"

    @staticmethod
    def ts_spectral_composite_package_abi_amd():
        return b"tessera.rocm.spectral_composite.v7"

    def ts_spectral_composite_arch_amd(self):
        return self._arch


def _as_device(monkeypatch, live, compile_arch=None):
    """Run the loaders as if ``live`` were the selected HIP device."""
    monkeypatch.setattr(SC, "_spectral_device_arch", lambda: live)
    monkeypatch.setattr(
        SC, "_spectral_compile_arch", lambda: compile_arch or live
    )


def _prebuilt(monkeypatch, tmp_path, *stamps):
    """Expose one fake prebuilt package per stamp, in search order."""
    packages = []
    fakes = {}
    for index, stamp in enumerate(stamps):
        path = tmp_path / f"p{index}" / "libtessera_spectral_rocm.so"
        path.parent.mkdir()
        path.touch()
        packages.append(path)
        fakes[str(path)] = _FakeAmdPackage(stamp)
    monkeypatch.setattr(SC, "_libs", {})
    monkeypatch.setattr(SC, "_prebuilt_amd_paths", lambda: tuple(packages))
    monkeypatch.setattr(SC.ctypes, "CDLL", lambda path: fakes[path])
    monkeypatch.setattr(SC, "_configure_amd_lib", lambda lib: lib)
    return [fakes[str(path)] for path in packages]


@pytest.mark.parametrize("arch", [b"gfx1200", b"gfx1250", b"unknown"])
def test_rocm_fft_fallback_is_not_a_composite_candidate(monkeypatch, tmp_path, arch):
    (fake,) = _prebuilt(monkeypatch, tmp_path, arch)
    _as_device(monkeypatch, "gfx1151")
    assert SC._amd_lib() is fake  # Architecture-neutral FFT ABI remains usable.
    assert SC._amd_composite_lib() is None
    assert SC._amd_device_lib() is None  # ...but never a foreign chip's code object


def test_rocm_composite_loader_accepts_exact_gfx1151_package(monkeypatch, tmp_path):
    (fake,) = _prebuilt(monkeypatch, tmp_path, b"gfx1151")
    _as_device(monkeypatch, "gfx1151")
    assert SC._amd_composite_lib() is fake
    assert SC._libs["amd_composite:gfx1151"] is fake


def test_rocm_composite_loader_accepts_exact_gfx1201_package(monkeypatch, tmp_path):
    # Regression: #830 consulted the prebuilt only for gfx1151, so Tajasarus's
    # gfx1201-stamped build became unreachable and its refusal (which names
    # the arch) was reported as a skip.
    foreign, fake = _prebuilt(monkeypatch, tmp_path, b"gfx1151", b"gfx1201")
    _as_device(monkeypatch, "gfx1201")
    assert SC._amd_composite_lib() is fake
    assert SC._amd_device_lib() is fake
    assert SC._amd_lib() is foreign  # legacy preference is not a device choice


def test_rocm_composite_refuses_a_foreign_stamp_on_each_chip(monkeypatch, tmp_path):
    _prebuilt(monkeypatch, tmp_path, b"gfx1151")
    _as_device(monkeypatch, "gfx1201")
    assert SC._amd_composite_lib() is None
    assert SC._amd_device_lib() is None


def test_rocm_gfx1151_refuses_a_gfx1201_only_prebuilt(monkeypatch, tmp_path):
    _prebuilt(monkeypatch, tmp_path, b"gfx1201")
    _as_device(monkeypatch, "gfx1151")
    assert SC._amd_composite_lib() is None
    assert SC._amd_device_lib() is None


def test_rocm_composite_feature_target_has_no_qualified_prebuilt(monkeypatch, tmp_path):
    # A feature-qualified target has no prebuilt stamped for it, and the
    # source hook compiles only the Stockham FFT, so there is no composite.
    _prebuilt(monkeypatch, tmp_path, b"gfx1151")
    _as_device(monkeypatch, "gfx1151", "gfx1151:xnack-")
    monkeypatch.setattr(SC, "_amd_source_lib", lambda: pytest.fail("source is not a composite"))
    assert SC._amd_composite_lib() is None


def test_rocm_candidate_prefers_the_live_chips_prebuilt(monkeypatch, tmp_path):
    _, fake = _prebuilt(monkeypatch, tmp_path, b"gfx1151", b"gfx1201")
    _as_device(monkeypatch, "gfx1201")
    monkeypatch.setattr(SC, "_amd_source_lib", lambda: pytest.fail("prebuilt exists"))
    assert SC._amd_candidate_lib() is fake


def test_rocm_composite_launch_rechecks_architecture(monkeypatch):
    from tessera.compiler import scheduled_spectral

    fake = _FakeAmdPackage(b"gfx1200")
    monkeypatch.setattr(SC, "_amd_composite_lib", lambda: fake)
    monkeypatch.setattr(
        scheduled_spectral,
        "validate_scheduled_spectral_metadata",
        lambda _metadata, input_shapes: object(),
    )
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        SC.run_rocm_spectral_composite({}, [])


# ── compile-cache hit must cost nothing (no scratch dir, no device probe) ────


def test_cpu_lib_creates_one_scratch_directory_for_the_whole_process(tmp_path,
                                                                    monkeypatch):
    """`_compile` serves every call after the first from `_libs`, so a scratch
    directory made BEFORE that check is abandoned empty on every later call --
    unbounded in TMPDIR, and (measured on this host) 64 us of mkdtemp per call
    on a path the composed STFT lane runs once per frame."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(SC, "_libs", {})
    monkeypatch.setattr(SC.tempfile, "tempdir", None, raising=False)
    for _ in range(5):
        SC._cpu_lib()
    made = list(tmp_path.glob("tessera_spectral_cpu_*"))
    assert len(made) <= 1


def test_cpu_lib_returns_the_cached_handle_without_recompiling(monkeypatch):
    SC._cpu_lib()
    if SC._libs.get("cpu") is None:
        pytest.skip("no C++ toolchain: the CPU lane declines, nothing to cache")
    monkeypatch.setattr(SC, "_compile", lambda *_a, **_k: pytest.fail(
        "cache hit must not reach the compiler"))
    monkeypatch.setattr(SC.tempfile, "mkdtemp", lambda *_a, **_k: pytest.fail(
        "cache hit must not create a scratch directory"))
    assert SC._cpu_lib() is SC._libs["cpu"]


def test_rocm_availability_probe_runs_once_per_process(monkeypatch):
    """`available()` is called per arbitration -- per composed STFT frame -- and
    its 4-point device transform measured 0.371 ms per call on gfx1151. The
    answer cannot change within a process, so it is probed once."""
    calls = []

    class _Lib:
        def ts_fft_stockham_amd_hostptr(self, _input, output, _n, _sign):
            calls.append(1)
            expected = np.ones(4, np.complex64)
            ctypes.memmove(output, expected.ctypes.data, expected.nbytes)
            return 0

    lib = _Lib()
    monkeypatch.setattr(SC, "_amd_candidate_lib", lambda: lib)
    monkeypatch.setattr(SC, "_amd_probe", {})
    candidate = SC.RocmStockhamFFTCandidate()
    assert [candidate.available() for _ in range(5)] == [True] * 5
    assert len(calls) == 1


def test_rocm_availability_rejects_success_status_with_wrong_output(monkeypatch):
    class _Lib:
        def ts_fft_stockham_amd_hostptr(self, _input, output, _n, _sign):
            wrong = np.zeros(4, np.complex64)
            ctypes.memmove(output, wrong.ctypes.data, wrong.nbytes)
            return 0

    monkeypatch.setattr(SC, "_amd_candidate_lib", lambda: _Lib())
    monkeypatch.setattr(SC, "_amd_probe", {})
    assert not SC.RocmStockhamFFTCandidate().available()


def test_rocm_source_candidate_uses_the_live_arch_not_a_foreign_prebuilt(monkeypatch):
    marker = object()
    _as_device(monkeypatch, "gfx1201")
    monkeypatch.setattr(SC, "_amd_prebuilt_exact", lambda _arch: None)
    monkeypatch.setattr(SC, "_amd_source_lib", lambda: marker)
    monkeypatch.setattr(SC, "_amd_lib", lambda: pytest.fail("foreign prebuilt loaded"))
    assert SC._amd_candidate_lib() is marker


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1200", "gfx1250"])
def test_rocm_source_candidate_accepts_other_live_architectures(monkeypatch, arch):
    marker = object()
    _as_device(monkeypatch, arch)
    monkeypatch.setattr(SC, "_amd_prebuilt_exact", lambda _arch: None)
    monkeypatch.setattr(SC, "_amd_source_lib", lambda: marker)
    monkeypatch.setattr(SC, "_amd_lib", lambda: pytest.fail("foreign prebuilt loaded"))
    assert SC._amd_candidate_lib() is marker


def test_rocm_feature_qualified_gfx1151_does_not_use_unqualified_prebuilt(monkeypatch):
    marker = object()
    _as_device(monkeypatch, "gfx1151", "gfx1151:xnack-")
    monkeypatch.setattr(SC, "_amd_prebuilt_exact", lambda _arch: pytest.fail("prebuilt consulted"))
    monkeypatch.setattr(SC, "_amd_source_lib", lambda: marker)
    monkeypatch.setattr(SC, "_amd_lib", lambda: pytest.fail("unqualified prebuilt loaded"))
    assert SC._amd_candidate_lib() is marker


def test_rocm_source_compiler_targets_the_selected_device(monkeypatch, tmp_path):
    marker = object()
    captured = []

    def compile_source(key, argv, output):
        captured.append((key, argv, output))
        return marker

    monkeypatch.setattr(SC, "_spectral_compile_arch", lambda: "gfx1201")
    monkeypatch.setattr(SC, "_libs", {})
    monkeypatch.setattr(SC.shutil, "which", lambda _tool: "/opt/rocm/bin/hipcc")
    monkeypatch.setattr(SC, "_build_dir", lambda _prefix: str(tmp_path))
    monkeypatch.setattr(SC, "_compile", compile_source)
    monkeypatch.setattr(SC, "_configure_amd_lib", lambda lib: lib)
    assert SC._amd_source_lib() is marker
    assert len(captured) == 1
    assert captured[0][0] == "amd_source:gfx1201"
    assert "--offload-arch=gfx1201" in captured[0][1]


def test_rocm_source_compiler_keeps_feature_qualified_target(monkeypatch, tmp_path):
    captured = []

    def compile_source(key, argv, output):
        captured.append((key, argv, output))
        return object()

    monkeypatch.setattr(SC, "_spectral_compile_arch", lambda: "gfx1201:xnack-")
    monkeypatch.setattr(SC, "_libs", {})
    monkeypatch.setattr(SC.shutil, "which", lambda _tool: "/opt/rocm/bin/hipcc")
    monkeypatch.setattr(SC, "_build_dir", lambda _prefix: str(tmp_path))
    monkeypatch.setattr(SC, "_compile", compile_source)
    monkeypatch.setattr(SC, "_configure_amd_lib", lambda lib: lib)
    assert SC._amd_source_lib() is not None
    assert captured[0][0] == "amd_source:gfx1201:xnack-"
    assert "--offload-arch=gfx1201:xnack-" in captured[0][1]


def test_rocm_spectral_refuses_explicit_arch_mismatch(monkeypatch):
    from tessera import runtime as rt

    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setenv("TESSERA_ROCM_ARCH", "gfx1151")
    assert SC._spectral_device_arch() is None
    monkeypatch.delenv("TESSERA_ROCM_ARCH")
    monkeypatch.setenv("TESSERA_ROCM_CHIP", "gfx1151")
    assert SC._spectral_device_arch() is None


def test_rocm_spectral_accepts_feature_qualified_matching_arch(monkeypatch):
    from tessera import runtime as rt

    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setenv("TESSERA_ROCM_ARCH", "gfx1201:xnack-")
    monkeypatch.delenv("TESSERA_ROCM_CHIP", raising=False)
    assert SC._spectral_device_arch() == "gfx1201"
    assert SC._spectral_compile_arch() == "gfx1201:xnack-"


def test_rocm_availability_probe_memoizes_a_failed_probe_too(monkeypatch):
    calls = []

    class _Lib:
        def ts_fft_stockham_amd_hostptr(self, *_a):
            calls.append(1)
            return 1                       # device present but transform failed

    lib = _Lib()
    monkeypatch.setattr(SC, "_amd_candidate_lib", lambda: lib)
    monkeypatch.setattr(SC, "_amd_probe", {})
    candidate = SC.RocmStockhamFFTCandidate()
    assert [candidate.available() for _ in range(3)] == [False] * 3
    assert len(calls) == 1


class _FakePlanLib:
    """Counts plan creation; each handle remembers the image that made it."""

    def __init__(self, arch: str):
        self.arch = arch
        self.created: list[int] = []

    def ts_fft_plan_create_for_artifact_amd(self, n, sign, digest, out):
        handle = 0x1000 + len(self.created) + (0x100000 if self.arch == "gfx1201" else 0)
        self.created.append(handle)
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = handle
        return 0

    def ts_fft_plan_destroy_amd(self, handle):
        pass

    def ts_spectral_composite_plan_create_amd(self, digest, workspace, out):
        return self.ts_fft_plan_create_for_artifact_amd(0, 0, digest, out)

    def ts_spectral_composite_plan_destroy_amd(self, handle):
        pass

    def ts_spectral_composite_plan_digest_amd(self, handle):
        return b"d" * 64

    def ts_spectral_composite_plan_workspace_bytes_amd(self, handle):
        return 0


def test_rocm_fft_plans_are_scoped_to_the_selected_devices_image(monkeypatch):
    # Codex review on #833: the key once held only (n, sign, digest), so after
    # switching the selected device a cached plan from the other chip's image
    # came back before the device-aware image lookup ever ran.
    libs = {"gfx1151": _FakePlanLib("gfx1151"), "gfx1201": _FakePlanLib("gfx1201")}
    live = ["gfx1151"]
    monkeypatch.setattr(SC, "_amd_device_lib", lambda: libs[live[0]])
    monkeypatch.setattr(SC, "_rocm_plan_cache", SC.collections.OrderedDict())
    digest = "a" * 64
    first_lib, first = SC._rocm_plan(64, -1, digest)
    assert first_lib is libs["gfx1151"]
    assert SC._rocm_plan(64, -1, digest) == (first_lib, first)  # a cache hit
    live[0] = "gfx1201"
    second_lib, second = SC._rocm_plan(64, -1, digest)
    assert second_lib is libs["gfx1201"]
    assert second.value != first.value
    assert len(libs["gfx1151"].created) == len(libs["gfx1201"].created) == 1


def test_rocm_composite_plans_and_their_fft_plans_share_one_image(monkeypatch):
    a, b = _FakePlanLib("gfx1151"), _FakePlanLib("gfx1201")
    monkeypatch.setattr(SC, "_rocm_plan_cache", SC.collections.OrderedDict())
    monkeypatch.setattr(SC, "_rocm_composite_plan_cache", SC.collections.OrderedDict())
    monkeypatch.setattr(SC, "_amd_device_lib", lambda: pytest.fail("composite passes its image"))
    contract = {"schedule_digest": "d" * 64, "workspace_bytes": 0}
    plan_a = SC._rocm_composite_plan(contract, a)
    plan_b = SC._rocm_composite_plan(contract, b)
    assert plan_a.value != plan_b.value  # one digest, two images, two plans
    assert SC._rocm_composite_plan(contract, a).value == plan_a.value
    fft_lib, _ = SC._rocm_plan(32, 1, "e" * 64, b)
    assert fft_lib is b
