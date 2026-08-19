"""One loaded image of the Apple GPU runtime, not two.

`_prebuilt_candidate` used to decide whether a prebuilt dylib was current by
`ctypes.CDLL`-ing it and probing for sentinel symbols. Loading is not a
read-only probe: it registers the runtime's Objective-C classes process-wide,
and `continue`-ing past a stale candidate does not unregister them — the ObjC
runtime pins an image that has defined classes, so there is nothing to undo. The
stale candidate stayed resident, the from-source dylib compiled next registered
the same classes again, and every run printed:

    objc[...]: Class TesseraMlpkgPipeline is implemented in both
    build/.../libTesseraAppleRuntime.dylib and
    /tmp/tessera_apple_gpu_runtime/libtessera_apple_gpu_runtime.<stamp>.dylib
    ... One of the duplicates must be removed or renamed.

Measured on this Mac before the fix: two images resident after one
`_load_apple_gpu_runtime()`. That is not only noise. Two images of one runtime
hold independent copies of every file-static — including the thread_local
last-error channel that `_apple_gpu_run_checked` reads to decide whether a
kernel failed — so which image a symbol resolves to changes what the caller
observes.

Staleness is now decided from the file's symbol table, so a stale candidate is
never loaded at all.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tessera._apple_gpu_dispatch as dispatch

def _cc() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _build_dylib(tmp_path, name: str, symbols) -> "object":
    """A real shared library exporting exactly ``symbols``."""
    compiler = _cc()
    if compiler is None:
        pytest.skip("no C compiler to build the fixture library")
    src = tmp_path / f"{name}.c"
    src.write_text(
        "\n".join(f"int {sym}(void) {{ return 0; }}" for sym in symbols) + "\n"
    )
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib = tmp_path / f"{name}{suffix}"
    result = subprocess.run(
        [compiler, "-shared", "-fPIC", str(src), "-o", str(lib)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"could not build the fixture library: {result.stderr[-300:]}")
    return lib


def _loaded_runtime_images() -> list[str]:
    """Apple-runtime Mach-O images currently resident in this process."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc._dyld_image_count.restype = ctypes.c_uint32
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    names = []
    for index in range(libc._dyld_image_count()):
        name = libc._dyld_get_image_name(index).decode()
        if "TesseraAppleRuntime" in name or "tessera_apple_gpu_runtime" in name:
            names.append(name)
    return names


class TestStalenessIsDecidedWithoutLoading:
    """Pure symbol-table logic — runs on every platform, including CI."""

    def test_library_exporting_every_sentinel_is_current(self, tmp_path):
        lib = _build_dylib(tmp_path, "current", dispatch._SENTINEL_SYMBOLS)
        assert dispatch._prebuilt_is_current(lib) is True

    def test_library_missing_one_sentinel_is_stale(self, tmp_path):
        """One missing sentinel is enough — that is the case that shipped: the
        `build/` dylib exported the f32 names and not the low-precision ones."""
        partial = list(dispatch._SENTINEL_SYMBOLS)[:-1]
        lib = _build_dylib(tmp_path, "partial", partial)
        assert dispatch._prebuilt_is_current(lib) is False

    def test_symbol_reader_strips_the_macho_underscore(self, tmp_path):
        """Mach-O prefixes C symbols with `_`; the sentinel names do not carry
        it, so a reader that forgot to strip would call every library stale."""
        lib = _build_dylib(tmp_path, "under", ["tessera_apple_gpu_probe_symbol"])
        defined = dispatch._defined_symbols(lib)
        assert defined is not None
        assert "tessera_apple_gpu_probe_symbol" in defined

    def test_undecidable_returns_none_rather_than_false(self, tmp_path, monkeypatch):
        """No `nm` means undecidable, not stale. Returning False there would
        reject a perfectly good prebuilt and silently rebuild from source."""
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        target = tmp_path / "anything.dylib"
        assert dispatch._defined_symbols(target) is None
        assert dispatch._prebuilt_is_current(target) is None

    @pytest.mark.hardware_apple_gpu
    def test_probing_a_stale_library_loads_no_image(self, tmp_path):
        """The regression itself: deciding staleness must not put the
        candidate's ObjC classes into the process."""
        lib = _build_dylib(tmp_path, "stale", ["tessera_not_a_sentinel"])
        before = set(_loaded_runtime_images())
        assert dispatch._prebuilt_is_current(lib) is False
        assert set(_loaded_runtime_images()) == before


#: Loads the runtime in a FRESH interpreter and reports what became resident.
#: A fresh process is the only place "exactly one image" is a real invariant:
#: within a pytest session many other tests legitimately build and dlopen their
#: own runtime copies into tmpdirs (18 of them in a full sweep), so asserting a
#: global count here would be asserting something untrue about the suite rather
#: than about the loader.
_PROBE = r"""
import ctypes, ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
import tessera._apple_gpu_dispatch as d
d.apple_gpu_runtime()
for i in range(libc._dyld_image_count()):
    n = libc._dyld_get_image_name(i).decode()
    if "TesseraAppleRuntime" in n or "tessera_apple_gpu_runtime" in n:
        print("IMAGE", n)
"""


class TestFreshProcessLoadsOneImage:
    """The user-visible symptom: a plain `python -c` run printing
    `Class ... is implemented in both ...` before doing any work."""

    @pytest.mark.hardware_apple_gpu
    def test_no_duplicate_class_warning_and_one_image(self, tmp_path):
        repo_root = Path(dispatch.__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "python")
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        images = [
            line.split(" ", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("IMAGE ")
        ]
        assert "is implemented in both" not in result.stderr, (
            "duplicate ObjC class registration: two images of the Apple GPU "
            f"runtime were loaded.\n{result.stderr[-1500:]}"
        )
        assert len(images) == 1, (
            "a fresh process must hold exactly one Apple GPU runtime image; "
            "more than one means the ObjC classes are registered twice and "
            "every file-static (including the thread_local last-error channel "
            f"`_apple_gpu_run_checked` reads) is duplicated: {images}"
        )

    @pytest.mark.hardware_apple_gpu
    def test_stale_prebuilt_does_not_add_a_second_image(self, tmp_path):
        """The exact configuration that shipped: a stale prebuilt pointed at by
        `$TESSERA_APPLE_GPU_RUNTIME_LIB`. Before the fix this loaded the stale
        library to probe it, then compiled and loaded a second one."""
        stale = _build_dylib(tmp_path, "libTesseraAppleRuntime", ["not_a_sentinel"])
        repo_root = Path(dispatch.__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "python")
        env["TESSERA_APPLE_GPU_RUNTIME_LIB"] = str(stale)
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        images = [
            line.split(" ", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("IMAGE ")
        ]
        assert "is implemented in both" not in result.stderr, result.stderr[-1500:]
        assert str(stale) not in images, "the stale candidate was loaded anyway"
        assert len(images) == 1, images


class TestRejectionIsReported:
    def test_skips_are_queryable(self):
        """A from-source rebuild caused by a stale prebuilt must be
        explainable; a silent rebuild reads as 'no prebuilt existed'."""
        dispatch.apple_gpu_runtime()
        skips = dispatch.apple_gpu_prebuilt_skips()
        assert isinstance(skips, tuple)
        for entry in skips:
            assert isinstance(entry, str) and entry

    def test_stale_skip_names_the_remedy(self, tmp_path, monkeypatch):
        """The message must say what to do, not just that something is wrong."""
        fake = _build_dylib(tmp_path, "libTesseraAppleRuntime", ["nope"])
        monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(fake))
        dispatch._prebuilt_candidate()
        skips = dispatch._prebuilt_skips
        assert any(str(fake) in entry for entry in skips)
        assert any("ninja -C build TesseraAppleRuntimeShared" in e for e in skips)
