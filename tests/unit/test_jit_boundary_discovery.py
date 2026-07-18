from __future__ import annotations

import pytest

from tessera import _jit_boundary as jb


def test_darwin_prefers_configured_build_apple_artifacts(monkeypatch):
    monkeypatch.delenv("TESSERA_JIT_LIB", raising=False)
    monkeypatch.delenv("TESSERA_OPT", raising=False)
    monkeypatch.delenv("TESSERA_OPT_BIN", raising=False)
    monkeypatch.setattr(jb.sys, "platform", "darwin")
    monkeypatch.setattr(jb, "_repo_root", lambda: "/repo")

    def fake_glob(pattern: str):
        if "/build-apple/" in pattern:
            return ["/repo/build-apple/tools/tessera-jit/libtessera_jit.dylib"]
        if "/build/" in pattern:
            return ["/repo/build/tools/tessera-jit/libtessera_jit.dylib"]
        return []

    monkeypatch.setattr(jb.glob, "glob", fake_glob)
    assert jb._find_dylib() == \
        "/repo/build-apple/tools/tessera-jit/libtessera_jit.dylib"

    monkeypatch.setattr(
        jb.os.path, "exists", lambda path: str(path).startswith("/repo/"))
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(jb, "_TESSERA_OPT_PATH", "unset")
    assert jb._find_tessera_opt() == \
        "/repo/build-apple/tools/tessera-opt/tessera-opt"


def test_jit_load_error_is_fail_closed_and_actionable(monkeypatch):
    monkeypatch.setattr(jb, "_LIB", None)
    monkeypatch.setattr(jb, "_find_dylib", lambda: "/stale/libtessera_jit.dylib")

    def fail(_path: str):
        raise OSError("missing libLLVM")

    monkeypatch.setattr(jb.ctypes, "CDLL", fail)
    with pytest.raises(jb.TesseraJitError, match="set TESSERA_JIT_LIB"):
        jb._load()
    assert jb.is_available() is False
