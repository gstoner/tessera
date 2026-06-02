"""Lane (c) — MSL-source → serialized ``.metallib`` dynamic-library AOT.

Proves the parallel AOT lane (distinct from the MPSGraph ``.mtlpackage``
path): compile an MSL dynamic library, serialize it to disk, reload it.
Gated on Apple GPU runtime availability.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tessera import apple_dylib as ad


def _require():
    if not ad.dylib_available():
        pytest.skip("Apple GPU runtime unavailable on this host")


def test_dylib_symbols_resolve():
    assert callable(ad.serialize_msl_dylib)
    assert callable(ad.load_dylib)
    assert "[[visible]]" in ad.SAMPLE_VISIBLE_MSL


def test_serialize_then_reload_roundtrip():
    _require()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "kern.metallib")
        assert ad.serialize_msl_dylib(ad.SAMPLE_VISIBLE_MSL, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        # Reload the *committed-to-disk* artifact — the AOT win.
        assert ad.load_dylib(out)


def test_serialize_rejects_bad_source():
    """Non-compiling MSL fails cleanly (False, no exception)."""
    _require()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "bad.metallib")
        assert ad.serialize_msl_dylib("this is not valid metal", out) is False
        assert not os.path.exists(out)


def test_load_missing_file_returns_false():
    _require()
    assert ad.load_dylib("/nonexistent/path/x.metallib") is False


def test_custom_install_name_roundtrips():
    _require()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "named.metallib")
        ok = ad.serialize_msl_dylib(
            ad.SAMPLE_VISIBLE_MSL, out,
            install_name="@loader_path/custom_tessera.metallib")
        assert ok
        assert ad.load_dylib(out)
