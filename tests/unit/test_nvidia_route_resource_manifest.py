"""Assembly of the sm_120 route-resource manifest from isolated reports.

Sync ``AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`` (``AUTOTUNE-SM120-ROUTE-RESOURCES``;
Codex P2 on PR #868). ``capture_route_resources.sh`` extends the committed
manifest with one Nsight report per route. Once those routes were committed,
the default run did every ncu capture and then failed at assembly, because
``add_isolated_routes`` refuses a route already present. Now:

* the default (add) mode fails fast -- before any capture -- naming the routes
  already present;
* ``--refresh`` replaces exactly the recaptured routes (their ``routes`` and
  ``details`` entries and the ``sources`` tagged with them), keeps every other
  route and untagged source byte-for-byte, and never leaves one route with an
  old and a new capture mixed.

Host-free: synthetic normalized reports, no ncu, no device.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_manifest_assembly", ROOT / "benchmarks/nvidia/build_test5_resource_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _report(route: str, fp: str, source: str | None = None) -> dict:
    out = {"rows": [{"kernel": f"{route}_kernel", "resource_fingerprint": fp}]}
    if source:
        out.update(source=source, source_sha256=source * 2)
    return out


def _base() -> dict:
    """A committed-style manifest: two name-mapped routes (untagged sources)."""
    return {"schema": "tessera.nvidia.route-resources.v1",
            "sources": [{"name": "legacy.ncu-rep", "sha256": "11"}],
            "routes": {"nvidia_mma_fused": ["sha256:f16"],
                       "nvidia_mma_gemm_shipped": ["sha256:gemm"]},
            "details": {"nvidia_mma_fused": [{"kernel": "k", "resource_fingerprint": "sha256:f16"}],
                        "nvidia_mma_gemm_shipped": [{"kernel": "gemm",
                                                     "resource_fingerprint": "sha256:gemm"}]}}


def test_fresh_base_adds_isolated_routes():
    mod = _mod()
    base = _base()
    out = mod.add_isolated_routes(base, {
        "nvidia_mma_fused_tf32": _report("tf32", "sha256:a", "tf32.ncu-repz"),
        "nvidia_gated": _report("gated", "sha256:b", "gated.ncu-repz")})
    assert out["routes"]["nvidia_mma_fused_tf32"] == ["sha256:a"]
    assert out["routes"]["nvidia_gated"] == ["sha256:b"]
    assert out["routes"]["nvidia_mma_fused"] == ["sha256:f16"]           # kept
    assert [s.get("route") for s in out["sources"]] == [None, "nvidia_gated",
                                                         "nvidia_mma_fused_tf32"]
    assert base == _base()                                               # not mutated


def test_default_mode_refuses_a_route_already_present():
    mod = _mod()
    added = mod.add_isolated_routes(_base(), {
        "nvidia_mma_fused_tf32": _report("tf32", "sha256:a", "tf32.ncu-repz")})
    with pytest.raises(ValueError, match="nvidia_mma_fused_tf32 already in the manifest.*--refresh"):
        mod.add_isolated_routes(added, {
            "nvidia_mma_fused_tf32": _report("tf32", "sha256:a2", "tf32b.ncu-repz")})
    assert mod.existing_routes(added, ["nvidia_mma_fused_tf32", "nvidia_gated"]) == [
        "nvidia_mma_fused_tf32"]


def test_refresh_replaces_exactly_the_recaptured_routes():
    mod = _mod()
    committed = mod.add_isolated_routes(_base(), {
        "nvidia_mma_fused_tf32": _report("tf32", "sha256:old", "tf32-old.ncu-repz"),
        "nvidia_gated": _report("gated", "sha256:g", "gated.ncu-repz")})
    out = mod.add_isolated_routes(committed, {
        "nvidia_mma_fused_tf32": _report("tf32", "sha256:new", "tf32-new.ncu-repz")},
        refresh=True)
    # The refreshed route carries only the new capture...
    assert out["routes"]["nvidia_mma_fused_tf32"] == ["sha256:new"]
    assert out["details"]["nvidia_mma_fused_tf32"] == [
        {"kernel": "tf32_kernel", "resource_fingerprint": "sha256:new"}]
    tagged = [s for s in out["sources"] if s.get("route") == "nvidia_mma_fused_tf32"]
    assert [s["name"] for s in tagged] == ["tf32-new.ncu-repz"]
    # ...and every route and source it did not capture is untouched.
    for route in ("nvidia_gated", "nvidia_mma_fused", "nvidia_mma_gemm_shipped"):
        assert out["routes"][route] == committed["routes"][route]
        assert out["details"][route] == committed["details"][route]
    assert [s for s in out["sources"] if s.get("route") != "nvidia_mma_fused_tf32"] == [
        s for s in committed["sources"] if s.get("route") != "nvidia_mma_fused_tf32"]


def test_refresh_refuses_a_route_it_would_not_replace():
    mod = _mod()
    with pytest.raises(ValueError, match="--refresh names routes not in the manifest"):
        mod.add_isolated_routes(_base(), {"nvidia_gated": _report("g", "sha256:g")},
                                refresh=True)


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/nvidia/build_test5_resource_manifest.py"), *args],
        capture_output=True, text=True)


def test_preflight_against_the_committed_manifest():
    """The committed manifest holds the 17 isolated routes, so an add-mode
    capture must be refused up front and a refresh admitted."""
    committed = ROOT / "benchmarks/baselines/nvidia_sm120_test5_route_resources.json"
    routes = ("nvidia_mma_fused_tf32", "nvidia_gated")
    refused = _cli("--base", str(committed), "--check-routes", *routes)
    assert refused.returncode == 1 and "nvidia_gated" in refused.stdout
    assert "--refresh" in refused.stdout
    assert _cli("--base", str(committed), "--refresh", "--check-routes", *routes).returncode == 0


def test_capture_script_fails_before_capturing(tmp_path):
    """Default mode against the committed manifest: nonzero exit, no ncu call,
    no output directory."""
    marker = tmp_path / "ncu-was-called"
    ncu = tmp_path / "ncu"
    ncu.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    ncu.chmod(0o755)
    out = tmp_path / "out"
    done = subprocess.run(
        ["bash", "benchmarks/nvidia/capture_route_resources.sh", str(out)],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "NCU": str(ncu), "PYTHON": sys.executable})
    assert done.returncode != 0
    assert "already in" in done.stdout
    assert not marker.exists() and not out.exists()


def test_cli_refresh_round_trip(tmp_path):
    """The CLI path the script runs at assembly, on a synthetic base."""
    mod = _mod()
    base = tmp_path / "base.json"
    base.write_text(json.dumps(mod.add_isolated_routes(_base(), {
        "nvidia_gated": _report("gated", "sha256:old", "old.ncu-repz")})))
    new = tmp_path / "gated.json"
    new.write_text(json.dumps(_report("gated", "sha256:new", "new.ncu-repz")))
    out = tmp_path / "out.json"
    add = _cli("--base", str(base), "--route", f"nvidia_gated={new}", "--output", str(out))
    assert add.returncode != 0 and "refusing to overwrite" in add.stderr
    ok = _cli("--base", str(base), "--refresh", "--route", f"nvidia_gated={new}",
              "--output", str(out))
    assert ok.returncode == 0, ok.stderr
    manifest = json.loads(out.read_text())
    assert manifest["routes"]["nvidia_gated"] == ["sha256:new"]
    assert manifest["routes"]["nvidia_mma_fused"] == ["sha256:f16"]
