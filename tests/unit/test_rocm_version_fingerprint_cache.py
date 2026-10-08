"""Version query reuse never survives executable or environment identity changes."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from tessera.compiler import rocm_native as native

@pytest.fixture
def query(monkeypatch,tmp_path):
    native._VERSION_FINGERPRINTS.clear()
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=Path(argv[0]).read_text(),stderr="",returncode=0)
    monkeypatch.setattr(native.subprocess,"run",run)
    tool=tmp_path/"compiler";tool.write_text("v1")
    yield tool,calls
    native._VERSION_FINGERPRINTS.clear()

def test_identical_tool_reuses_version(query):
    tool,calls=query
    first=native._version_fingerprint(tool)
    assert native._version_fingerprint(tool)==first
    assert len(calls)==1

@pytest.mark.parametrize("change",["replace","rewrite","symlink","environment"])
def test_identity_change_requeries(query,monkeypatch,change):
    tool,calls=query
    first=native._version_fingerprint(tool)
    if change=="replace":
        new=tool.with_name("replacement");new.write_text("v2");new.replace(tool)
    elif change=="rewrite":tool.write_text("v2")
    elif change=="symlink":
        target=tool.with_name("target");target.write_text("v2");tool.unlink();tool.symlink_to(target)
    else:monkeypatch.setenv("LD_LIBRARY_PATH","/new/compiler/libraries")
    second=native._version_fingerprint(tool)
    assert len(calls)==2
    assert (second==first)==(change=="environment")

def test_failed_query_is_not_cached(query,monkeypatch):
    tool,calls=query
    def fail(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout="",stderr="unavailable",returncode=1)
    monkeypatch.setattr(native.subprocess,"run",fail)
    native._version_fingerprint(tool);native._version_fingerprint(tool)
    assert len(calls)==2

def test_cache_is_bounded(query,tmp_path):
    _,calls=query
    for index in range(native._VERSION_FINGERPRINT_LIMIT+9):
        tool=tmp_path/str(index);tool.write_text(str(index))
        native._version_fingerprint(tool)
    assert len(native._VERSION_FINGERPRINTS)==native._VERSION_FINGERPRINT_LIMIT
