"""Checked static program retention, mutation admission and bounded lifecycle."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

from tessera.compiler import prepared_rocm_nvfp4_program as prepared
from tessera.compiler import rocm_nvfp4_program as source
from tessera.compiler.native_artifact import ArtifactContractError


@pytest.fixture
def resolver(monkeypatch):
    monkeypatch.setattr(prepared,"_CACHE",prepared.OrderedDict())
    calls=[]
    def parse(manifest):
        calls.append(deepcopy(manifest))
        if (type(manifest) is not dict or type(manifest["argument_names"]) is not list
                or manifest["argument_names"]!=["x"] or type(manifest["role_indices"]) is not list
                or any(type(i) is not int for i in manifest["role_indices"])
                or manifest["role_indices"]!=[0]):
            raise ValueError("malformed contract")
        return SimpleNamespace(graph_ir=manifest["graph_ir"],argument_names=("x",),
                               roles=manifest["role_indices"])
    monkeypatch.setattr(source,"program_from_manifest",parse)
    def artifact(graph="g"):
        return SimpleNamespace(graph_ir=graph,metadata={"native_program":
            {"graph_ir":graph,"argument_names":["x"],"role_indices":[0]}, "arg_names":["x"]})
    return calls,artifact


def test_retention_consumes_detached_snapshot_and_identical_replay(resolver):
    calls,make=resolver
    artifact=make()
    first=prepared.resolve_program(artifact)
    assert prepared.resolve_program(deepcopy(artifact)) is first
    assert len(calls)==1
    artifact.metadata["native_program"]["role_indices"].append(1)
    assert first.roles==[0]
    with pytest.raises(ValueError):
        prepared.resolve_program(artifact)
    assert len(calls)==2


@pytest.mark.parametrize("mutation",["bool","tuple","argument_tuple","parent_graph","parent_names"])
def test_invalid_mutation_cannot_alias_warm_contract(resolver,mutation):
    calls,make=resolver
    artifact=make()
    prepared.resolve_program(artifact)
    manifest=artifact.metadata["native_program"]
    if mutation=="bool":manifest["role_indices"]=[False]
    elif mutation=="tuple":manifest["role_indices"]=(0,)
    elif mutation=="argument_tuple":manifest["argument_names"]=("x",)
    elif mutation=="parent_graph":artifact.graph_ir="other"
    else:artifact.metadata["arg_names"]=["renamed"]
    with pytest.raises((ValueError,ArtifactContractError)):
        prepared.resolve_program(artifact)
    assert len(calls)==2


def test_lru_retention_is_bounded(resolver,monkeypatch):
    calls,make=resolver
    monkeypatch.setattr(prepared,"_LIMIT",2)
    a=prepared.resolve_program(make("a"))
    prepared.resolve_program(make("b"))
    assert prepared.resolve_program(make("a")) is a
    prepared.resolve_program(make("c"))
    assert len(prepared._CACHE)==2
    prepared.resolve_program(make("b"))
    assert [v["graph_ir"] for v in calls]==["a","b","c","b"]


def test_signed_zero_metadata_identity_is_exact():
    assert prepared._freeze({"scale":0.})!=prepared._freeze({"scale":-0.})
    assert prepared._thaw(prepared._freeze((False,[1],{"x":.3})))==(False,[1],{"x":.3})


def test_process_guard_precedes_lock(resolver,monkeypatch):
    _,make=resolver
    class ForbiddenLock:
        def __enter__(self):
            pytest.fail("fork entered inherited lock")
        def __exit__(self,*args):
            pass
    monkeypatch.setattr(prepared,"_LOCK",ForbiddenLock())
    monkeypatch.setattr(prepared.os,"getpid",lambda:prepared._OWNER_PID+1)
    with pytest.raises(RuntimeError,match="cross fork"):
        prepared.resolve_program(make())
