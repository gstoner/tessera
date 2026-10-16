"""Host-only cache/ownership regressions; these do not claim CUDA execution."""
from types import SimpleNamespace
import os

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import native_attention_program, native_attention_vjp_runtime as adapter
from tessera.compiler import scheduled_matmul


@pytest.fixture
def harness(monkeypatch, tmp_path):
    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"selected compiler identity")
    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", lambda: compiler)
    monkeypatch.setattr(adapter, "_cache", adapter.OrderedDict())
    monkeypatch.setattr(adapter, "_pid", os.getpid())
    compiled, launches = [], []

    def compile_program(graph, active, **kwargs):
        compiled.append((graph, active, kwargs))
        seeded = "seeded" in graph
        def member(role):
            return SimpleNamespace(
                descriptor=SimpleNamespace(provenance={
                    "schedule_digest": role, "lse_cotangent": seeded}),
                tile_ir=role + " tile", target_ir=role + " native target")
        return SimpleNamespace(
            input_indices=(0, 1, 2),
            to_json=lambda: '{"compiled":"' + graph + '"}',
            program_digest="compiled-" + graph + str(active),
            pair=SimpleNamespace(forward=member("forward"), backward=member("backward"),
                                 contract_digest="lineage"))

    def launch(artifact, values):
        # Dispatch is still entered each call and reads the actual current roots.
        launches.append((artifact, values, artifact.artifact_hash))
        return {"ok": True, "execution_kind": "native_gpu",
                "output": tuple(x.copy() for x in values[:3])}

    monkeypatch.setattr(native_attention_program, "compile_attention_vjp_program", compile_program)
    monkeypatch.setattr(rt, "launch", launch)
    declaration = SimpleNamespace(family="attention", schedule_consumer="schedule",
                                  tile_consumer="tile", target_consumers={"nvidia_sm120": "native"})
    def invoke(*, graph="graph", inputs=None, wrt=("q",), seeded=False):
        roots = inputs if inputs is not None else tuple(np.full((1, 1, 2), i, np.float32) for i in range(3))
        return adapter.execute_family(
            source=SimpleNamespace(op_name="tessera.flash_attn",
                                   kwargs={"lse_checkpoint": "saved"} if seeded else {}),
            target="nvidia_sm120", ordered_inputs=roots,
            arg_names=("q", "k", "v")[:len(roots)],
            source_arg_names=("q", "k", "v")[:len(roots)],
            out_cotangents=(np.ones((1, 1, 2), np.float32),) * (1 + int(seeded)),
            wrt_names=wrt, declaration=declaration, source_graph_ir=graph)
    return SimpleNamespace(invoke=invoke, compiled=compiled, launches=launches, compiler=compiler)


def test_warm_dispatch_reuses_identity_without_serializing_and_reads_new_roots(harness, monkeypatch):
    cold = harness.invoke()
    def forbidden(*args, **kwargs):
        raise AssertionError("warm reverse must not serialize compiled artifact")
    monkeypatch.setattr(rt.RuntimeArtifact, "to_json", forbidden)
    roots = tuple(np.full((1, 1, 2), i + 10, np.float32) for i in range(3))
    warm = harness.invoke(inputs=roots)
    assert len(harness.compiled) == 1
    assert len(harness.launches) == 2
    assert cold.runtime_artifact is not warm.runtime_artifact
    assert cold.runtime_artifact.artifact_hash == warm.runtime_artifact.artifact_hash
    for old, new, root in zip(cold.gradients, warm.gradients, roots, strict=True):
        np.testing.assert_array_equal(new, root)
        assert not np.array_equal(old, new)


def test_invocation_metadata_and_proof_do_not_poison_private_prototype(harness):
    first = harness.invoke()
    canonical = first.runtime_artifact.to_json()
    first.runtime_artifact.metadata["arg_names"].append("poison")
    first.runtime_artifact.metadata["program_json"] = "poison"
    first.execution["state_lineage_digest"] = "poison"
    second = harness.invoke()
    assert second.runtime_artifact.to_json() == canonical
    assert second.execution["state_lineage_digest"] == "lineage"
    assert len(harness.compiled) == 1


@pytest.mark.parametrize("change", ["graph", "wrt", "compiler"])
def test_compiled_identity_changes_invalidate_cache(harness, change):
    harness.invoke()
    kwargs = {}
    if change == "graph":
        kwargs["graph"] = "other graph"
    elif change == "wrt":
        kwargs["wrt"] = ("v", "q")
    else:
        replacement = harness.compiler.with_suffix(".replacement")
        replacement.write_bytes(b"replacement compiler identity")
        replacement.replace(harness.compiler)
    harness.invoke(**kwargs)
    assert len(harness.compiled) == 2
    assert len(harness.launches) == 2


@pytest.mark.parametrize("mismatch", ["root_count", "seed"])
def test_frontend_role_mismatch_rejects_before_launch_and_does_not_poison(harness, mismatch):
    harness.invoke()
    kwargs = {"inputs": tuple(np.zeros((1, 1, 2), np.float32) for _ in range(2))}
    if mismatch == "seed":
        kwargs = {"seeded": True}
    with pytest.raises(ValueError, match="frontend roles disagree"):
        harness.invoke(**kwargs)
    assert len(harness.launches) == 1
    harness.invoke()
    assert len(harness.launches) == 2
    assert len(harness.compiled) == 1


def test_seeded_native_roles_are_preserved(harness):
    result = harness.invoke(graph="seeded graph", seeded=True)
    assert result.runtime_artifact.metadata["arg_names"] == [
        "primal_0", "primal_1", "primal_2", "cotangent", "lse_cotangent"]
    assert len(harness.launches[0][1]) == 5


def test_lru_eviction_recompiles_only_evicted_entry(harness, monkeypatch):
    monkeypatch.setattr(adapter, "_LIMIT", 2)
    for graph in ("a", "b", "a", "c", "a", "b"):
        harness.invoke(graph=graph)
    assert [item[0] for item in harness.compiled] == ["a", "b", "c", "b"]
    assert len(adapter._cache) == 2


def test_fork_guard_remains_before_dispatch(harness, monkeypatch):
    harness.invoke()
    monkeypatch.setattr(adapter, "_pid", os.getpid() + 1)
    with pytest.raises(ValueError, match="cannot cross fork"):
        harness.invoke()
    assert len(harness.launches) == 1
