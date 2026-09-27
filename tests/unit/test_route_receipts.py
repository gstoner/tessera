"""EVIDENCE-PACKET-1: per-call route receipts for GA/EBM composition.

Host-free: native lanes are simulated by patching a ``_try_*`` helper with a
decorated fake, so every assertion here is about attribution, not about
which silicon this host has. Sync ``EVIDENCE-PACKET-1-2026-09-27``.
"""

from __future__ import annotations

import ast
import copy
import importlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from tessera import _route_receipts as rr

ROOT = Path(__file__).resolve().parents[2]

#: The GA/EBM modules whose public primitives choose a native lane per call.
_MODULES = {
    "tessera.ga.ops": "python/tessera/ga/ops.py",
    "tessera.ga.calculus": "python/tessera/ga/calculus.py",
    "tessera.ebm.energy": "python/tessera/ebm/energy.py",
    "tessera.ebm.partition": "python/tessera/ebm/partition.py",
    "tessera.ebm.geo_sampling": "python/tessera/ebm/geo_sampling.py",
}
#: Names whose use means native runtime work.
_NATIVE_NAMES = {"runtime", "rt", "bind_symbol", "dispatch_via_manifest",
                 "_apple_gpu_dispatch", "jit_bridge", "_bridge"}


# --------------------------------------------------------------------------
# Drift gates: every native lane is a decorated helper, every public caller
# of one leaves a receipt, and nothing reaches native work any other way.
# --------------------------------------------------------------------------

def _module_functions(rel: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse((ROOT / rel).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _names(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                out.update(node.module.split("."))
            for alias in node.names:
                out.add(alias.asname or alias.name)
                out.update(alias.name.split("."))
    return out


def _reaching(funcs: dict[str, ast.FunctionDef]) -> set[str]:
    """Functions that reach a ``_try_*`` helper directly or through a private
    helper of the same module."""
    refs = {name: _names(fn) for name, fn in funcs.items()}
    reaching = {n for n in funcs if n.startswith("_try_")}
    changed = True
    while changed:
        changed = False
        for name, r in refs.items():
            if name in reaching:
                continue
            private = {p for p in reaching if p.startswith("_") and not p.startswith("_try_")}
            if any(x.startswith("_try_") for x in r) or r & private:
                reaching.add(name)
                changed = True
    return reaching


@pytest.mark.parametrize("module,rel", sorted(_MODULES.items()))
def test_every_native_lane_helper_is_decorated(module, rel) -> None:
    mod = importlib.import_module(module)
    for name in _module_functions(rel):
        if name.startswith("_try_"):
            target = getattr(getattr(mod, name), "__tessera_native_target__", None)
            assert target == rr.native_target_for(name), f"{module}.{name} is not @native_attempt"


@pytest.mark.parametrize("module,rel", sorted(_MODULES.items()))
def test_every_public_caller_of_a_native_lane_leaves_a_receipt(module, rel) -> None:
    mod = importlib.import_module(module)
    funcs = _module_functions(rel)
    missing = [
        name for name in sorted(_reaching(funcs))
        if not name.startswith("_")
        and getattr(getattr(mod, name), "__tessera_public_route__", None) is None
    ]
    assert not missing, f"{module}: public callers of _try_* without @public_route: {missing}"


@pytest.mark.parametrize("module,rel", sorted(_MODULES.items()))
def test_native_work_is_reached_only_through_a_try_helper(module, rel) -> None:
    offenders = []
    for name, fn in _module_functions(rel).items():
        if name.startswith("_try_"):
            continue
        hits = _names(fn) & _NATIVE_NAMES
        if hits:
            offenders.append(f"{name}: {sorted(hits)}")
    assert not offenders, (
        f"{module}: native runtime reached outside a _try_* helper, so no receipt "
        f"would name it: {offenders}")


def test_the_scanned_modules_are_every_ga_ebm_module_with_a_native_lane() -> None:
    """A new GA/EBM module with a ``_try_*`` lane must join the gate."""
    scanned = {ROOT / rel for rel in _MODULES.values()}
    for package in ("ga", "ebm"):
        for path in sorted((ROOT / "python" / "tessera" / package).glob("*.py")):
            funcs = _module_functions(path.relative_to(ROOT).as_posix())
            if any(n.startswith("_try_") for n in funcs):
                assert path in scanned, f"{path} has a _try_* lane but is not gated"


def test_every_try_prefix_names_a_declared_target() -> None:
    assert rr.native_target_for("_try_apple_gpu_inner_step") == "apple_gpu_runtime"
    assert rr.native_target_for("_try_x86_energy_quadratic_f32") == "x86_avx512"
    assert rr.native_target_for("_try_rocm_energy_quadratic_f32") == "rocm"
    assert rr.native_target_for("_try_cuda_gpu_sphere_langevin_step_f32") == "cuda"
    with pytest.raises(ValueError, match="no declared native target"):
        rr.native_target_for("_try_metal_something")


# --------------------------------------------------------------------------
# Receipt semantics.
# --------------------------------------------------------------------------

def _lane(name: str, returns):
    def fake(*args, **kwargs):
        return returns
    fake.__name__ = name
    return rr.native_attempt(fake)


def test_outside_a_capture_nothing_is_recorded() -> None:
    lane = _lane("_try_x86_fake", 1.0)

    @rr.public_route("t.op")
    def op():
        return lane()

    assert op() == 1.0
    with rr.capture_route_receipts() as log:
        pass
    assert log.receipts == [] and log.refusal().startswith("ROUTE_RECEIPT_EMPTY")
    assert log.route() == rr.ROUTE_UNATTRIBUTED


def test_routes_reference_native_nested_and_mixed() -> None:
    native = _lane("_try_x86_fake", 1.0)
    declined = _lane("_try_rocm_fake", None)

    @rr.public_route("t.native")
    def native_op():
        return native()

    @rr.public_route("t.reference")
    def reference_op():
        return declined() or 0.0

    @rr.public_route("t.outer")
    def outer():
        return native_op() + reference_op()

    with rr.capture_route_receipts() as log:
        native_op()
        reference_op()
        outer()
    top = {r.op: r.route for r in log.top_level()}
    assert top == {"t.native": "x86_avx512", "t.reference": "python_reference", "t.outer": "mixed"}
    assert log.route() == "mixed"
    summary = log.summary()
    assert summary["attribution"] == "complete" and summary["calls"] == 3
    assert summary["native_dispatches"] == {"x86_avx512:_try_x86_fake": 2}
    assert rr.device_label(log.route()) == "mixed+cpu"


def test_an_orphan_native_dispatch_makes_the_span_unattributed() -> None:
    lane = _lane("_try_apple_gpu_fake", 2.0)

    @rr.public_route("t.op")
    def op():
        return 0.0

    with rr.capture_route_receipts() as log:
        op()
        lane()  # native work no public receipt names
    assert log.refusal().startswith("ROUTE_RECEIPT_ORPHAN_DISPATCH")
    assert log.route() == rr.ROUTE_UNATTRIBUTED
    assert rr.device_label(log.route()) == "unattributed"
    assert log.summary()["attribution"] == "incomplete"


def test_a_raising_call_leaves_no_receipt_and_restores_the_frame_stack() -> None:
    @rr.public_route("t.boom")
    def boom():
        raise RuntimeError("x")

    @rr.public_route("t.ok")
    def ok():
        return 1

    with rr.capture_route_receipts() as log:
        with pytest.raises(RuntimeError):
            boom()
        ok()
    assert [(r.op, r.depth) for r in log.receipts] == [("t.ok", 0)]


def test_captures_are_per_thread() -> None:
    lane = _lane("_try_x86_fake", 1.0)

    @rr.public_route("t.op")
    def op():
        return lane()

    with rr.capture_route_receipts() as log:
        worker = threading.Thread(target=op)
        worker.start()
        worker.join()
    assert log.receipts == [] and log.orphan_dispatches == []


# --------------------------------------------------------------------------
# The real primitives and the composition benchmarks.
# --------------------------------------------------------------------------

def test_energy_quadratic_receipt_names_the_lane_that_ran(monkeypatch) -> None:
    energy = importlib.import_module("tessera.ebm.energy")

    x = np.ones((4, 3), np.float32)
    y = np.zeros((4, 3), np.float32)
    for name in ("_try_x86_energy_quadratic_f32", "_try_rocm_energy_quadratic_f32",
                 "_try_apple_gpu_energy_quadratic_f32"):
        monkeypatch.setattr(energy, name, _lane(name, None))
    with rr.capture_route_receipts() as log:
        ref = energy.energy_quadratic(x, y)
    assert [r.route for r in log.top_level()] == ["python_reference"]

    # Simulate the x86 AVX-512 lane answering: the receipt must say so. Before
    # receipts, the jit_bridge trace saw only the Apple manifest lane, so this
    # call would have read as "no native dispatch".
    monkeypatch.setattr(energy, "_try_x86_energy_quadratic_f32",
                        _lane("_try_x86_energy_quadratic_f32", ref))
    with rr.capture_route_receipts() as log:
        energy.energy_quadratic(x, y)
    (receipt,) = log.top_level()
    assert receipt.op == "tessera.ebm.energy_quadratic"
    assert receipt.route == "x86_avx512"
    assert receipt.native == (("x86_avx512", "_try_x86_energy_quadratic_f32"),)


def test_energy_core_row_carries_its_route(monkeypatch) -> None:
    import benchmarks.energy_core.core as core
    energy = importlib.import_module("tessera.ebm.energy")
    partition = importlib.import_module("tessera.ebm.partition")

    for mod in (energy, partition):
        for name in dir(mod):
            if name.startswith("_try_"):
                monkeypatch.setattr(mod, name, _lane(name, None))
    cfg = core.EnergyCoreConfig(B=4, D=3, n_steps=2)
    row = core.EnergyCoreBenchmark(warmup=0, reps=1).run_one(cfg).to_dict()
    assert row["route"] == "python_reference" and row["device"] == "cpu"
    receipts = row["route_receipts"]
    assert receipts["attribution"] == "complete"
    assert set(receipts["ops"]) >= {"tessera.ebm.langevin_step",
                                    "tessera.ebm.partition_exact_from_energies"}
    assert row["promotion_eligible"] is False
    assert rr.validate_receipt_summary(receipts) == "python_reference"


# --------------------------------------------------------------------------
# Committed receipts (benchmarks/baselines/ga_ebm_route_receipts_20260927):
# one record per fleet host, recorded at one clean commit.
# --------------------------------------------------------------------------

_RECEIPTS = ROOT / "benchmarks" / "baselines" / "ga_ebm_route_receipts_20260927"


def _records() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(_RECEIPTS.glob("*.json"))}


def test_every_committed_receipt_row_rederives() -> None:
    records = _records()
    assert set(records) == {"mac_m1max", "princess_luna", "tajasarus", "super_bear"}
    for host, record in records.items():
        assert record["host"]["worktree_dirty"] is False, host
        assert record["promotion_eligible"] is False
        for suite, rows in record["suites"].items():
            for row in rows:
                route = rr.validate_receipt_summary(row["route_receipts"])
                assert row["route"] == route and row["device"] == rr.device_label(route)
                assert row["route_receipts"]["attribution"] == "complete", (host, suite)
                assert row["promotion_eligible"] is False


def _ops(record: dict, suite: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in record["suites"][suite]:
        for op, by_route in row["route_receipts"]["ops"].items():
            out.setdefault(op, set()).update(by_route)
    return out


def test_committed_receipts_name_the_lanes_each_host_ran() -> None:
    """What the receipts established, per host (sync EVIDENCE-PACKET-1-2026-09-27).

    The Zen 5 hosts ran the EBM energy and partition primitives on the x86
    AVX-512 lane; the Zen 2 CUDA host has no such lane and ran the reference;
    only the Mac reached the Apple GPU runtime. No composition reached a ROCm
    or CUDA GPU lane on any host.
    """
    records = _records()
    for host in ("princess_luna", "tajasarus"):
        ops = _ops(records[host], "energy_core")
        assert ops["tessera.ebm.energy_quadratic"] == {"x86_avx512"}
        assert ops["tessera.ebm.partition_exact_from_energies"] == {"x86_avx512"}
        assert ops["tessera.ebm.langevin_step"] == {"python_reference"}
    bear = _ops(records["super_bear"], "energy_core")
    assert set().union(*bear.values()) == {"python_reference"}
    mac = _ops(records["mac_m1max"], "energy_core")
    assert mac["tessera.ebm.langevin_step"] == {"apple_gpu_runtime"}
    for record in records.values():
        for suite in record["suites"]:
            routes = set().union(*_ops(record, suite).values())
            assert not routes & {"rocm", "cuda"}


@pytest.mark.parametrize("edit,match", [
    (lambda s: s.update(route="x86_avx512"), "not the derived"),
    (lambda s: s["routes"].update(python_reference=999), "per-op counts"),
    (lambda s: s.update(orphan_dispatches=1), "attribution differs"),
    (lambda s: s["routes"].update(npu=1), "undeclared route"),
    (lambda s: s.update(schema="other"), "not a tessera.route_receipts.v1"),
])
def test_a_doctored_receipt_summary_refuses(edit, match) -> None:
    row = _records()["princess_luna"]["suites"]["energy_core"][0]
    summary = copy.deepcopy(row["route_receipts"])
    rr.validate_receipt_summary(summary)
    edit(summary)
    with pytest.raises(ValueError, match=match):
        rr.validate_receipt_summary(summary)
