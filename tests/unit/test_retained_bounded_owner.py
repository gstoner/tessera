"""Host-side contract of the retained bounded quantized-MLP owner.

GFX1201-PUBLIC-QUANTIZED-CALL-OVERHEAD-20261028. The device proof lives in
tests/device/rocm/test_public_quantized_mlp_row_bound_device.py; these tests
drive the reuse/refresh/failure policy through a fake prepared owner.
"""
import os
import weakref

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import graph_ir
from tessera.compiler import rocm_quantized_product_frontend as frontend
from tessera.compiler.rocm_quantized_product_frontend import RetainedBoundedOwner


class FakeOwner:
    def __init__(self, log, values):
        self.log = log
        self.closed = False
        self.fail_update_activations = None
        self.fail_update = None
        self.fail_invoke = None
        self.m = np.asarray(values[0]).shape[0]
        log.append("prepare")

    def update_activations(self, x):
        if self.fail_update_activations:
            raise self.fail_update_activations
        self.m = np.asarray(x).shape[0]
        self.log.append("activations")

    def update(self, values):
        if self.fail_update:
            raise self.fail_update
        self.m = np.asarray(values[0]).shape[0]
        self.log.append("update")

    def invoke(self):
        if self.fail_invoke:
            raise self.fail_invoke
        return 1, None

    def read(self, generation):
        return (np.full((self.m, 2), generation, np.float32),)

    def close(self):
        assert not self.closed
        self.closed = True
        self.log.append("close")


class FakeNative:
    def __init__(self, log):
        self.log = log
        self.owners = []

    def prepare(self, values, *, runtime_library):
        owner = FakeOwner(self.log, values)
        self.owners.append(owner)
        return owner


class FakeProgram:
    bounded = True

    def __init__(self):
        self.log = []
        self.native = FakeNative(self.log)

    def validate(self):
        pass

    def receipts(self):
        return ({"ok": True, "execution_kind": "native_gpu"},) * 7

    def _bind(self, args):
        return list(args), type("Lib", (), {"_name": "fake.so"})()


def roots(m, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((m, 4)).astype(np.float32)] + [
        rng.integers(0, 256, size=(3, 2), dtype=np.uint8) for _ in range(6)]


def test_same_weights_replace_only_the_activation():
    program = FakeProgram()
    owner = RetainedBoundedOwner(program)
    weights = roots(3)[1:]
    for m in (3, 1, 5):
        out, receipts = owner.execute([roots(m, seed=m)[0], *weights])
        assert out.shape == (m, 2) and len(receipts) == 7
    # Equal bytes in fresh arrays are the same weights.
    owner.execute([roots(2)[0], *[w.copy() for w in weights]])
    assert program.log == ["prepare", "activations", "activations", "activations"]


def test_changed_or_mutated_weights_take_the_full_update():
    program = FakeProgram()
    owner = RetainedBoundedOwner(program)
    values = roots(3)
    owner.execute(values)
    values[2][0, 0] ^= 1  # caller mutates a weight root in place
    owner.execute(values)
    owner.execute(values)
    reshaped = roots(3)
    reshaped[1] = reshaped[1].reshape(2, 3)
    owner.execute(reshaped)
    retyped = roots(3)
    retyped[1] = retyped[1].view(np.int8)
    owner.execute(retyped)
    assert program.log == ["prepare", "update", "activations", "update", "update"]


@pytest.mark.parametrize("stage", ["fail_update_activations", "fail_update"])
def test_admission_refusal_keeps_the_owner_and_its_snapshot(stage):
    program = FakeProgram()
    owner = RetainedBoundedOwner(program)
    values = roots(3)
    owner.execute(values)
    setattr(program.native.owners[0], stage, ValueError("active M exceeds its capacity"))
    refused = values if stage == "fail_update_activations" else roots(3, seed=9)
    with pytest.raises(ValueError, match="capacity"):
        owner.execute(refused)
    setattr(program.native.owners[0], stage, None)
    # The held weights are still resident: the original set needs no upload.
    owner.execute(values)
    assert program.log == ["prepare", "activations"]


@pytest.mark.parametrize("stage", ["fail_update_activations", "fail_update", "fail_invoke"])
def test_native_failure_drops_the_owner(stage):
    error = RuntimeError("native scaled HIP program status 7")
    program = FakeProgram()
    owner = RetainedBoundedOwner(program)
    values = roots(3)
    owner.execute(values)
    setattr(program.native.owners[0], stage, error)
    changed = roots(3, seed=9) if stage == "fail_update" else values
    with pytest.raises(RuntimeError, match="status 7"):
        owner.execute(changed)
    assert program.native.owners[0].closed
    owner.execute(values)
    assert len(program.native.owners) == 2 and program.log[-1] == "prepare"


def test_failed_prepare_retains_nothing():
    program = FakeProgram()

    def refuse(values, *, runtime_library):
        raise ValueError("storage/shape differs")
    program.native.prepare = refuse
    owner = RetainedBoundedOwner(program)
    with pytest.raises(ValueError, match="storage/shape"):
        owner.execute(roots(3))
    assert owner._owner is None and owner._weights is None


def test_owner_never_crosses_fork(monkeypatch):
    program = FakeProgram()
    owner = RetainedBoundedOwner(program)
    owner.execute(roots(3))
    child = os.getpid() + 1
    monkeypatch.setattr(frontend.os, "getpid", lambda: child)
    with pytest.raises(RuntimeError, match="cannot cross fork"):
        owner.execute(roots(3))
    owner.close()  # a child never closes its parent's handle
    assert not program.native.owners[0].closed
    monkeypatch.undo()
    owner.close()
    owner.close()
    assert program.log.count("close") == 1


def test_only_a_bounded_program_retains_an_owner():
    program = FakeProgram()
    program.bounded = False
    with pytest.raises(ValueError, match="only a bounded"):
        RetainedBoundedOwner(program)


def test_jit_function_supports_the_owner_finalizer():
    fn = ts.jit(lambda x: x)
    weakref.finalize(fn, lambda: None).detach()


def test_loc_path_is_memoized_per_working_directory(tmp_path, monkeypatch):
    graph_ir._resolved_loc_path.cache_clear()
    here = graph_ir.__file__
    assert graph_ir._loc_path(here) == "python/tessera/compiler/graph_ir.py"
    assert graph_ir._loc_path(here) == "python/tessera/compiler/graph_ir.py"
    assert graph_ir._resolved_loc_path.cache_info().hits >= 1
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    first = graph_ir._loc_path("model.py")
    monkeypatch.chdir(tmp_path / "b")
    second = graph_ir._loc_path("model.py")
    assert first == second == "model.py"  # outside the repo keeps the name
    repo = graph_ir._REPO_ROOT
    monkeypatch.chdir(repo / "python")
    assert graph_ir._loc_path("tessera/compiler/graph_ir.py") == "python/tessera/compiler/graph_ir.py"
    monkeypatch.chdir(repo)
    assert graph_ir._loc_path("tessera/compiler/graph_ir.py") == "tessera/compiler/graph_ir.py"
