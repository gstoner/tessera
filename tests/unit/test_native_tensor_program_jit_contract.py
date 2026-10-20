"""Native JIT inspection and artifact restoration retain real compiler ancestry."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from tessera.compiler.native_tensor_jit import input_signature, lowering_artifacts
from tessera.compiler.native_tensor_program import (
    compile_native_tensor_program, restore_runtime_artifact, runtime_artifact,
)
from tests.unit.test_native_tensor_program_pipeline import norm_product, traced_source

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


@pytest.fixture(scope="module")
def program():
    return compile_native_tensor_program(traced_source(norm_product, "f16"))


@pytest.mark.compiler_route
def test_artifact_has_four_adjacent_actual_native_stages(program):
    artifact = runtime_artifact(program, argument_names=("a", "b"))
    assert (artifact.graph_ir, artifact.schedule_ir, artifact.tile_ir, artifact.target_ir) == (
        program.source_ir, program.schedule_ir, program.tile_ir, program.target_ir)
    assert "schedule." in artifact.schedule_ir
    assert "tile.program_invoke " in artifact.tile_ir
    assert "tessera_nvidia.program_invoke " in artifact.target_ir
    stages = lowering_artifacts(program)
    assert [stage.level for stage in stages] == ["graph", "schedule", "tile", "target"]
    assert all(b.input_digest == a.output_digest for a, b in zip(stages, stages[1:]))
    assert artifact.metadata["restoration"] == "matching_compiler_required"


@pytest.mark.compiler_route
@pytest.mark.parametrize("field", ["schedule_ir", "tile_ir"])
def test_intermediate_fields_cannot_forge_source_ancestry(program, field):
    with pytest.raises(ValueError, match="differs"):
        replace(program, **{field: getattr(program, field) + "\n// altered\n"})


@pytest.mark.compiler_route
@pytest.mark.parametrize("field", ["source_ir", "schedule_ir", "tile_ir", "target_ir", "images"])
def test_runtime_restore_checks_semantics_beyond_parent_field_equality(program, field):
    artifact = copy.deepcopy(runtime_artifact(program))
    data = artifact.metadata["native_program"]
    if field == "images":
        data[field][0] = "eA=="
    else:
        data[field] += "\n// altered\n"
        artifact = replace(artifact, **{"graph_ir" if field == "source_ir" else field: data[field]})
    with pytest.raises((ValueError, RuntimeError)):
        restore_runtime_artifact(artifact)


@pytest.mark.compiler_route
def test_runtime_restore_never_takes_a_compiler_path_from_payload(program):
    artifact = runtime_artifact(program)
    artifact.metadata["native_program"]["compiler"] = "/untrusted/compiler"
    with pytest.raises(ValueError, match="differs"):
        restore_runtime_artifact(artifact)


@pytest.mark.compiler_route
def test_runtime_restore_requires_trusted_configured_compiler(program, monkeypatch):
    import tessera.compiler.native_tensor_program as module
    artifact = runtime_artifact(program)
    monkeypatch.setattr(module, "find_tessera_opt", lambda: None)
    with pytest.raises(RuntimeError, match="requires the matching compiler"):
        restore_runtime_artifact(artifact)


def test_signature_reuses_metadata_across_changed_values():
    first = np.zeros((3, 4), np.float16), np.zeros((4, 2), np.float16)
    second = tuple(np.ones_like(value) for value in first)
    assert input_signature(first) == input_signature(second)


def test_signature_preserves_other_storage_and_layout_routes():
    half = np.zeros((4, 8), np.float16)
    assert input_signature((half, half)) is not None
    assert input_signature((half[:, ::2], half)) is None
    assert input_signature((np.asfortranarray(half), half)) is None
    assert input_signature((half.astype(np.float32), half)) is None
    assert input_signature((half.reshape(2, 2, 8), half)) is None


def test_retained_jit_pins_owned_inputs_and_rebinds_without_compiler(monkeypatch):
    from contextlib import contextmanager
    from pathlib import Path
    from types import SimpleNamespace
    import tessera as ts
    from tessera import runtime as rt
    import tessera.compiler.native_tensor_program as packages
    import tessera.compiler.native_tensor_result as results
    import tessera.compiler.scheduled_matmul as tools

    fn = ts.jit(target="nvidia_sm120")(norm_product)
    roots = np.zeros((17, 32), np.float16), np.zeros((32, 11), np.float16)
    active = []
    output = object()
    binds = []

    @contextmanager
    def scope(values, stream=None):
        assert values == roots
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    class Owner:
        handle = 1
        def __call__(self, values):
            assert active and values == roots
            return output
        def close(self):
            self.handle = 0

    class Program:
        members = ("producer", "consumer")
        def prepare(self):
            binds.append(True)
            return Owner()

    monkeypatch.setattr(results, "input_scope", scope)
    monkeypatch.setattr(tools, "find_tessera_opt", lambda: Path("/configured/compiler"))
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch",
                        lambda: SimpleNamespace(tessera_nvidia_native_frame_prepare_v1=True))
    monkeypatch.setattr(fn, "_traced_autodiff_module", lambda *args, **kwargs: object())
    monkeypatch.setattr(packages, "compile_native_tensor_program", lambda *args, **kwargs: Program())
    assert fn._try_native_tensor_program_call(roots, {}) is output
    assert not active

    def forbidden(*args, **kwargs):
        pytest.fail("retained JIT rediscovered compiler or source")
    monkeypatch.setattr(tools, "find_tessera_opt", forbidden)
    monkeypatch.setattr(packages, "compile_native_tensor_program", forbidden)
    monkeypatch.setattr(fn, "_traced_autodiff_module", forbidden)
    assert fn._try_native_tensor_program_call(roots, {}) is output
    assert len(binds) == 1
    fn.close_native_storage()
    assert fn._try_native_tensor_program_call(roots, {}) is output
    assert len(binds) == 2
    fn.close_native_storage()


@pytest.mark.parametrize("typestr,strides", [
    ("|u1", None), ("|V1", None), ("<f4", None),
    ("<f2", (2, 4)), ("<f2", (16, 2)),
])
def test_resident_profile_declines_other_storage_and_layouts_before_strict_parser(
    typestr, strides, monkeypatch,
):
    import tessera.compiler.resident_nvidia_tensor as resident
    class Root:
        __cuda_array_interface__ = {
            "version": 3, "shape": (2, 3), "typestr": typestr,
            "data": (4096, True), "stream": 1, "strides": strides,
        }
        def __array__(self, *args, **kwargs):
            pytest.fail("metadata admission coerced resident data")
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported profile entered the strict compact-half parser")
    monkeypatch.setattr(resident, "cuda_frontend_specs", forbidden)
    assert input_signature((Root(), Root())) is None


def test_supported_resident_profile_still_checks_invalid_pointer():
    class Root:
        __cuda_array_interface__ = {
            "version": 3, "shape": (2, 3), "typestr": "<f2",
            "data": (0, True), "stream": 1, "strides": None,
        }
    with pytest.raises(ValueError, match="pointer metadata"):
        input_signature((Root(), Root()))


@pytest.fixture
def thread_owned_jit(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import threading
    import tessera as ts
    from tessera import runtime as rt
    import tessera.compiler.native_tensor_program as packages
    import tessera.compiler.scheduled_matmul as tools

    owners, compiled = [], []
    class Owner:
        def __init__(self):
            self.thread = threading.current_thread()
            self.handle = 1
            owners.append(self)
        def __call__(self, roots):
            assert threading.current_thread() is self.thread, "foreign thread reused native owner"
            assert self.handle
            return self
        def close(self):
            assert threading.current_thread() is self.thread, "foreign thread closed native owner"
            self.handle = 0
    class Program:
        members = ("producer", "consumer")
        def prepare(self):
            return Owner()
    def compile_product(*args, **kwargs):
        compiled.append(True)
        return Program()
    fn = ts.jit(target="nvidia_sm120")(norm_product)
    monkeypatch.setattr(fn, "_traced_autodiff_module", lambda *args, **kwargs: object())
    monkeypatch.setattr(packages, "compile_native_tensor_program", compile_product)
    monkeypatch.setattr(tools, "find_tessera_opt", lambda: Path("/configured/compiler"))
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch",
                        lambda: SimpleNamespace(tessera_nvidia_native_frame_prepare_v1=True))
    def roots(m=17):
        return np.zeros((m, 32), np.float16), np.zeros((32, 11), np.float16)
    return SimpleNamespace(fn=fn, roots=roots, owners=owners, compiled=compiled)


def test_thread_local_owners_cleanup_and_inspection(thread_owned_jit):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    env = thread_owned_jit
    fn, args = env.fn, env.roots()
    primary = fn._try_native_tensor_program_call(args, {})
    ready = threading.Barrier(3, timeout=10)
    resume = threading.Event()
    def worker():
        first = None
        try:
            first = fn._try_native_tensor_program_call(args, {})
            assert first is not primary
            assert fn._try_native_tensor_program_call(args, {}) is first
            assert fn._native_tensor_last_owner is first
            marker = object()
            fn._native_tensor_thread_state.artifact = marker
            ready.wait()
            assert resume.wait(10)
            assert first.handle and fn.runtime_artifact() is marker
            fn.close_native_storage()
            assert not first.handle
            rebound = fn._try_native_tensor_program_call(args, {})
            assert rebound is not first
            return first, rebound
        except BaseException:
            ready.abort()
            raise
        finally:
            fn.close_native_storage()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker) for _ in range(2)]
            try:
                try:
                    ready.wait()
                except threading.BrokenBarrierError:
                    for future in futures:
                        future.result(timeout=20)
                    raise
                assert fn._native_tensor_last_owner is primary
                fn.close_native_storage()
                assert not primary.handle
            finally:
                resume.set()
            pairs = [future.result(timeout=20) for future in futures]
            assert pairs[0][0] is not pairs[1][0]
        assert len(env.compiled) == 1
        assert all(not owner.handle for owner in env.owners)
    finally:
        resume.set()
        fn.close_native_storage()


def test_shared_product_eviction_never_closes_a_foreign_thread_owner(thread_owned_jit):
    from concurrent.futures import ThreadPoolExecutor
    env = thread_owned_jit
    fn = env.fn
    primary = fn._try_native_tensor_program_call(env.roots(), {})
    def worker():
        try:
            for m in range(32, 58):
                fn._try_native_tensor_program_call(env.roots(m), {})
                assert len(fn._native_tensor_calls) <= 24
            assert primary.handle
        finally:
            fn.close_native_storage()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker).result(timeout=20)
        assert primary.handle
        assert all(not owner.handle for owner in env.owners if owner is not primary)
        assert len(fn._native_tensor_program_cache) <= 24
    finally:
        fn.close_native_storage()
    assert not primary.handle
