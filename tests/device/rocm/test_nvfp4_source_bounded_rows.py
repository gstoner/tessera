"""Bounded source-preserving frames on the owning gfx1201 runtime."""
import ctypes as C
import os
import subprocess
import sys

import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tests.device.rocm.test_prepared_nvfp4_expansion_program import frontend_product, inputs_and_oracle
from tests.unit.test_native_nvfp4_expansion_program import source_graph

pytestmark = pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
                               reason="requires owning gfx1201 and matched native rows provider")


@pytest.mark.parametrize("output", ["f32", "bf16"])
@pytest.mark.parametrize("roles", [(0, 1, 2, 3, 4), (2, 0, 4, 1, 3)])
def test_bounded_source_owner_shrink_grow_preserves_capacity_and_weights(output, roles):
    assert rt._rocm_live_arch() == "gfx1201"
    source = source_graph(19, 7, 48, roles, output).replace(
        'tessera.arch = "gfx1201"', 'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = 64 : i64')
    package = package_native_nvfp4_expansion_program(source)
    roots, expected = inputs_and_oracle(19, 7, 48, roles, output)
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
        initial = owner.rows_info()
        assert initial[:2] == (64, 19) and initial[3] == 8
        owner.import_weights()
        generation, _ = owner.invoke_consumer()
        np.testing.assert_array_equal(owner.read(generation)[0], expected)
        retained = owner.read(generation)[0]
        for m in (1, 64, 33, 19):
            new = list(roots)
            index = np.arange(m) % 19
            new[roles[3]] = roots[roles[3]][index].copy()
            new[roles[4]] = roots[roles[4]][index].copy()
            owner.update_activations(new[roles[3]], new[roles[4]])
            with pytest.raises(RuntimeError):
                owner.read(generation)
            generation, _ = owner.invoke_consumer()
            np.testing.assert_array_equal(owner.read(generation)[0], expected[index])
            stats = owner.rows_info()
            assert stats == (64, m, *initial[2:])
        np.testing.assert_array_equal(retained, expected)
        invalid = np.zeros((65, 48), dtype=roots[roles[3]].dtype)
        with pytest.raises(ValueError, match="capacity"):
            owner.update_activations(invalid, np.ones((65, 3), np.float32))
        np.testing.assert_array_equal(owner.read(generation)[0], expected)
        # Native rejection also leaves the current generation untouched.
        bad = owner.lib.tessera_rocm_program_update_rows
        bad.argtypes = [C.c_uint64, C.c_int64, C.c_uint32, C.POINTER(C.c_uint32),
                        C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
        a, sa = roots[roles[3]], roots[roles[4]]
        assert bad(owner.handle, 65, 2, (C.c_uint32 * 2)(*roles[3:]),
                   (C.c_void_p * 2)(a.ctypes.data, sa.ctypes.data),
                   (C.c_uint64 * 2)(a.nbytes, sa.nbytes)) == 1
        np.testing.assert_array_equal(owner.read(generation)[0], expected)


def test_bounded_source_aligned_lds_capacity_executes_ragged_active_rows():
    assert rt._rocm_live_arch() == "gfx1201"
    source = source_graph(19, 4096, 16).replace(
        'tessera.arch = "gfx1201"', 'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = 256 : i64')
    package = package_native_nvfp4_expansion_program(source)
    assert package.consumer.descriptor.provenance["staging"] == "lds"
    assert package.consumer.descriptor.provenance["lds_whole_m"] is False
    assert "whole_m = false" in package.consumer.target_ir
    roots, expected = inputs_and_oracle(19, 4096, 16, tuple(range(5)), "f32")
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
        owner.import_weights()
        for m in (19, 1, 256, 127):
            index = np.arange(m) % 19
            owner.update_activations(roots[3][index].copy(), roots[4][index].copy())
            generation, _ = owner.invoke_consumer()
            np.testing.assert_array_equal(owner.read(generation)[0], expected[index])


def test_bounded_source_public_compiler_free_replay(monkeypatch, tmp_path):
    assert rt._rocm_live_arch() == "gfx1201"
    function = ts.jit(target="rocm_gfx1201")(frontend_product)
    roots, expected = inputs_and_oracle(19, 7, 48, tuple(range(5)), "f32")
    program = function.compile_native_nvfp4_program(*roots, m_bound=64)
    from tessera.compiler.rocm_nvfp4_program import runtime_artifact
    artifact = rt.RuntimeArtifact.from_json(runtime_artifact(program).to_json())
    path = tmp_path / "bounded-source.json"
    path.write_text(artifact.to_json())
    data_path = tmp_path / "roots.npz"
    np.savez(data_path, **{f"a{i}": a for i, a in enumerate(roots)}, expected=expected)
    script = """
import sys
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler import rocm_nvfp4_expansion_program, nvfp4_expansion, reference_typed_scaled_matmul
def forbidden(*a, **kw): raise AssertionError("fresh replay invoked compiler or oracle")
rocm_nvfp4_expansion_program.package_native_nvfp4_expansion_program = forbidden
nvfp4_expansion.reference_nvfp4_expand = forbidden
reference_typed_scaled_matmul.reference_typed_scaled_matmul = forbidden
artifact = rt.RuntimeArtifact.from_json(open(sys.argv[1]).read())
data = np.load(sys.argv[2])
roots = [data[f"a{i}"] for i in range(5)]
for i in (1, 3): roots[i] = roots[i].view(ml_dtypes.float8_e4m3fn)
for m in (1, 64, 19):
    index = np.arange(m) % 19
    args = [*roots[:3], roots[3][index].copy(), roots[4][index].copy()]
    receipt = rt.launch(artifact, args)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    np.testing.assert_array_equal(receipt["output"], data["expected"][index])
"""
    child = subprocess.run([sys.executable, "-c", script, str(path), str(data_path)],
        env=dict(os.environ, TESSERA_OPT="/no/compiler", TESSERA_ROCM_OPT="/no/compiler"),
        text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    def forbidden(*args, **kwargs):
        pytest.fail("bounded source replay called compiler or frontend")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(function, "_fn", forbidden)
    monkeypatch.setattr(function, "_trace_frontend_capture", forbidden)
    for m in (19, 1, 64, 19):
        index = np.arange(m) % 19
        args = [*roots[:3], roots[3][index].copy(), roots[4][index].copy()]
        np.testing.assert_array_equal(function(*args), expected[index])
        receipt = rt.launch(artifact, args)
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        np.testing.assert_array_equal(receipt["output"], expected[index])



def test_bounded_source_full_update_recomputes_prefix_and_validates_before_state_change():
    source = source_graph().replace(
        'tessera.arch = "gfx1201"', 'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = 64 : i64')
    package = package_native_nvfp4_expansion_program(source)
    # Initial runtime M need not equal the trace M.
    roots, expected = inputs_and_oracle(1, 7, 48, tuple(range(5)), "f32")
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
        assert owner.rows_info()[:2] == (64, 1)
        generation, _ = owner.invoke()
        np.testing.assert_array_equal(owner.read(generation)[0], expected)
        initial = owner.rows_info()
        for m in (64, 19, 1):
            roots, expected = inputs_and_oracle(m, 7, 48, tuple(range(5)), "f32")
            owner.update(roots)
            assert owner.rows_info() == (64, m, *initial[2:])
            with pytest.raises(RuntimeError, match="status 10"):
                owner.invoke_consumer()
            generation, _ = owner.invoke()
            np.testing.assert_array_equal(owner.read(generation)[0], expected)
            invalid = list(roots)
            invalid[4] = np.ones((m, 4), np.float32)
            with pytest.raises(ValueError, match="storage/shape"):
                owner.update(invalid)
            assert owner.rows_info() == (64, m, *initial[2:])
            np.testing.assert_array_equal(owner.read(generation)[0], expected)


@pytest.mark.parametrize("fault", ["m_scalar", "grid_axis", "capacity", "duplicate",
                                   "a_width", "sa_width", "output_width", "row_bytes"])
def test_native_row_frame_rejects_invalid_contract_before_owner_creation(fault):
    from tessera.compiler.rocm_nvfp4_expansion_program import PreparedNVFP4ExpansionProgram, _Rows, _MultiOutputStep
    from tessera.compiler.native_scaled_program import _Buffer
    source = source_graph().replace(
        'tessera.arch = "gfx1201"', 'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = 64 : i64')
    package = package_native_nvfp4_expansion_program(source)
    unprepared = PreparedNVFP4ExpansionProgram.__new__(PreparedNVFP4ExpansionProgram)
    binding = unprepared._bind_abi(package)
    rows = unprepared._rows
    if fault == "m_scalar":
        rows.m_scalar = 1
    elif fault == "grid_axis":
        rows.grid_axis = 0
    elif fault == "capacity":
        rows.capacity_m = 18
    elif fault == "duplicate":
        rows.slots[1] = rows.slots[0]
    elif fault == "a_width":
        binding.buffers[3].elements //= 2
    elif fault == "sa_width":
        binding.buffers[4].elements *= 4
    elif fault == "output_width":
        binding.buffers[7].elements *= 4
    else:
        rows.row_bytes[1] += 4
    roots, _ = inputs_and_oracle(19, 7, 48, tuple(range(5)), "f32")
    library = C.CDLL(os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"])
    prepare = library.tessera_rocm_program_prepare_multi_output_rows
    prepare.argtypes = [C.c_char_p, C.c_uint32, C.c_uint32, C.POINTER(_Buffer), C.c_uint32,
                        C.POINTER(_MultiOutputStep), C.POINTER(C.c_void_p), C.POINTER(C.c_uint64),
                        C.POINTER(_Rows), C.c_int64, C.POINTER(C.c_uint64)]
    prepare.restype = C.c_int
    handle = C.c_uint64(999)
    assert prepare(b"gfx1201", 5, 8, binding.buffers, 2, binding.steps,
                   (C.c_void_p * 5)(*[a.ctypes.data for a in roots]),
                   (C.c_uint64 * 5)(*[a.nbytes for a in roots]),
                   C.byref(rows), 19, C.byref(handle)) == 1
    assert handle.value == 0
