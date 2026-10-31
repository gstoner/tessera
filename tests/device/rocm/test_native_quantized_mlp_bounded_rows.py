"""Owning gfx1201/gfx1151 execution of the bounded-row native seven-member MLP.

GFX1201-NVFP4-BOUNDED-MLP-20261027. One owner holds capacity storage; every
row-dependent member launches at the active M while both weight expansions
stay resident.
"""
import ctypes as C
import json

import numpy as np
import pytest
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_mlp_program import (
    PreparedQuantizedMLPProgram, _RowFrame, _RowStep)
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tests.unit.test_native_quantized_mlp_program import source as _gfx1201_source

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]

#: The architectures with exact-device proof of the quantized MLP. gfx1151 was
#: promoted on Princess-Luna (GFX1151-QUANTIZED-MLP-20261030 slice 4); each
#: host proves only its own architecture, and no result transfers between them.
OWNING_ARCHES = ("gfx1201", "gfx1151")


def owning_arch():
    arch = rt._rocm_live_arch()
    if arch not in OWNING_ARCHES:
        pytest.skip(f"the quantized MLP proof needs an owning {'/'.join(OWNING_ARCHES)} device, not {arch}")
    return arch


def owning_target():
    return "rocm_" + owning_arch()


def source(m, h, k, n):
    """The authored MLP Graph, on the live owning architecture."""
    text, arch = _gfx1201_source(m, h, k, n), owning_arch()
    if arch == "gfx1201":
        return text
    assert text.count('"rocm_gfx1201"') == 1 and text.count('tessera.arch = "gfx1201"') == 1
    return text.replace('"rocm_gfx1201"', f'"rocm_{arch}"').replace(
        'tessera.arch = "gfx1201"', f'tessera.arch = "{arch}"')


def provider():
    owning_arch()
    lib = rt._load_rocm_native_movement_runtime()
    assert lib is not None
    return lib._name


def bounded(m, h, k, n, bound):
    text = source(m, h, k, n).replace(
        "tessera.native.quantized_product}",
        f"tessera.native.quantized_product, tessera.native.quantized_m_bound = {bound} : i64}}")
    return package_quantized_product_graph(text)


def padded(roots, capacity, seed):
    """Capacity roots whose first rows are ``roots``; the padding is arbitrary."""
    pad = inputs(capacity, *roots_shape(roots)[1:], seed=seed)[0]
    pad[: roots[0].shape[0]] = roots[0]
    return [pad, *roots[1:]]


def roots_shape(roots):
    m, k = roots[0].shape
    return m, roots[1].shape[0] // 2, k, roots[4].shape[0]


def static_rows(capacity_package, roots, capacity, lib, seed):
    """Static capacity program over padded rows: same schedule, no M guard."""
    with capacity_package.prepare(padded(roots, capacity, seed), runtime_library=lib) as owner:
        generation, _ = owner.invoke()
        return owner.read(generation)[0][: roots[0].shape[0]]


SHAPES = [((32, 48, 19), 8, (3, 8, 1, 5)),
          ((16, 16, 1), 1, (1,)),
          ((128, 12288, 256), 256, (37, 256, 1))]


@pytest.mark.parametrize("dims,bound,actives", SHAPES)
def test_bounded_mlp_matches_capacity_static_rows_bitwise(dims, bound, actives):
    lib = provider()
    h, k, n = dims
    package = bounded(actives[0], h, k, n, bound)
    assert package.bounded and package.gate.arbitrary_m and package.down.arbitrary_m
    # GFX1201-PORTABLE-BOUNDED-QUANTIZED-RECORD-20261028: the v5 record replays
    # both arbitrary-M products and the row frame.
    replay = type(package).from_manifest(json.loads(json.dumps(package.to_manifest())))
    assert replay.bounded and replay.gate.arbitrary_m and replay.down.arbitrary_m
    assert replay.binding_digest == package.binding_digest
    capacity = package_quantized_product_graph(source(bound, h, k, n))
    first = inputs(actives[0], h, k, n)
    with package.prepare(first, runtime_library=lib) as owner:
        assert owner.rows_info()[:2] == (bound, actives[0])
        for index, m in enumerate(actives):
            roots = inputs(m, h, k, n, seed=400 + index)
            before = [x.copy() for x in roots]
            owner.update(roots)
            assert owner.active_m == m and owner.rows_info()[1] == m
            generation, _ = owner.invoke()
            actual = owner.read(generation)[0]
            assert actual.shape == (m, n)
            np.testing.assert_array_equal(
                actual.view(np.uint16),
                static_rows(capacity, roots, bound, lib, seed=900 + index).view(np.uint16))
            for a, b in zip(roots, before, strict=True):
                np.testing.assert_array_equal(a, b)
            if m <= 64:
                # Independent staged math at the active shape.
                assert_compiled_chain(package_quantized_product_graph(source(m, h, k, n)),
                                      roots, actual, lib)


def test_activation_update_keeps_expanded_weights_resident():
    lib = provider()
    h, k, n, bound = 32, 48, 19, 16
    package = bounded(5, h, k, n, bound)
    capacity = package_quantized_product_graph(source(bound, h, k, n))
    roots = inputs(5, h, k, n)
    with package.prepare(roots, runtime_library=lib) as owner:
        owner.invoke()
        for m, seed in ((11, 31), (2, 32), (16, 33)):
            changed = inputs(m, h, k, n, seed=seed)
            # Same weights; only the activation (and with it M) changes.
            changed[1:] = roots[1:]
            owner.update_activations(changed[0])
            # Every private row slot is stale after a row change.
            for first in (2, 3, 4, 6):
                with pytest.raises(RuntimeError, match="10"):
                    owner.invoke_range(first, 1)
            # Steps 1 and 5 (both weight expansions) are never re-run.
            owner.invoke_range(0, 1)
            owner.invoke_range(2, 3)
            generation, _ = owner.invoke_range(6, 1)
            actual = owner.read(generation)[0]
            np.testing.assert_array_equal(
                actual.view(np.uint16),
                static_rows(capacity, changed, bound, lib, seed=seed).view(np.uint16))


def test_bounded_owner_refuses_rows_outside_its_frame():
    lib = provider()
    h, k, n, bound = 32, 48, 19, 8
    package = bounded(3, h, k, n, bound)
    with pytest.raises(ValueError, match="exceeds its capacity"):
        package.prepare(inputs(9, h, k, n), runtime_library=lib)
    with package.prepare(inputs(3, h, k, n), runtime_library=lib) as owner:
        with pytest.raises(ValueError, match="exceeds its capacity"):
            owner.update_activations(inputs(9, h, k, n)[0])
        native = owner.lib.tessera_rocm_program_update_rows
        native.argtypes = [C.c_uint64, C.c_int64, C.c_uint32, C.POINTER(C.c_uint32),
                           C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
        weights = np.ascontiguousarray(inputs(3, h, k, n)[1])
        x = np.ascontiguousarray(inputs(4, h, k, n)[0])

        def update(rows, slot, array):
            return native(owner.handle, rows, 1, (C.c_uint32 * 1)(slot),
                          (C.c_void_p * 1)(array.ctypes.data), (C.c_uint64 * 1)(array.nbytes))
        # A row change without the row-dependent activation root is refused.
        assert update(4, 1, weights) == 1
        # Rows above capacity, and an activation span for different rows.
        assert update(9, 0, x) == 1
        assert update(5, 0, x) == 1
        assert owner.rows_info()[1] == 3
        generation, _ = owner.invoke()
        assert owner.read(generation)[0].shape == (3, n)


def _tampered(package, lib, roots, mutate):
    """Prepare through the native ABI with one frame field changed."""
    owner = PreparedQuantizedMLPProgram.__new__(PreparedQuantizedMLPProgram)
    PreparedQuantizedMLPProgram.__init__(owner, package, roots, runtime_library=lib)
    owner.close()
    frame = owner._frame_struct
    slots, strides, rows = owner._frame_arrays
    copies = ((C.c_uint32 * len(slots))(*slots), (C.c_uint64 * len(strides))(*strides),
              (_RowStep * len(rows))(*[_RowStep(r.step, r.scalar, r.scalar_per_row, r.grid_axis,
                                                r.grid_tile, r.grid_per_row) for r in rows]))
    mutated = _RowFrame(frame.capacity_m, frame.slot_count, frame.step_count,
                        C.cast(copies[0], C.POINTER(C.c_uint32)),
                        C.cast(copies[1], C.POINTER(C.c_uint64)),
                        C.cast(copies[2], C.POINTER(_RowStep)))
    mutate(mutated, copies)
    binding = owner._binding
    arrays, addresses, sizes = owner._inputs(roots)
    prepare = owner.lib.tessera_rocm_program_prepare_multi_output_row_frame
    handle = C.c_uint64()
    code = prepare(owning_arch().encode(), binding.argument_count, len(binding.storage), binding.buffers,
                   len(binding.steps), binding.steps, addresses, sizes, C.byref(mutated),
                   owner._active_m, C.byref(handle))
    if handle.value:
        owner.lib.tessera_rocm_program_close(handle)
    return code


def _drop_step(index):
    def mutate(frame, copies):
        rows = copies[2]
        keep = [rows[i] for i in range(frame.step_count) if i != index]
        for i, row in enumerate(keep):
            rows[i] = row
        frame.step_count -= 1
    return mutate


def _set_step(index, field, value):
    def mutate(frame, copies):
        setattr(copies[2][index], field, value)
    return mutate


def _add_invariant_step(frame, copies):
    # Step 1 is the gate weight expansion; it touches no row slot.
    rows = (_RowStep * (frame.step_count + 1))(*copies[2], _RowStep(1, 0, 1, 0, 256, 1))
    copies = (copies[0], copies[1], rows)
    frame.steps = C.cast(rows, C.POINTER(_RowStep))
    frame.step_count += 1
    _add_invariant_step.keep = rows


def _drop_slot(frame, copies):
    # Without the gate/up parent the SiLU-mul member reads invariant storage.
    slots, strides = copies[0], copies[1]
    keep = [(slots[i], strides[i]) for i in range(frame.slot_count) if slots[i] != 11]
    for i, (slot, stride) in enumerate(keep):
        slots[i], strides[i] = slot, stride
    frame.slot_count -= 1


@pytest.mark.parametrize("mutate", [
    _drop_step(2), _add_invariant_step, _drop_slot,
    _set_step(0, "scalar_per_row", 2), _set_step(1, "grid_tile", 7),
    _set_step(2, "grid_axis", 1), _set_step(3, "scalar", 1),
], ids=["missing-row-step", "invariant-row-step", "missing-row-slot", "scalar-scale",
        "grid-tile", "grid-axis", "scalar-index"])
def test_native_row_frame_admission_is_derived_from_the_plan(mutate):
    # Capacity 64 gives every flat member a multi-workgroup grid, so a moved
    # grid axis or tile cannot coincide with the derived value.
    lib = provider()
    package = bounded(3, 32, 48, 19, 64)
    assert _tampered(package, lib, inputs(3, 32, 48, 19), mutate) == 1


def test_untampered_frame_prepares():
    lib = provider()
    package = bounded(3, 32, 48, 19, 64)
    assert _tampered(package, lib, inputs(3, 32, 48, 19), lambda frame, copies: None) == 0


def test_bounded_plan_keeps_original_request_and_capacity_source():
    package = bounded(3, 32, 48, 19, 8)
    plan = json.loads(package.program_json)
    assert plan["schema"] == "tessera.native.quantized_product.v5"
    assert "tessera.native.quantized_m_bound" in plan["original_graph_ir"]
    assert "tessera.native.quantized_m_bound" not in plan["source_graph_ir"]
    assert plan["active_m"] == 3 and plan["m_bound"] == 8
