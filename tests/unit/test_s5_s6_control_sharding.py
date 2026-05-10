"""S5/S6 standalone control-flow, transforms, sharding, and collectives."""

from __future__ import annotations

import numpy as np

import tessera as ts


def test_scan_matches_recurrent_loop_and_exposes_axis_context():
    xs = np.array([1.0, 2.0, 3.0])

    def step(carry, x):
        assert ts.axis_size("time") == 3
        y = carry + x + ts.axis_index("time")
        return y, y

    carry, ys = ts.scan(step, 0.0, xs, axis_name="time")
    assert carry == 9.0
    np.testing.assert_array_equal(ys, [1.0, 4.0, 9.0])


def test_associative_scan_and_structured_loops():
    xs = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(ts.associative_scan(lambda a, b: a + b, xs), [1, 3, 6, 10])

    assert ts.while_loop(lambda x: x < 8, lambda x: x * 2, 1) == 8
    assert ts.fori_loop(0, 4, lambda i, acc: acc + i, 0) == 6
    assert ts.cond(True, lambda x: x + 1, lambda x: x - 1, 10) == 11
    branches = (lambda x: x + 1, lambda x: x * 2, lambda x: x - 3)
    assert ts.switch(1, branches, 7) == 14


def test_map_pmap_and_axis_helpers():
    xs = np.array([10, 20, 30])

    def mapped(x):
        return x + ts.axis_index("items") + ts.axis_size("items")

    np.testing.assert_array_equal(ts.map(mapped, xs, axis_name="items"), [13, 24, 35])

    def f(x):
        return x + ts.axis_index("devices")

    np.testing.assert_array_equal(ts.pmap(f, axis_name="devices")(xs), [10, 21, 32])


def test_grad_vmap_and_value_and_grad_compose():
    def loss(x):
        return ts.ops.reduce(ts.ops.mul(x, x), op="sum")

    xs = np.array([[1.0, 2.0], [3.0, 4.0]])
    per_example_grads = ts.autodiff.vmap(ts.autodiff.grad(loss))(xs)
    np.testing.assert_array_equal(per_example_grads, 2.0 * xs)

    value, grad = ts.value_and_grad(loss)(np.array([2.0, 3.0]))
    np.testing.assert_allclose(value, 13.0)
    np.testing.assert_array_equal(grad, [4.0, 6.0])


def test_grad_through_scan_and_remat_scan_compose():
    def recurrent_loss(x):
        def step(carry, item):
            new_carry = ts.ops.add(carry, item)
            return new_carry, new_carry

        final, _ = ts.scan(step, np.array([0.0]), x)
        return ts.ops.mul(final, final)

    x = np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(ts.autodiff.grad(recurrent_loss)(x), [[12.0], [12.0], [12.0]])

    remat_loss = ts.autodiff.rematerialize(recurrent_loss)
    np.testing.assert_array_equal(remat_loss(x), recurrent_loss(x))


def test_named_mesh_partition_spec_and_named_sharding():
    mesh = ts.NamedMesh(("dp", "tp"), {"dp": 2, "tp": 4}, devices=range(8))
    spec = ts.partition_spec("dp", None)
    sharding = ts.named_sharding(mesh, spec, memory_kind="hbm")

    assert mesh.size == 8
    assert mesh.axis_size("tp") == 4
    assert sharding.spec == spec
    assert sharding.memory_kind == "hbm"


def test_shard_map_splits_and_reassembles_reference_outputs():
    mesh = ts.NamedMesh(("dp",), (2,))
    spec = ts.PartitionSpec("dp")

    def f(x):
        return x + ts.axis_index("dp")

    out = ts.shard_map(f, mesh=mesh, in_specs=spec, out_specs=spec)(np.arange(8))
    np.testing.assert_array_equal(out, [0, 1, 2, 3, 5, 6, 7, 8])


def test_collective_primitives_on_stacked_rank_values():
    values = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 7.0]])

    np.testing.assert_array_equal(ts.psum(values), [6.0, 14.0])
    np.testing.assert_array_equal(ts.pmean(values), [2.0, 14.0 / 3.0])
    np.testing.assert_array_equal(ts.pmax(values), [3.0, 7.0])
    np.testing.assert_array_equal(ts.pmin(values), [1.0, 2.0])
    np.testing.assert_array_equal(
        ts.collective_permute(values, ((0, 2), (1, 0), (2, 1))),
        [[3.0, 2.0], [2.0, 7.0], [1.0, 5.0]],
    )
    np.testing.assert_array_equal(ts.broadcast_to_axis(np.array([9, 8]), axis_size=3),
                                  [[9, 8], [9, 8], [9, 8]])
