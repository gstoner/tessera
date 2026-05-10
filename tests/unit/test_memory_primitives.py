import numpy as np

import tessera as ts


def test_memory_read_returns_weighted_top_k_values():
    table = ts.MemoryTable(
        keys=np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]),
        values=np.array([[10.0, 0.0], [0.0, 20.0], [8.0, 2.0]]),
    )

    result = ts.memory_read(table, np.array([1.0, 0.0]), top_k=2)

    np.testing.assert_array_equal(result.indices, np.array([0, 2]))
    assert result.values.shape == (2,)
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.values[0] > result.values[1]


def test_memory_write_appends_and_evicts_oldest_rows():
    table = ts.MemoryTable(
        keys=np.array([[1.0, 0.0], [0.0, 1.0]]),
        values=np.array([[1.0], [2.0]]),
    )

    updated = ts.memory_write(
        table,
        keys=np.array([[0.5, 0.5], [0.25, 0.75]]),
        values=np.array([[3.0], [4.0]]),
        max_entries=3,
    )

    assert updated.size == 3
    np.testing.assert_array_equal(updated.values[:, 0], np.array([2.0, 3.0, 4.0]))


def test_memory_evict_removes_explicit_indices_and_preserves_metadata():
    table = ts.MemoryTable(
        keys=np.array([[1.0], [2.0], [3.0]]),
        values=np.array([[10.0], [20.0], [30.0]]),
        metadata={"age": np.array([0, 1, 2])},
    )

    updated = ts.memory_evict(table, indices=[1])

    np.testing.assert_array_equal(updated.keys[:, 0], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(updated.metadata["age"], np.array([0, 2]))
