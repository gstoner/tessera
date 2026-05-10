"""Tests for `tessera.state` pytree primitives (S-series sprint S3)."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pytest

import tessera as ts
from tessera.state import (
    STATE_COLLECTIONS,
    STATE_COLLECTION_SPECS,
    empty_state_tree,
    module_state_tree,
    register_pytree_node,
    state_filter,
    state_partition,
    tree_all,
    tree_any,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_reduce,
    tree_structure,
    tree_transpose,
    tree_unflatten,
)


# ── basic flatten/unflatten round-trip ─────────────────────────────────────


def test_leaf_round_trip():
    leaves, treedef = tree_flatten(42)
    assert leaves == [42]
    assert tree_unflatten(treedef, leaves) == 42


def test_dict_round_trip_canonicalizes_key_order():
    a = {"y": 1, "x": 2, "z": 3}
    b = {"z": 3, "x": 2, "y": 1}  # different insertion order
    leaves_a, def_a = tree_flatten(a)
    leaves_b, def_b = tree_flatten(b)
    assert def_a == def_b, "structurally-equal dicts must produce equal treedefs"
    assert leaves_a == leaves_b, "canonical (key-sorted) leaf order"
    # Round-trip preserves the dict.
    assert tree_unflatten(def_a, leaves_a) == a


def test_list_and_tuple_round_trip():
    tree = [1, (2, 3), {"k": 4}]
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3, 4]
    out = tree_unflatten(treedef, leaves)
    assert out == tree
    assert isinstance(out, list)
    assert isinstance(out[1], tuple)
    assert isinstance(out[2], dict)


def test_namedtuple_round_trip():
    Point = namedtuple("Point", ["x", "y"])
    tree = {"a": Point(1, 2), "b": [Point(3, 4)]}
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3, 4]
    out = tree_unflatten(treedef, leaves)
    assert isinstance(out["a"], Point)
    assert out["a"] == Point(1, 2)
    assert isinstance(out["b"][0], Point)


def test_dataclass_round_trip():
    @dataclass
    class State:
        weights: int
        bias: int

    tree = State(weights=5, bias=7)
    leaves, treedef = tree_flatten(tree)
    assert leaves == [5, 7]
    out = tree_unflatten(treedef, leaves)
    assert isinstance(out, State)
    assert out == State(weights=5, bias=7)


def test_nested_round_trip_with_arrays():
    tree = {
        "params": {
            "linear": {"weight": np.zeros((2, 3)), "bias": np.ones((3,))},
        },
        "rng": np.array([1, 2, 3]),
    }
    leaves, treedef = tree_flatten(tree)
    assert len(leaves) == 3
    rebuilt = tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(rebuilt["rng"], tree["rng"])
    np.testing.assert_array_equal(
        rebuilt["params"]["linear"]["weight"], tree["params"]["linear"]["weight"]
    )


# ── tree_map ────────────────────────────────────────────────────────────────


def test_tree_map_unary():
    tree = {"a": 1, "b": [2, 3], "c": (4,)}
    out = tree_map(lambda x: x * 10, tree)
    assert out == {"a": 10, "b": [20, 30], "c": (40,)}


def test_tree_map_binary_requires_matching_structure():
    a = {"x": 1, "y": 2}
    b = {"x": 10, "y": 20}
    out = tree_map(lambda u, v: u + v, a, b)
    assert out == {"x": 11, "y": 22}

    bad = {"x": 1, "z": 2}  # different keys
    with pytest.raises(ValueError, match="different structures"):
        tree_map(lambda u, v: u + v, a, bad)


def test_tree_map_preserves_container_types():
    Point = namedtuple("Point", ["x", "y"])
    tree = {"p": Point(1, 2), "lst": [3, 4]}
    out = tree_map(lambda x: x + 100, tree)
    assert isinstance(out["p"], Point)
    assert out["p"] == Point(101, 102)
    assert out["lst"] == [103, 104]


# ── tree_reduce ─────────────────────────────────────────────────────────────


def test_tree_reduce_with_init():
    tree = {"a": 1, "b": [2, 3], "c": (4, 5)}
    total = tree_reduce(lambda a, b: a + b, tree, init=0)
    assert total == 15


def test_tree_reduce_without_init_uses_first_leaf():
    tree = [10, 20, 30]
    total = tree_reduce(lambda a, b: a + b, tree)
    assert total == 60


def test_tree_reduce_empty_without_init_raises():
    with pytest.raises(ValueError, match="empty pytree"):
        tree_reduce(lambda a, b: a + b, [])


# ── tree_transpose ──────────────────────────────────────────────────────────


def test_tree_transpose_list_of_dicts_to_dict_of_lists():
    outer = [{"a": 0}, {"a": 1}, {"a": 2}]
    outer_def = tree_structure([0, 0, 0])
    inner_def = tree_structure({"a": 0})
    out = tree_transpose(outer_def, inner_def, outer)
    assert out == {"a": [0, 1, 2]}


# ── tree_structure equality + leaves convenience ────────────────────────────


def test_tree_structure_and_leaves_helpers():
    tree = {"a": 1, "b": (2, 3)}
    assert tree_leaves(tree) == [1, 2, 3]
    sd = tree_structure(tree)
    assert sd.num_leaves() == 3


def test_tree_all_and_any():
    tree = {"a": 1, "b": [2, 3]}
    assert tree_all(lambda x: isinstance(x, int), tree)
    assert not tree_all(lambda x: x > 1, tree)
    assert tree_any(lambda x: x > 2, tree)


# ── error messages on mismatched leaf count ─────────────────────────────────


def test_unflatten_too_few_leaves():
    _, treedef = tree_flatten({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="ran out of leaves"):
        tree_unflatten(treedef, [99])


def test_unflatten_too_many_leaves():
    _, treedef = tree_flatten({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="more leaves"):
        tree_unflatten(treedef, [1, 2, 3])


# ── treedef equality + hashability ──────────────────────────────────────────


def test_treedef_equality_and_hash():
    _, def_a = tree_flatten({"a": 1, "b": 2})
    _, def_b = tree_flatten({"a": 99, "b": -7})
    _, def_c = tree_flatten({"a": 1, "b": 2, "c": 3})
    assert def_a == def_b
    assert def_a != def_c
    assert hash(def_a) == hash(def_b)


# ── custom pytree node registration ─────────────────────────────────────────


class _UserContainer:
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail

    def __eq__(self, other):
        return (
            isinstance(other, _UserContainer)
            and self.head == other.head
            and self.tail == other.tail
        )


def test_register_pytree_node_for_user_class():
    register_pytree_node(
        _UserContainer,
        flatten=lambda obj: ([obj.head, obj.tail], ("user_container",)),
        unflatten=lambda aux, children: _UserContainer(children[0], children[1]),
    )
    tree = _UserContainer(1, _UserContainer(2, 3))
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3]
    out = tree_unflatten(treedef, leaves)
    assert out == tree


# ── state taxonomy ──────────────────────────────────────────────────────────


def test_state_collections_lock_the_taxonomy():
    expected = {
        "params", "buffers", "batch_stats", "optimizer_slots",
        "rng_state", "recurrent_state", "memory_state", "metrics",
    }
    assert set(STATE_COLLECTIONS) == expected


def test_state_collection_specs_distinguish_mutability_and_trainability():
    assert STATE_COLLECTION_SPECS["params"].trainable
    assert not STATE_COLLECTION_SPECS["buffers"].trainable
    assert STATE_COLLECTION_SPECS["batch_stats"].mutable
    assert STATE_COLLECTION_SPECS["recurrent_state"].mutable
    assert not STATE_COLLECTION_SPECS["recurrent_state"].persistent


def test_empty_state_tree_contains_every_collection():
    state = empty_state_tree()
    assert set(state) == set(STATE_COLLECTIONS)
    assert all(value == {} for value in state.values())


def test_module_state_tree_projects_params_buffers_and_batch_stats():
    class Block(ts.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = ts.nn.Linear(3, 2)
            self.bn = ts.nn.BatchNorm1d(2)
            self.register_buffer("mask", np.ones(2, dtype=np.float32))
            self.register_buffer("scratch", np.zeros(2, dtype=np.float32), persistent=False)

    state = module_state_tree(Block())
    assert "linear.weight" in state["params"]
    assert "linear.bias" in state["params"]
    assert "mask" in state["buffers"]
    assert "scratch" not in state["buffers"]
    assert "bn.running_mean" in state["batch_stats"]
    assert "bn.running_var" in state["batch_stats"]
    assert "bn.num_batches_tracked" in state["batch_stats"]


def test_module_state_tree_supports_extra_collections_and_filtering():
    m = ts.nn.Linear(2, 2)
    state = module_state_tree(
        m,
        include_empty=False,
        extra={
            "optimizer_slots": {"weight.momentum": np.zeros((2, 2))},
            "metrics": {"loss": 1.25},
        },
    )
    assert set(state) == {"params", "optimizer_slots", "metrics"}
    opt_and_metrics = state_filter(state, ["optimizer_slots", "metrics"])
    assert set(opt_and_metrics) == {"optimizer_slots", "metrics"}


def test_state_filter_keeps_only_requested_collections():
    state = {
        "params": {"w": 1},
        "optimizer_slots": {"m": 2},
        "rng_state": np.array([0]),
    }
    only_params = state_filter(state, ["params"])
    assert set(only_params.keys()) == {"params"}

    params_and_rng = state_filter(state, ["params", "rng_state"])
    assert set(params_and_rng.keys()) == {"params", "rng_state"}


def test_state_filter_rejects_unknown_collections():
    with pytest.raises(ValueError, match="unknown state collections"):
        state_filter({}, ["params", "definitely_not_a_collection"])


def test_state_partition_returns_disjoint_groups():
    state = {
        "params": {"w": 1},
        "buffers": {"b": 2},
        "optimizer_slots": {"m": 3},
    }
    trainables, non_trainables, opt = state_partition(
        state, ["params"], ["buffers"], ["optimizer_slots"]
    )
    assert trainables == {"params": {"w": 1}}
    assert non_trainables == {"buffers": {"b": 2}}
    assert opt == {"optimizer_slots": {"m": 3}}


def test_state_partition_rejects_overlapping_groups():
    with pytest.raises(ValueError, match="groups overlap"):
        state_partition(
            {"params": {}, "buffers": {}},
            ["params", "buffers"],
            ["params"],  # `params` appears in both groups
        )
