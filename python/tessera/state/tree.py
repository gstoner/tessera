"""Tessera-native pytree primitives — S-series sprint S3.

A pytree is a tree-of-containers whose leaves are tensors (or any Python
value the user marks as a leaf). This module provides:

  - `tree_flatten(tree)`            -> (leaves, treedef)
  - `tree_unflatten(treedef, leaves)` -> tree
  - `tree_map(fn, tree, *rest)`     -> tree
  - `tree_reduce(fn, tree, init=...)`-> Any
  - `tree_transpose(outer_def, inner_def, tree)` -> tree
  - `register_pytree_node(cls, flatten, unflatten)` for user containers

It is intentionally framework-independent: nothing here imports JAX, PyTorch,
or Flax. The vocabulary borrows from `jax.tree_util` and `flax.nnx`, but the
implementation, container handlers, and state taxonomy are Tessera's own.

Design notes:
  - **Determinism.** `tree_flatten` returns leaves in a canonical order — for
    dicts that means key-sorted; for lists/tuples that means
    insertion order. Tests exercise that two structurally-equal trees flatten
    to the same `treedef`.
  - **Round-trip.** `tree_unflatten(treedef, tree_flatten(tree)[0])` must
    reproduce the original `tree` exactly (including container types).
  - **State taxonomy.** S3's other deliverable is a typed view of model state
    (params / buffers / batch_stats / optimizer_slots / rng_state /
    recurrent_state / memory_state / metrics). `state_filter` /
    `state_partition` operate over `dict[str, ...]` state trees keyed on this
    taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, fields as dataclass_fields
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
)


# ───────────────────────────────────────────────────────────────────────────
# Pytree node registry
# ───────────────────────────────────────────────────────────────────────────


# A node handler is a pair (flatten_fn, unflatten_fn).
# - flatten_fn(node) -> (children, aux_data)
#     `children` is an iterable of subtrees in canonical order.
#     `aux_data` is any *hashable* metadata that's needed to reconstruct
#     the container (keys for a dict, type for a NamedTuple, etc.).
# - unflatten_fn(aux_data, children) -> node
_NODE_REGISTRY: dict[type, tuple[Callable[..., Any], Callable[..., Any]]] = {}


def register_pytree_node(
    cls: type,
    flatten: Callable[[Any], tuple[Iterable[Any], Hashable]],
    unflatten: Callable[[Hashable, Iterable[Any]], Any],
) -> None:
    """Register `cls` as an internal pytree node.

    A class registered here is no longer treated as a leaf — `tree_flatten`
    will descend into it. Built-in containers (dict, list, tuple, NamedTuple,
    dataclass) are pre-registered.
    """
    if cls in _NODE_REGISTRY:
        raise ValueError(f"pytree node already registered for {cls!r}")
    _NODE_REGISTRY[cls] = (flatten, unflatten)


# ───────────────────────────────────────────────────────────────────────────
# Built-in container handlers
# ───────────────────────────────────────────────────────────────────────────


def _flatten_dict(node: Mapping[Any, Any]) -> tuple[list[Any], tuple]:
    # Sort keys for canonical ordering. Two dicts with the same keys must
    # flatten to the same treedef regardless of insertion order.
    keys = sorted(node.keys(), key=lambda k: (type(k).__name__, k))
    children = [node[k] for k in keys]
    return children, ("dict", tuple(keys))


def _unflatten_dict(aux: tuple, children: Sequence[Any]) -> dict:
    _, keys = aux
    return dict(zip(keys, children))


def _flatten_list(node: list) -> tuple[list, tuple]:
    return list(node), ("list", len(node))


def _unflatten_list(aux: tuple, children: Sequence[Any]) -> list:
    return list(children)


def _flatten_tuple(node: tuple) -> tuple[list, tuple]:
    return list(node), ("tuple", len(node))


def _unflatten_tuple(aux: tuple, children: Sequence[Any]) -> tuple:
    return tuple(children)


# Pre-register the built-ins. NamedTuple and dataclass support is handled
# below via type detection rather than a static registration so that
# user-authored named tuples / dataclasses Just Work.
_NODE_REGISTRY[dict] = (_flatten_dict, _unflatten_dict)
_NODE_REGISTRY[list] = (_flatten_list, _unflatten_list)
_NODE_REGISTRY[tuple] = (_flatten_tuple, _unflatten_tuple)


def _is_namedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def _flatten_namedtuple(node: Any) -> tuple[list, tuple]:
    return list(node), ("namedtuple", type(node), tuple(node._fields))


def _unflatten_namedtuple(aux: tuple, children: Sequence[Any]) -> Any:
    _, cls, _ = aux
    return cls(*children)


def _flatten_dataclass(node: Any) -> tuple[list, tuple]:
    field_names = tuple(f.name for f in dataclass_fields(node))
    children = [getattr(node, name) for name in field_names]
    return children, ("dataclass", type(node), field_names)


def _unflatten_dataclass(aux: tuple, children: Sequence[Any]) -> Any:
    _, cls, names = aux
    return cls(**dict(zip(names, children)))


def _node_handler(node: Any):
    """Return (flatten, unflatten) for `node`, or `None` if it's a leaf."""
    cls = type(node)
    if cls in _NODE_REGISTRY:
        return _NODE_REGISTRY[cls]
    if _is_namedtuple(node):
        return _flatten_namedtuple, _unflatten_namedtuple
    if is_dataclass(node) and not isinstance(node, type):
        return _flatten_dataclass, _unflatten_dataclass
    return None


# ───────────────────────────────────────────────────────────────────────────
# treedef
# ───────────────────────────────────────────────────────────────────────────


class TreeDef:
    """Opaque, picklable description of a pytree's structure (no leaves).

    Two trees with the same structure produce equal `TreeDef`s, so
    `treedef_a == treedef_b` is the correct way to assert structural
    equivalence (e.g., for asserting `grad(f)` returns a tree that matches
    `params`).
    """

    __slots__ = ("aux", "children")

    def __init__(self, aux: Any, children: tuple["TreeDef", ...] | None) -> None:
        # `aux is None` and `children is None` together identify a leaf.
        self.aux = aux
        self.children = children

    @property
    def is_leaf(self) -> bool:
        return self.aux is None and self.children is None

    def num_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.num_leaves() for c in self.children)  # type: ignore[union-attr]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeDef):
            return NotImplemented
        if self.is_leaf:
            return other.is_leaf
        if other.is_leaf:
            return False
        return self.aux == other.aux and self.children == other.children

    def __hash__(self) -> int:
        if self.is_leaf:
            return hash("leaf")
        return hash((self.aux, self.children))

    def __repr__(self) -> str:
        if self.is_leaf:
            return "TreeDef(*)"
        return f"TreeDef({self.aux!r}, [{', '.join(repr(c) for c in self.children)}])"  # type: ignore[union-attr]


_LEAF_DEF = TreeDef(aux=None, children=None)


# ───────────────────────────────────────────────────────────────────────────
# Core operations
# ───────────────────────────────────────────────────────────────────────────


def tree_flatten(tree: Any) -> tuple[list[Any], TreeDef]:
    """Flatten a pytree into its leaves + a structural description."""
    leaves: list[Any] = []
    treedef = _flatten_into(tree, leaves)
    return leaves, treedef


def _flatten_into(node: Any, leaves: list[Any]) -> TreeDef:
    handler = _node_handler(node)
    if handler is None:
        leaves.append(node)
        return _LEAF_DEF
    flatten_fn, _ = handler
    children, aux = flatten_fn(node)
    sub_defs = tuple(_flatten_into(child, leaves) for child in children)
    # Tag the treedef with the class so unflatten can dispatch reliably,
    # including for user-registered containers whose aux is opaque.
    return TreeDef(aux=(type(node), aux), children=sub_defs)


def tree_unflatten(treedef: TreeDef, leaves: Iterable[Any]) -> Any:
    """Reconstruct a pytree from a treedef + a leaf iterable."""
    leaf_iter = iter(leaves)
    result = _unflatten_from(treedef, leaf_iter)
    extra = next(leaf_iter, _SENTINEL)
    if extra is not _SENTINEL:
        raise ValueError(
            "tree_unflatten received more leaves than the treedef expects"
        )
    return result


_SENTINEL = object()


def _unflatten_from(treedef: TreeDef, leaf_iter: Iterator[Any]) -> Any:
    if treedef.is_leaf:
        leaf = next(leaf_iter, _SENTINEL)
        if leaf is _SENTINEL:
            raise ValueError(
                "tree_unflatten ran out of leaves before the treedef was satisfied"
            )
        return leaf
    cls, inner_aux = treedef.aux  # set in `_flatten_into`
    children = [_unflatten_from(c, leaf_iter) for c in treedef.children]  # type: ignore[union-attr]
    # NamedTuple and dataclass instances aren't in _NODE_REGISTRY (they're
    # handled by type-detection in `_node_handler`); dispatch them directly.
    if isinstance(inner_aux, tuple) and inner_aux and inner_aux[0] == "namedtuple":
        return _unflatten_namedtuple(inner_aux, children)
    if isinstance(inner_aux, tuple) and inner_aux and inner_aux[0] == "dataclass":
        return _unflatten_dataclass(inner_aux, children)
    # Built-in or user-registered container — look up by class.
    if cls in _NODE_REGISTRY:
        _, unflatten_fn = _NODE_REGISTRY[cls]
        return unflatten_fn(inner_aux, children)
    raise ValueError(f"unknown pytree node class: {cls!r}")


def tree_map(fn: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    """Apply `fn` to each set of corresponding leaves across pytrees.

    All trees must share the same structure; otherwise raises ValueError.
    """
    main_leaves, treedef = tree_flatten(tree)
    rest_leaves = []
    for other in rest:
        other_leaves, other_def = tree_flatten(other)
        if other_def != treedef:
            raise ValueError(
                "tree_map received trees with different structures: "
                f"{treedef!r} vs {other_def!r}"
            )
        rest_leaves.append(other_leaves)
    new_leaves = [fn(*ls) for ls in zip(main_leaves, *rest_leaves)]
    return tree_unflatten(treedef, new_leaves)


def tree_reduce(
    fn: Callable[[Any, Any], Any],
    tree: Any,
    init: Any = _SENTINEL,
) -> Any:
    """Left-fold `fn` over a pytree's leaves.

    With `init` omitted the first leaf becomes the seed (matches functools).
    """
    leaves, _ = tree_flatten(tree)
    if init is _SENTINEL:
        if not leaves:
            raise ValueError("tree_reduce of empty pytree without initial value")
        acc = leaves[0]
        for leaf in leaves[1:]:
            acc = fn(acc, leaf)
        return acc
    acc = init
    for leaf in leaves:
        acc = fn(acc, leaf)
    return acc


def tree_transpose(outer_def: TreeDef, inner_def: TreeDef, tree: Any) -> Any:
    """Swap an outer/inner nesting in a pytree of pytrees.

    Example:
        outer = list of 3, inner = dict with keys {'a','b'}
        tree = [{'a':x0,'b':y0}, {'a':x1,'b':y1}, {'a':x2,'b':y2}]
        result = {'a':[x0,x1,x2], 'b':[y0,y1,y2]}
    """
    flat_outer = []
    for outer_child in _iter_outer(tree, outer_def):
        flat_inner, child_def = tree_flatten(outer_child)
        if child_def != inner_def:
            raise ValueError(
                "tree_transpose: inner subtree does not match `inner_def`"
            )
        flat_outer.append(flat_inner)
    # transpose: flat_outer[i][j]  ->  flat_outer[j][i]
    transposed = [list(col) for col in zip(*flat_outer)]
    inner_pieces = [tree_unflatten(outer_def, col) for col in transposed]
    return tree_unflatten(inner_def, inner_pieces)


def _iter_outer(tree: Any, outer_def: TreeDef) -> Iterator[Any]:
    """Yield the children of the outer level (one per outer leaf slot)."""
    leaves, td = tree_flatten(tree)
    # Recover the outer children by re-walking with a placeholder.
    # The simplest correct way: reconstruct the outer container by treating
    # its slots as leaves. We exploit the fact that `outer_def.num_leaves()`
    # gives us the count of outer slots.
    n = outer_def.num_leaves()
    # Group `leaves` into chunks whose count matches each outer slot.
    # Because all outer slots share `inner_def`, each chunk has size
    # `inner_def.num_leaves()`.
    chunk = len(leaves) // n
    if chunk * n != len(leaves):
        raise ValueError(
            "tree_transpose: leaf count is not a multiple of outer slots"
        )
    inner_def_ref = None
    for i in range(n):
        sub = leaves[i * chunk : (i + 1) * chunk]
        # Walk td to find the i-th outer child's def. For the supported
        # uniform shape (every outer slot has identical inner_def), every
        # outer child has the same shape — we can rebuild it from any one.
        # Find inner_def by descending td once.
        if inner_def_ref is None:
            inner_def_ref = _nth_outer_child_def(td, 0)
        yield tree_unflatten(inner_def_ref, sub)


def _nth_outer_child_def(td: TreeDef, n: int) -> TreeDef:
    if td.is_leaf:
        return td
    # First-level children of the outer container are the candidates.
    # `tree_transpose` requires homogeneous inner shape, so any child works.
    return td.children[n]  # type: ignore[index]


def tree_leaves(tree: Any) -> list[Any]:
    """Convenience: just the leaves, no treedef."""
    leaves, _ = tree_flatten(tree)
    return leaves


def tree_structure(tree: Any) -> TreeDef:
    """Convenience: just the structure, no leaves."""
    _, treedef = tree_flatten(tree)
    return treedef


def tree_all(predicate: Callable[[Any], bool], tree: Any) -> bool:
    return all(predicate(leaf) for leaf in tree_leaves(tree))


def tree_any(predicate: Callable[[Any], bool], tree: Any) -> bool:
    return any(predicate(leaf) for leaf in tree_leaves(tree))


# ───────────────────────────────────────────────────────────────────────────
# State taxonomy + filters/partitions
#
# Tessera's standalone-compiler state tree distinguishes 8 collection kinds.
# `nn.Module.state_dict()` produces a flat dict; here we expose the typed
# layered view that S10 (optimizers), S11 (losses), and S12 (checkpointing)
# all consume.
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StateCollectionSpec:
    """Semantic contract for one Tessera model-state collection."""

    name: str
    trainable: bool = False
    mutable: bool = False
    persistent: bool = True


STATE_COLLECTIONS: tuple[str, ...] = (
    "params",            # trainable parameters
    "buffers",           # non-trainable named tensors persisted in state_dict
    "batch_stats",       # running mean/var for BatchNorm-style layers
    "optimizer_slots",   # momentum/variance/etc. for Adam/AdamW/Lion/...
    "rng_state",         # serialized RNGKey state
    "recurrent_state",   # h/c for LSTM, hidden state for SSMs
    "memory_state",      # Titans/Atlas-style learned memory banks
    "metrics",           # running losses, accuracies, eval metrics
)

STATE_COLLECTION_SPECS: dict[str, StateCollectionSpec] = {
    "params": StateCollectionSpec("params", trainable=True, persistent=True),
    "buffers": StateCollectionSpec("buffers", persistent=True),
    "batch_stats": StateCollectionSpec("batch_stats", mutable=True, persistent=True),
    "optimizer_slots": StateCollectionSpec("optimizer_slots", mutable=True, persistent=True),
    "rng_state": StateCollectionSpec("rng_state", mutable=True, persistent=True),
    "recurrent_state": StateCollectionSpec("recurrent_state", mutable=True, persistent=False),
    "memory_state": StateCollectionSpec("memory_state", mutable=True, persistent=True),
    "metrics": StateCollectionSpec("metrics", mutable=True, persistent=False),
}


def empty_state_tree() -> dict[str, dict[str, Any]]:
    """Return an empty state tree with every known collection present."""
    return {name: {} for name in STATE_COLLECTIONS}


_BATCH_STAT_BASENAMES = frozenset(
    {"running_mean", "running_var", "num_batches_tracked"}
)


def _is_batch_stat_name(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _BATCH_STAT_BASENAMES


def module_state_tree(
    module: Any,
    *,
    include_empty: bool = True,
    extra: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Project a ``tessera.nn.Module`` into the S3 state taxonomy.

    Parameters and persistent buffers are copied out as numpy arrays so callers
    can filter, partition, serialize, or checkpoint the result without holding
    framework object handles. BatchNorm-style running-stat buffers are routed to
    ``batch_stats``; other persistent buffers go to ``buffers``.
    """
    state = empty_state_tree() if include_empty else {}

    def ensure(collection: str) -> dict[str, Any]:
        if collection not in STATE_COLLECTIONS:
            raise ValueError(f"unknown state collection: {collection}")
        return state.setdefault(collection, {})

    if not hasattr(module, "named_parameters") or not hasattr(module, "named_buffers"):
        raise TypeError("module_state_tree expects a tessera.nn.Module-like object")

    for name, param in module.named_parameters():
        ensure("params")[name] = param.numpy().copy()

    for name, buffer in module.named_buffers():
        if not getattr(buffer, "persistent", True):
            continue
        collection = "batch_stats" if _is_batch_stat_name(name) else "buffers"
        ensure(collection)[name] = buffer.numpy().copy()

    if extra is not None:
        for collection, values in extra.items():
            if collection not in STATE_COLLECTIONS:
                raise ValueError(f"unknown state collection: {collection}")
            ensure(collection).update(dict(values))

    if not include_empty:
        state = {k: v for k, v in state.items() if v}
    return state


def state_filter(state: Mapping[str, Any], keep: Iterable[str]) -> dict[str, Any]:
    """Return a new state dict containing only the listed collections."""
    keep_set = frozenset(keep)
    bad = keep_set - frozenset(STATE_COLLECTIONS)
    if bad:
        raise ValueError(f"unknown state collections: {sorted(bad)}")
    return {k: state[k] for k in keep_set if k in state}


def state_partition(
    state: Mapping[str, Any], *groups: Iterable[str]
) -> tuple[dict[str, Any], ...]:
    """Split a state dict into multiple disjoint group dicts.

    Example:
        params, opt = state_partition(state, ['params'], ['optimizer_slots'])
    """
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for group in groups:
        keep = list(group)
        # Detect overlap across groups (Decision: partitions are disjoint).
        overlap = seen.intersection(keep)
        if overlap:
            raise ValueError(
                f"state_partition groups overlap on: {sorted(overlap)}"
            )
        seen.update(keep)
        result.append(state_filter(state, keep))
    return tuple(result)


__all__ = [
    "TreeDef",
    "StateCollectionSpec",
    "tree_flatten",
    "tree_unflatten",
    "tree_map",
    "tree_reduce",
    "tree_transpose",
    "tree_leaves",
    "tree_structure",
    "tree_all",
    "tree_any",
    "register_pytree_node",
    "STATE_COLLECTIONS",
    "STATE_COLLECTION_SPECS",
    "empty_state_tree",
    "module_state_tree",
    "state_filter",
    "state_partition",
]
