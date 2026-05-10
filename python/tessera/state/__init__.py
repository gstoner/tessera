"""Tessera-native state-tree primitives (S-series sprint S3).

Exports the pytree surface (`tree_flatten`/`unflatten`/`map`/`reduce`/...)
and the state taxonomy (`STATE_COLLECTIONS`, `state_filter`,
`state_partition`). See `tree.py` for design notes.
"""

from .tree import (
    STATE_COLLECTIONS,
    STATE_COLLECTION_SPECS,
    StateCollectionSpec,
    TreeDef,
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

__all__ = [
    "STATE_COLLECTIONS",
    "STATE_COLLECTION_SPECS",
    "StateCollectionSpec",
    "TreeDef",
    "empty_state_tree",
    "module_state_tree",
    "register_pytree_node",
    "state_filter",
    "state_partition",
    "tree_all",
    "tree_any",
    "tree_flatten",
    "tree_leaves",
    "tree_map",
    "tree_reduce",
    "tree_structure",
    "tree_transpose",
    "tree_unflatten",
]
