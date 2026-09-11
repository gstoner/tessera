"""Bounded hook-free discovery of ordinary Python object graphs.

Copies scalar payloads and discovers cycles/aliases in exact builtin containers
and ordinary instance dictionaries. No imports, constructors, properties, slots,
custom iteration, arbitrary extension heaps or concurrent host mutation.
"""

from dataclasses import dataclass
import json
import math
import types
from typing import Any


@dataclass(frozen=True)
class ObjectSnapshot:
    payloads: tuple[bytes, ...]
    edges: tuple[tuple[int, ...], ...]
    roots: tuple[int, ...]


def discover_objects(*roots, max_nodes=256, max_bytes=262144, max_edges=32):
    if not roots or any(type(v) is not int or v <= 0 for v in (max_nodes, max_bytes, max_edges)):
        raise ValueError("object discovery requires roots and positive budgets")
    objects: list[Any] = []
    indices: dict[int, int] = {}
    payloads: list[bytes] = []
    edges: list[tuple[int, ...]] = []

    def intern(obj):
        identity = id(obj)
        if identity not in indices:
            if len(objects) >= max_nodes:
                raise ValueError("object graph exceeds node budget")
            indices[identity] = len(objects)
            objects.append(obj)
        return indices[identity]

    root_ids = tuple(intern(obj) for obj in roots)
    total = 0
    cursor = 0
    while cursor < len(objects):
        obj = objects[cursor]
        cls = type(obj)
        children: tuple[Any, ...] = ()
        if obj is None or cls in (str, int, bool):
            record = ["scalar", obj]
        elif cls is float:
            if not math.isfinite(obj):
                raise ValueError("object payload requires finite floats")
            record = ["scalar", obj]
        elif cls is bytes:
            record = ["bytes", obj.hex()]
        elif cls in (list, tuple):
            children = tuple(obj)
            record = ["list" if cls is list else "tuple"]
        elif cls is dict:
            items = tuple(dict.items(obj))
            if any(type(key) is not str for key, _ in items):
                raise ValueError("object dictionaries require exact string keys")
            record = ["dict", [key for key, _ in items]]
            children = tuple(value for _, value in items)
        else:
            mro = type.__getattribute__(cls, "__mro__")
            if any("__slots__" in type.__getattribute__(base, "__dict__") for base in mro):
                raise ValueError("slotted objects require a declared native layout")
            descriptor = next(
                (
                    type.__getattribute__(base, "__dict__")["__dict__"]
                    for base in mro
                    if "__dict__" in type.__getattribute__(base, "__dict__")
                ),
                None,
            )
            if not isinstance(descriptor, types.GetSetDescriptorType):
                raise ValueError("object discovery refuses custom dictionary descriptors")
            fields = object.__getattribute__(obj, "__dict__")
            if type(fields) is not dict or any(type(key) is not str for key in fields):
                raise ValueError("object fields require an ordinary string dictionary")
            items = tuple(dict.items(fields))
            namespace = type.__getattribute__(cls, "__dict__")
            if type(namespace.get("__module__", "")) is not str:
                raise ValueError("object class module requires a plain string")
            record = [
                "instance",
                namespace.get("__module__", ""),
                type.__getattribute__(cls, "__qualname__"),
                [key for key, _ in items],
            ]
            children = tuple(value for _, value in items)
        if len(children) > max_edges:
            raise ValueError("object graph exceeds reference budget")
        data = json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode()
        total += len(data)
        if total > max_bytes:
            raise ValueError("object graph exceeds payload budget")
        payloads.append(data)
        edges.append(tuple(intern(child) for child in children))
        cursor += 1
    return ObjectSnapshot(tuple(payloads), tuple(edges), root_ids)
