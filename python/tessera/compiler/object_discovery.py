"""Bounded hook-free discovery of ordinary Python object graphs.

Copies scalar payloads and discovers cycles/aliases in exact builtin containers
and ordinary instance dictionaries. No imports, constructors, properties,
custom iteration, arbitrary extension heaps or concurrent host mutation.
Declared slots are opt-in and read only through native member descriptors.
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


@dataclass(frozen=True)
class ExtensionLayout:
    """Explicit trusted extractor for one exact extension type.

    The caller keeps the object graph quiescent. The extractor returns opaque
    bytes and an exact tuple of strong referents; no global registry or implicit
    traversal of extension internals is used.
    """
    object_type: type
    schema: str
    extract: Any



def discover_objects(*roots, max_nodes=256, max_bytes=262144, max_edges=32, allow_slots=False, extension_layouts=()):
    if type(allow_slots) is not bool:
        raise ValueError("allow_slots must be boolean")
    if not roots or any(type(v) is not int or v <= 0 for v in (max_nodes, max_bytes, max_edges)):
        raise ValueError("object discovery requires roots and positive budgets")
    if type(extension_layouts) is not tuple:
        raise ValueError("extension layouts require an explicit tuple")
    layouts = {}
    for layout in extension_layouts:
        if (type(layout) is not ExtensionLayout or not isinstance(layout.object_type, type)
                or type(layout.schema) is not str or not layout.schema or not callable(layout.extract)
                or id(layout.object_type) in layouts):
            raise ValueError("extension layouts require unique exact types, schemas and extractors")
        if any(layout.object_type is builtin for builtin in (type(None), str, int, bool, float, bytes, bytearray, list, tuple, dict, set, frozenset)):
            raise ValueError("extension layouts cannot override builtin discovery")
        layouts[id(layout.object_type)] = layout
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
        if id(cls) in layouts:
            namespace = type.__getattribute__(cls, "__dict__")
            module = (namespace["__module__"] if "__module__" in namespace else
                      type.__dict__["__module__"].__get__(cls))
            if type(module) is not str:
                raise ValueError("object class module requires a plain string")
            layout = layouts[id(cls)]
            extracted = layout.extract(obj)
            if (type(extracted) is not tuple or len(extracted) != 2
                    or type(extracted[0]) is not bytes or type(extracted[1]) is not tuple):
                raise ValueError("extension extractor requires bytes and a tuple of referents")
            payload, children = extracted
            if len(payload) > max_bytes:
                raise ValueError("object graph exceeds payload budget")
            record = ["extension", module, type.__getattribute__(cls, "__qualname__"),
                      layout.schema, payload.hex()]
        elif type(cls) is not type:
            raise ValueError("custom metaclass requires an explicit extension layout")
        elif obj is None or any(cls is builtin for builtin in (str, int, bool)):
            record = ["scalar", obj]
        elif cls is float:
            if not math.isfinite(obj):
                raise ValueError("object payload requires finite floats")
            record = ["scalar", obj]
        elif cls is bytes or cls is bytearray:
            record = ["bytes" if cls is bytes else "bytearray", obj.hex()]
        elif cls is list or cls is tuple:
            children = tuple(obj)
            record = ["list" if cls is list else "tuple"]
        elif cls is set or cls is frozenset:
            # Exact builtin iteration does not call element hash/equality hooks.
            children = tuple(obj)
            record = ["set" if cls is set else "frozenset"]
        elif cls is dict:
            items = tuple(dict.items(obj))
            if all(type(key) is str for key, _ in items):
                record = ["dict", [key for key, _ in items]]
                children = tuple(value for _, value in items)
            else:
                record = ["mapping"]
                children = tuple(child for pair in items for child in pair)
        else:
            namespace = type.__getattribute__(cls, "__dict__")
            module = namespace.get("__module__", "")
            if type(module) is not str:
                raise ValueError("object class module requires a plain string")
            mro = type.__getattribute__(cls, "__mro__")
            for base in mro:
                if base is object:
                    continue
                base_fields = type.__getattribute__(base, "__dict__")
                native_descriptors = (types.MethodDescriptorType, types.WrapperDescriptorType,
                                      types.GetSetDescriptorType)
                if (not (type.__getattribute__(base, "__flags__") & (1 << 9)) or
                        any(any(type(value) is kind for kind in native_descriptors) for name, value in base_fields.items()
                            if name not in ("__dict__", "__weakref__"))):
                    raise ValueError("builtin subclass payload requires an explicit exact-type extension layout")
            if any("__slots__" in type.__getattribute__(base, "__dict__") for base in mro):
                if not allow_slots:
                    raise ValueError("slotted objects require a declared native layout")
                if any(base is not object and "__slots__" not in type.__getattribute__(base, "__dict__") for base in mro):
                    raise ValueError("slotted discovery requires a fully declared slot hierarchy")
                fields = []
                for base in reversed(mro):
                    namespace = type.__getattribute__(base, "__dict__")
                    owner_module = namespace.get("__module__", "")
                    if type(owner_module) is not str:
                        raise ValueError("object class module requires a plain string")
                    declared = namespace.get("__slots__", ())
                    declared = (declared,) if type(declared) is str else declared
                    if (type(declared) is not tuple and type(declared) is not list) or any(type(n) is not str for n in declared):
                        raise ValueError("slots require a plain string declaration")
                    for name in declared:
                        if name == '__weakref__':
                            continue
                        if name == '__dict__' or name.startswith('__'):
                            raise ValueError("dictionary and private slots need an explicit layout")
                        descriptor = namespace.get(name)
                        if type(descriptor) is not types.MemberDescriptorType or descriptor.__objclass__ is not base:
                            raise ValueError("slots require native member descriptors")
                        try:
                            value = descriptor.__get__(obj, cls)
                        except AttributeError:
                            continue
                        fields.append((owner_module + '.' + type.__getattribute__(base, '__qualname__') + ':' + name, value))
                record = ['slotted_instance', module, type.__getattribute__(cls, '__qualname__'), [n for n,_ in fields]]
                children = tuple(value for _,value in fields)
                if len(children) > max_edges:
                    raise ValueError("object graph exceeds reference budget")
                data = json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode()
                total += len(data)
                if total > max_bytes:
                    raise ValueError("object graph exceeds payload budget")
                payloads.append(data)
                edges.append(tuple(intern(child) for child in children))
                cursor += 1
                continue
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
            record = [
                "instance",
                module,
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
