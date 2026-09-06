"""Generate tensor bindings from ABI manifests preserved by native MLIR lowering.

The manifest is frontend/compiler ABI authority, not a proof of Python/kernel
semantic equivalence. Pointer-only target IR cannot reconstruct tensor shapes.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import re
from .native_gpu_storage import _decode_image
from .native_gpu_tensor import IndexSpec, TensorSpec, NativeTensorCall

ATTRIBUTE = 'tessera.native_tensor_contract'


def attach_tensor_contract(source: str, specs, *, grid, block) -> str:
    """Producer seam: emit the ABI once alongside its native kernel recipe."""
    if ATTRIBUTE in source:
        raise ValueError('native source already has a tensor contract')
    data = {'schema': 1, 'arguments': [dict(kind='tensor' if isinstance(s, TensorSpec) else 'index',
                                          **asdict(s)) for s in specs],
            'grid': list(grid), 'block': list(block)}
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
    encoded = encoded.replace('\\', '\\5C').replace('"', '\\22')
    result, count = re.subn(r'(?m)^module \{', lambda _: f'module attributes {{{ATTRIBUTE} = "{encoded}"}} {{', source)
    if count != 1:
        raise ValueError('tensor manifest producer requires one plain top-level module')
    return result


def read_tensor_contract(package):
    package.validate()
    matches = re.findall(r'tessera\.native_tensor_contract\s*=\s*"((?:\\.|[^"\\])*)"', package.arena_ir)
    if len(matches) != 1:
        raise ValueError('native package requires exactly one compiler-preserved tensor manifest')
    data = json.loads(_decode_image(matches[0]).decode('utf8'))
    if set(data) != {'schema', 'arguments', 'grid', 'block'} or type(data['schema']) is not int or data['schema'] != 1:
        raise ValueError('unsupported native tensor manifest')
    return data


def tensor_contract_specs(data):
    specs: list[TensorSpec | IndexSpec] = []
    for row in data['arguments']:
        row = dict(row)
        kind = row.pop('kind')
        if kind == 'tensor':
            if type(row['writable']) is not bool:
                raise ValueError('tensor writable flag must be boolean')
            row['shape'] = tuple(row['shape'])
            specs.append(TensorSpec(**row))
        elif kind == 'index':
            specs.append(IndexSpec(**row))
        else:
            raise ValueError('unknown tensor manifest argument')
    return tuple(specs)


def generate_tensor_binding(package, signature) -> NativeTensorCall:
    data = read_tensor_contract(package)
    return NativeTensorCall(package, signature, tensor_contract_specs(data), grid=tuple(data['grid']), block=tuple(data['block']))
