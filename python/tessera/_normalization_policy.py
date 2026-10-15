"""Explicit eager normalization precision, matching native widening semantics."""
from collections.abc import Mapping
import numpy as np
from .dtype import canonicalize_dtype


def working_array(value, policy):
    if not isinstance(policy, Mapping) or set(policy) - {'storage', 'accum'}:
        raise ValueError('normalization numeric_policy supports storage and accum fields')
    array = np.asarray(value._data if hasattr(value, '_data') else value)
    storage = canonicalize_dtype(array.dtype.name)
    precision = {'fp16': (16, 11), 'bf16': (16, 8), 'fp32': (32, 24), 'fp64': (64, 53)}
    if storage not in precision:
        raise ValueError('normalization numeric_policy requires floating storage')
    if 'storage' in policy and canonicalize_dtype(policy['storage']) != storage:
        raise ValueError('normalization numeric_policy storage differs from input')
    accum = canonicalize_dtype(policy.get('accum', storage))
    if accum not in precision:
        raise ValueError('normalization numeric_policy requires floating accumulation')
    width, bits = precision[storage]
    accum_width, accum_bits = precision[accum]
    if accum_bits > bits:
        if accum_width <= width:
            raise ValueError('normalization accumulator cannot be represented by widening storage')
        working = array.astype({'fp32': np.float32, 'fp64': np.float64}[accum])
    else:
        working = array
    return working, array.dtype
