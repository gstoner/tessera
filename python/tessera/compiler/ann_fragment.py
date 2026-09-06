"""MSW-9: immutable affine/ReLU fragment views over existing Graph IR.

This module emits no IR and owns no backend route. Dense mathematical slots and
unique parameter storage are separate inventories. Composition is checked only
when the caller explicitly permits reassociation.
"""
from dataclasses import dataclass, replace
import hashlib
import math
import re
import numpy as np


@dataclass(frozen=True)
class ParameterSnapshot:
    storage_id: str
    shape: tuple[int, ...]
    data: bytes
    trainable: bool = True

    def __post_init__(self):
        if (type(self.trainable) is not bool or not isinstance(self.storage_id, str) or not self.storage_id or
                not isinstance(self.shape, tuple) or not self.shape or
                any(type(d) is not int or d <= 0 for d in self.shape) or
                type(self.data) is not bytes or len(self.data) != 4 * math.prod(self.shape) or
                not np.isfinite(np.frombuffer(self.data, dtype=np.float32)).all()):
            raise ValueError('invalid immutable ANN parameter snapshot')

    @classmethod
    def capture(cls, storage_id, value, *, trainable=True):
        value = np.asarray(value)
        if not isinstance(storage_id, str) or not storage_id or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('ANN parameters require named finite fp32 storage')
        return cls(storage_id, value.shape, value.tobytes(order='C'), trainable)

    def array(self):
        return np.frombuffer(self.data, dtype=np.float32).reshape(self.shape)


@dataclass(frozen=True)
class AffineLayer:
    weight: ParameterSnapshot
    bias: ParameterSnapshot
    activation: str | None


@dataclass(frozen=True)
class ANNFragment:
    layers: tuple[AffineLayer, ...]
    batch: int
    source_digest: str

    def __post_init__(self):
        if (type(self.batch) is not int or self.batch <= 0 or type(self.layers) is not tuple or not self.layers or
                not re.fullmatch(r'[0-9a-f]{64}', self.source_digest)):
            raise ValueError('invalid ANN fragment identity or batch')
        owners: dict[str, ParameterSnapshot] = {}
        width = None
        for layer in self.layers:
            if (not isinstance(layer, AffineLayer) or len(layer.weight.shape) != 2 or
                    layer.bias.shape != (layer.weight.shape[1],) or
                    layer.activation not in (None, 'relu') or
                    (width is not None and width != layer.weight.shape[0])):
                raise ValueError('ANN layer contract disagrees')
            width = layer.weight.shape[1]
            for value in (layer.weight, layer.bias):
                if value.storage_id in owners and owners[value.storage_id] != value:
                    raise ValueError('ANN sharing contract disagrees')
                owners[value.storage_id] = value

    @property
    def dense_slots(self):
        return sum(layer.weight.shape[0] * layer.weight.shape[1] + layer.bias.shape[0] for layer in self.layers)

    @property
    def unique_storage_slots(self):
        values = {v.storage_id: v for layer in self.layers for v in (layer.weight, layer.bias)}
        return sum(len(v.data) // 4 for v in values.values())

    @property
    def unique_trainable_slots(self):
        values = {v.storage_id: v for layer in self.layers for v in (layer.weight, layer.bias)}
        return sum(len(v.data) // 4 for v in values.values() if v.trainable)


def _shape(text):
    match = re.fullmatch(r'tensor<([1-9][0-9]*(?:x[1-9][0-9]*)*)xf32>', str(text))
    if not match:
        raise ValueError('ANN fragment requires static unencoded fp32 tensors')
    return tuple(map(int, match[1].split('x')))


def extract_ann_fragment(function, parameters):
    """Read a complete single-output affine/bias/ReLU chain with explicit owners.

    parameters maps argument SSA names to ParameterSnapshot. Unknown ownership,
    side effects, numeric policies and additional consumers fail closed.
    """
    if any('numeric' in key or 'math_mode' in key for key in function.fn_attrs):
        raise ValueError('ANN function numeric policy requires explicit lowering support')
    if function.structured_cfg is not None or len(function.return_values) != 1:
        raise ValueError('ANN fragment requires a flat single-output function')
    if any(a.effect not in (None, 'read') or a.layout or a.shard_spec for a in function.args):
        raise ValueError('ANN arguments require plain read-only tensor semantics')
    def ssa(name):
        return '%' + name.lstrip('%') if isinstance(name, str) else name
    body = [replace(op, result=ssa(op.result), operands=[ssa(v) for v in op.operands]) for op in function.body]
    if len({ssa(name) for name in parameters}) != len(parameters):
        raise ValueError('duplicate ANN parameter bindings')
    parameters = {ssa(name): value for name, value in parameters.items()}
    names = [ssa(a.name) for a in function.args] + [op.result for op in body]
    if len(names) != len(set(names)):
        raise ValueError('ANN SSA names must be unique')
    args = {ssa(a.name): _shape(a.ir_type) for a in function.args}
    if not set(parameters) <= set(args):
        raise ValueError('ANN parameter is not a function argument')
    inputs = set(args) - set(parameters)
    if len(inputs) != 1:
        raise ValueError('ANN fragment requires one input and explicit parameter ownership')
    owners: dict[str, ParameterSnapshot] = {}
    for name, value in parameters.items():
        if not isinstance(value, ParameterSnapshot) or args[name] != value.shape:
            raise ValueError('ANN parameter shape or ownership disagrees')
        # Even hand-constructed snapshots must be well-formed and finite.
        array = value.array()
        if not value.storage_id or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('ANN parameter snapshot is invalid')
        if value.storage_id in owners and owners[value.storage_id] != value:
            raise ValueError('shared ANN storage has inconsistent snapshots')
        owners[value.storage_id] = value
    current = inputs.pop()
    if len(args[current]) != 2:
        raise ValueError('ANN input must be a batch matrix')
    batch, width = args[current]
    layers = []
    used = set()
    ops = iter(body)
    pending = None
    while True:
        op = pending or next(ops, None)
        pending = None
        if op is None:
            break
        def admit(operation, names):
            if (operation.op_name not in names or operation.attrs or operation.kwargs or
                    operation.numeric_policy is not None or operation.result is None or
                    getattr(operation, 'regions', None)):
                raise ValueError('unsupported ANN operation or numeric policy')
        admit(op, ('tessera.matmul',))
        if len(op.operands) != 2 or op.operands[0] != current or op.operands[1] not in parameters:
            raise ValueError('ANN affine chain or weight ownership disagrees')
        weight = parameters[op.operands[1]]
        if len(weight.shape) != 2 or weight.shape[0] != width or _shape(op.result_type) != (batch, weight.shape[1]):
            raise ValueError('ANN matmul dimensions disagree')
        used.add(op.operands[1])
        bias_op = next(ops, None)
        if bias_op is None:
            raise ValueError('ANN affine layer requires explicit bias')
        admit(bias_op, ('tessera.add', 'tessera.bias_add'))
        if len(bias_op.operands) != 2 or bias_op.operands[0] != op.result or bias_op.operands[1] not in parameters:
            raise ValueError('ANN bias provenance disagrees')
        bias = parameters[bias_op.operands[1]]
        width = weight.shape[1]
        if bias.shape != (width,) or _shape(bias_op.result_type) != (batch, width):
            raise ValueError('ANN bias dimensions disagree')
        used.add(bias_op.operands[1])
        current = bias_op.result
        activation = None
        pending = next(ops, None)
        if pending is not None and pending.op_name == 'tessera.relu':
            admit(pending, ('tessera.relu',))
            if pending.operands != [current] or _shape(pending.result_type) != (batch, width):
                raise ValueError('ANN activation provenance disagrees')
            current, activation, pending = pending.result, 'relu', None
        layers.append(AffineLayer(weight, bias, activation))
    if not layers or [ssa(v) for v in function.return_values] != [current] or used != set(parameters):
        raise ValueError('ANN fragment has extra outputs or unused parameters')
    if tuple(map(_shape, function.result_types)) != ((batch, width),):
        raise ValueError('ANN public result shape disagrees')
    return ANNFragment(tuple(layers), batch, hashlib.sha256(function.to_mlir(canonical=True).encode()).hexdigest())


def validate_affine_composition(before, after, *, allow_reassociation=False):
    """Validate a concrete composition candidate before native pair evaluation."""
    if allow_reassociation is not True:
        raise ValueError('ANN composition requires explicit reassociation permission')
    if len(before.layers) != 2 or len(after.layers) != 1 or before.batch != after.batch:
        raise ValueError('ANN composition requires two affine layers replaced by one')
    first, second = before.layers
    fused = after.layers[0]
    if first.activation is not None or fused.activation != second.activation:
        raise ValueError('ANN composition may not remove or change an activation')
    if first.weight.shape[1] != second.weight.shape[0]:
        raise ValueError('ANN composition dimensions disagree')
    expected_weight = first.weight.array() @ second.weight.array()
    expected_bias = first.bias.array() @ second.weight.array() + second.bias.array()
    if not np.array_equal(fused.weight.array(), expected_weight) or not np.array_equal(fused.bias.array(), expected_bias):
        raise ValueError('ANN fused parameter values or dimensions disagree')
    return dict(before_dense_slots=before.dense_slots, after_dense_slots=after.dense_slots,
                before_unique_storage_slots=before.unique_storage_slots,
                after_unique_storage_slots=after.unique_storage_slots,
                before_unique_trainable_slots=before.unique_trainable_slots,
                after_unique_trainable_slots=after.unique_trainable_slots)
