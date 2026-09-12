"""Native x86 absolute packaging; descriptors project the serialized contract."""
from dataclasses import dataclass
import copy
import json
import re
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class ScheduledAbsolute:
    graph_ir: str
    schedule_ir: str
    tile_ir: str

    def project(self):
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError('absolute packaging requires the native compiler')
        if run_tessera_opt(tool, self.graph_ir, '--tessera-graph-to-schedule') != self.schedule_ir:
            raise ValueError('absolute Schedule disagrees with Graph replay')
        if run_tessera_opt(tool, self.schedule_ir, '--tessera-schedule-to-tile') != self.tile_ir:
            raise ValueError('absolute Tile disagrees with Schedule replay')
        contracts = re.findall(r'tessera.absolute_contract = \{([^{}]+)\}', self.tile_ir)
        if len(contracts) != 1:
            raise ValueError('absolute requires one serialized contract')
        contract = contracts[0]
        names = re.search(r'bindings = (\[[^\]]+\])', contract)
        shape = re.search(r'shape = array<i64: ([0-9, ]+)>', contract)
        if names is None or shape is None:
            raise ValueError('absolute contract is missing bindings/shape')
        bindings = json.loads(names[1])
        dims = tuple(int(n.strip()) for n in shape[1].split(','))
        if len(bindings) != 2 or any(type(n) is not str or not n.isidentifier() for n in bindings):
            raise ValueError('absolute bindings require identifiers')
        for key, value in [('kind','abs'), ('storage','f32'), ('layout','row_major'), ('numeric_policy','ieee_abs_clear_sign')]:
            if f'{key} = "{value}"' not in contract:
                raise ValueError('absolute contract has an unsupported policy')
        digest = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir)
        if len(digest) != 1:
            raise ValueError('absolute requires one Schedule identity')
        return bindings, dims, digest[0]


def lower_absolute(module):
    # Only the frontend admission boundary handles Python Graph objects.
    from .x86_native import _elementwise_contract
    contract = _elementwise_contract(module)
    if contract is None or contract[:2] != ('unary', 'abs'):
        raise ValueError('scheduled absolute requires static same-shape f32 absolute')
    target = copy.deepcopy(module)
    target.functions[0].body[0].op_name = 'tessera.absolute'
    target.module_attrs.update({'tessera.target':'"x86"', 'tessera.arch':'"zen5-avx512"',
        'tessera.launch_bindings':json.dumps([contract[2][0], contract[3]])})
    graph = target.to_mlir(target='x86', canonical=True)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError('absolute lowering requires the native compiler')
    schedule = run_tessera_opt(tool, graph, '--tessera-graph-to-schedule')
    tile = run_tessera_opt(tool, schedule, '--tessera-schedule-to-tile')
    return ScheduledAbsolute(graph, schedule, tile)


def package_absolute(artifact, *, pipeline_name):
    from . import x86_native as x
    import math
    names, shape, digest = artifact.project()
    symbol, abi = 'tessera_x86_avx512_unary_f32', x.X86_UNARY_F32_ABI
    target, payload, compiler, toolchain = x._lower(artifact.tile_ir, symbol, 'elementwise')
    image = x._image(target_ir=target, payload=payload, compiler=compiler, toolchain=toolchain,
                     pipeline_name=pipeline_name, symbol=symbol, abi=abi)
    descriptor = x.LaunchDescriptor(image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=tuple(x.BufferBinding(i, name, 'input' if i == 0 else 'output', 'fp32', len(shape), 'row_major', 4)
                      for i, name in enumerate(names)),
        scalars=(x.ScalarArgument(2, 'N', 'int64'),),
        shape_guards=tuple(x.ShapeGuard(name, axis, 'eq', extent) for name in names for axis, extent in enumerate(shape)),
        geometry=x.LaunchGeometry(policy='x86_avx512_flat'),
        ordering=x.OrderingSemantics(ordered_submission=True, residency='all', synchronization=('return',)),
        provenance={'work_item':'E2E-REAL-6', 'route':'avx512_c_abi', 'family':'unary', 'kind':'abs',
                    'shape':list(shape), 'elements':math.prod(shape), 'storage':'f32', 'output_storage':'f32',
                    'numeric_policy':'ieee_abs_clear_sign', 'schedule_digest':digest})
    return x.X86NativePackage(artifact.tile_ir, target, target, image, descriptor)
