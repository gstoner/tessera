"""Native x86 absolute/floor packaging from replayed serialized contracts."""
from dataclasses import dataclass
import copy
import json
import re
from typing import ClassVar, cast
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class ScheduledAbsolute:
    kind: ClassVar[str] = "abs"
    contract_name: ClassVar[str] = "absolute"
    numeric_policy: ClassVar[str] = "ieee_abs_clear_sign"
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
        contracts = re.findall(rf'tessera\.{self.contract_name}_contract = \{{([^{{}}]+)\}}', self.tile_ir)
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
        for key, value in [('kind',self.kind), ('storage','f32'), ('layout','row_major'), ('numeric_policy',self.numeric_policy)]:
            if f'{key} = "{value}"' not in contract:
                raise ValueError('absolute contract has an unsupported policy')
        digest = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir)
        if len(digest) != 1:
            raise ValueError('absolute requires one Schedule identity')
        return bindings, dims, digest[0]


def lower_absolute(module):
    return _lower_unary(module, ScheduledAbsolute, "tessera.absolute")


def _lower_unary(module, artifact_type, op_name):
    # Only the frontend admission boundary handles Python Graph objects.
    from .x86_native import _elementwise_contract, _cohort2_contract
    if artifact_type.contract_name == "cumsum":
        scan = _cohort2_contract(module)
        if not scan or scan["family"] != "scan" or scan["kind"] != "sum":
            raise ValueError('scheduled cumsum requires a trailing-axis f32 scan')
        bindings = [cast(tuple[str, ...], scan["inputs"])[0], str(scan["output"])]
    else:
        contract = _elementwise_contract(module)
        if contract is None or contract[:2] != ('unary', artifact_type.kind):
            raise ValueError('scheduled unary requires its static same-shape f32 operation')
        bindings = [contract[2][0], contract[3]]
    target = copy.deepcopy(module)
    target.functions[0].body[0].op_name = op_name
    target.module_attrs.update({'tessera.target':'"x86"', 'tessera.arch':'"zen5-avx512"',
        'tessera.launch_bindings':json.dumps(bindings)})
    graph = target.to_mlir(target='x86', canonical=True)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError('absolute lowering requires the native compiler')
    schedule = run_tessera_opt(tool, graph, '--tessera-graph-to-schedule')
    tile = run_tessera_opt(tool, schedule, '--tessera-schedule-to-tile')
    return artifact_type(graph, schedule, tile)


def package_absolute(artifact, *, pipeline_name):
    return package_unary(artifact, pipeline_name=pipeline_name)


def package_unary(artifact, *, pipeline_name):
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
        provenance={'work_item':'E2E-REAL-6', 'route':'avx512_c_abi', 'family':'unary', 'kind':artifact.kind,
                    'shape':list(shape), 'elements':math.prod(shape), 'storage':'f32', 'output_storage':'f32',
                    'numeric_policy':artifact.numeric_policy, 'schedule_digest':digest})
    return x.X86NativePackage(artifact.tile_ir, target, target, image, descriptor)


@dataclass(frozen=True)
class ScheduledFloor(ScheduledAbsolute):
    kind: ClassVar[str] = "floor"
    contract_name: ClassVar[str] = "floor"
    numeric_policy: ClassVar[str] = "ieee_floor"


def lower_floor(module):
    return _lower_unary(module, ScheduledFloor, "tessera.floor")


@dataclass(frozen=True)
class ScheduledCeil(ScheduledAbsolute):
    kind: ClassVar[str] = "ceil"
    contract_name: ClassVar[str] = "ceil"
    numeric_policy: ClassVar[str] = "ieee_ceil"


def lower_ceil(module):
    return _lower_unary(module, ScheduledCeil, "tessera.ceil")


@dataclass(frozen=True)
class ScheduledCumsum(ScheduledAbsolute):
    kind: ClassVar[str] = "sum"
    contract_name: ClassVar[str] = "cumsum"
    numeric_policy: ClassVar[str] = "f32_inclusive_scan"


def lower_cumsum(module):
    return _lower_unary(module, ScheduledCumsum, "tessera.cumsum")


def package_cumsum(artifact, *, pipeline_name):
    from . import x86_native as x
    import math
    names, shape, digest = artifact.project()
    symbol, abi = 'tessera_x86_avx512_scan_f32', x.X86_SCAN_F32_ABI
    target, payload, compiler, toolchain = x._lower(artifact.tile_ir, symbol, 'scan')
    image = x._image(target_ir=target, payload=payload, compiler=compiler, toolchain=toolchain,
                     pipeline_name=pipeline_name, symbol=symbol, abi=abi)
    descriptor = x.LaunchDescriptor(image_digest=image.image_digest,entry_symbol=symbol,abi_id=abi,
        buffers=tuple(x.BufferBinding(i,name,'input' if i == 0 else 'output','fp32',len(shape),'row_major',4) for i,name in enumerate(names)),
        scalars=(x.ScalarArgument(2,'Rows','int64'),x.ScalarArgument(3,'Cols','int64')),
        shape_guards=tuple(x.ShapeGuard(name,axis,'eq',extent) for name in names for axis,extent in enumerate(shape)),
        geometry=x.LaunchGeometry(policy='x86_avx512_scan'),
        ordering=x.OrderingSemantics(ordered_submission=True,residency='all',synchronization=('return',)),
        provenance={'work_item':'E2E-REAL-6','route':'avx512_c_abi','family':'scan','kind':'sum',
          'shape':list(shape),'output_shape':list(shape),'rows':math.prod(shape[:-1]),'cols':shape[-1],
          'storage':'f32','inclusive':True,'numeric_policy':artifact.numeric_policy,'schedule_digest':digest})
    return x.X86NativePackage(artifact.tile_ir,target,target,image,descriptor)
