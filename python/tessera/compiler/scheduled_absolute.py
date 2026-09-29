"""Native x86 absolute/floor packaging from replayed serialized contracts."""
from dataclasses import dataclass
from collections import OrderedDict
import copy
import json
import re
from threading import RLock
from typing import ClassVar
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


_LOWER_CACHE_LIMIT = 64
_LOWER_CACHE: OrderedDict[tuple[object, ...], tuple[str, str]] = OrderedDict()
_LOWER_CACHE_LOCK = RLock()


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
    # Admit through the canonical native x86 contract; these packagers remain
    # as differential baselines for the migrated native Schedule route.
    from . import native_x86_kernel
    try:
        request = native_x86_kernel.admit(module)
    except ValueError as exc:
        raise ValueError(
            f"scheduled {artifact_type.contract_name} requires its static f32 operation: {exc}"
        ) from exc
    if request.record != artifact_type.contract_name:
        raise ValueError(f"scheduled {artifact_type.contract_name} requires its own operation")
    bindings = list(request.bindings)
    target = copy.deepcopy(module)
    target.functions[0].body[0].op_name = op_name
    target.module_attrs.update({'tessera.target':'"x86"', 'tessera.arch':'"zen5-avx512"',
        'tessera.launch_bindings':json.dumps(bindings)})
    graph = target.to_mlir(target='x86', canonical=True)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError('absolute lowering requires the native compiler')
    from . import x86_native as x
    identity = x._file_identity(tool)
    key = (graph, artifact_type, identity, run_tessera_opt) if identity is not None else None
    cached = None
    if key is not None:
        with _LOWER_CACHE_LOCK:
            cached = _LOWER_CACHE.get(key)
            if cached is not None:
                _LOWER_CACHE.move_to_end(key)
    if cached is None:
        schedule = run_tessera_opt(tool, graph, '--tessera-graph-to-schedule')
        tile = run_tessera_opt(tool, schedule, '--tessera-schedule-to-tile')
        if key is not None and identity == x._file_identity(tool):
            with _LOWER_CACHE_LOCK:
                _LOWER_CACHE[key] = (schedule, tile)
                _LOWER_CACHE.move_to_end(key)
                if len(_LOWER_CACHE) > _LOWER_CACHE_LIMIT:
                    _LOWER_CACHE.popitem(last=False)
    else:
        schedule, tile = cached
    return artifact_type(graph, schedule, tile)


def package_absolute(artifact, *, pipeline_name):
    return package_unary(artifact, pipeline_name=pipeline_name)


def package_unary(artifact, *, pipeline_name):
    from . import x86_native as x
    from copy import deepcopy
    # An exact immutable artifact is validated and lowered once per compiler
    # and native-image identity. A changed Graph, Schedule, Tile, pipeline, or
    # rebuilt toolchain misses; the first request always executes native replay.
    tool_identity = x._file_identity(x._tessera_opt())
    library_identity = x._file_identity(x._library_path(x.X86_AVX512_ARCHITECTURE))
    key = None
    if tool_identity is not None and library_identity is not None:
        key = ("scheduled_absolute", artifact, pipeline_name, tool_identity,
               library_identity, x._lower, x._image)
        with x._UNARY_PACKAGE_CACHE_LOCK:
            if cached := x._SCHEDULED_UNARY_PACKAGE_CACHE.get(key):
                x._SCHEDULED_UNARY_PACKAGE_CACHE.move_to_end(key)
                return deepcopy(cached)
    package = _package_unary_uncached(artifact, pipeline_name=pipeline_name)
    if key is not None and (tool_identity, library_identity) == (
        x._file_identity(x._tessera_opt()),
        x._file_identity(x._library_path(x.X86_AVX512_ARCHITECTURE)),
    ):
        with x._UNARY_PACKAGE_CACHE_LOCK:
            x._SCHEDULED_UNARY_PACKAGE_CACHE[key] = deepcopy(package)
            x._SCHEDULED_UNARY_PACKAGE_CACHE.move_to_end(key)
            if len(x._SCHEDULED_UNARY_PACKAGE_CACHE) > x._UNARY_PACKAGE_CACHE_LIMIT:
                x._SCHEDULED_UNARY_PACKAGE_CACHE.popitem(last=False)
    return package


def _package_unary_uncached(artifact, *, pipeline_name):
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
class ScheduledTrunc(ScheduledAbsolute):
    kind: ClassVar[str] = "trunc"
    contract_name: ClassVar[str] = "trunc"
    numeric_policy: ClassVar[str] = "ieee_trunc"


def lower_trunc(module):
    return _lower_unary(module, ScheduledTrunc, "tessera.trunc")


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
