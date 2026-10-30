"""Checked compiler-owned gfx1201 NVFP4 expansion packages.

Python serializes semantic Graph IR and validates ABI metadata. All numerical
lowering is performed by the native Graph/Schedule/Tile/ROCm passes.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ShapeGuard,
)
from .nvfp4_expansion import checked_expansion_inputs, nvfp4_expansion_policy
from .rocm_native import ROCMNativePackage, _compile_native_tile_ir
from .rocm_nvfp4_ingest_native import _execute_arrays, _required_field
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

NVFP4_EXPAND_ABI = "tessera.rocm.nvfp4_expand.five_memrefs.static.v1"
#: Shape-free image of a checked static expansion: the K16 group count is one
#: i64 scalar and the grid is (groups/256, N). Its descriptor binds the
#: capacity shape; a shape frame scales the scalar and grid x below it.
NVFP4_EXPAND_RUNTIME_K_ABI = "tessera.rocm.nvfp4_expand.five_memrefs.runtime_k.v1"
#: Single-projection image whose N is one i64 scalar on the flat grid; K stays
#: static, so the row/group split divides by a constant.
NVFP4_EXPAND_RUNTIME_N_ABI = "tessera.rocm.nvfp4_expand.five_memrefs.runtime_n.v1"
_DIRECTIVE = "tessera_rocm.nvfp4_expand"
_checked_targets: dict[str, str] = {}


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _contract(tile, target):
    if (tile.count("tile.nvfp4_expand_kernel") != 1
            or target.count("tessera_rocm.nvfp4_expand") != 1):
        raise ValueError("NVFP4 expansion requires its isolated native producer")
    names = tuple(json.loads(_required_field(r"bindings = (\[[^\]]+\])", tile)))
    n = int(_required_field(r"\bn = (\d+) : i64", target))
    k = int(_required_field(r"\bk = (\d+) : i64", target))
    offsets = tuple(json.loads(_required_field(r"row_offsets = (\[[^\]]+\])", target)))
    digest = _required_field(r'schedule_hash = "([0-9a-f]{64})"', target)
    entry = _required_field(r'name = "([^"]+)"', target)
    if (len(names) != 5 or len(set(names)) != 5
            or any(not isinstance(name, str) or not name for name in names)
            or n <= 0 or k <= 0 or k % 16 or n > (2**63 - 1) // k
            or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n
            or any(type(x) is not int for x in offsets)
            or any(a >= b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("NVFP4 expansion shape/binding/projection contract changed")
    return names, n, k, offsets, digest, entry


def _descriptor(image, info, graph_digest, schedule_ir_digest, tile_ir_digest, *, runtime_k=False,
                runtime_n=False, entry=None):
    from .native_artifact import ScalarArgument
    names, n, k, offsets, digest, static_entry = info
    shapes = ((n, k // 2), (n, k // 16), (len(offsets) - 1,), (n, k), (k // 16, n))
    dtypes = ("uint8", "fp8_e4m3", "fp64", "fp8_e4m3", "fp32")
    groups = k // 16
    extra = {"runtime_k": True} if runtime_k else {"runtime_n": True} if runtime_n else {}
    abi = (NVFP4_EXPAND_RUNTIME_K_ABI if runtime_k else NVFP4_EXPAND_RUNTIME_N_ABI if runtime_n
           else NVFP4_EXPAND_ABI)
    return LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry or static_entry,
        abi_id=abi,
        buffers=tuple(BufferBinding(
            i, name, "input" if i < 3 else "output", dtypes[i], len(shapes[i]),
            "row_major", 8 if i == 2 else 4 if i == 4 else 1)
            for i, name in enumerate(names)),
        scalars=((ScalarArgument(5, "groups", "int64"),) if runtime_k else
                 (ScalarArgument(5, "n", "int64"),) if runtime_n else ()),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", dim)
                           for name, shape in zip(names, shapes)
                           for axis, dim in enumerate(shape)),
        geometry=LaunchGeometry(grid=((groups + 255) // 256, n, 1) if runtime_k
                                else ((n * groups + 255) // 256, 1, 1),
                                workgroup=(256, 1, 1)),
        ordering=OrderingSemantics(ordered_submission=True, residency="none",
                                  synchronization=("completion",)),
        provenance={
            "work_item": "ROCM-NVFP4-INGEST-1",
            "route": "GraphIR->ScheduleIR->TileIR->ROCm Target IR->LLVM->HSACO",
            "numeric_policy": nvfp4_expansion_policy(), "row_offsets": list(offsets),
            "shape_nk": [n, k], "schedule_digest": digest,
            "graph_digest": graph_digest, "schedule_ir_digest": schedule_ir_digest,
            "tile_ir_digest": tile_ir_digest,
            "ownership": "private_outputs_distinct_readonly_inputs",
            **extra,
        })


def _checked_target(tile):
    """Re-derive the static Target a runtime-K image is the projection of."""
    from .rocm_native import _run_opt
    from .rocm_pipeline import ROCMExecutablePipeline, ROCMInputLevel, ROCMOutputLevel
    cached = _checked_targets.get(tile)
    if cached is None:
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("runtime-K NVFP4 expansion requires matching tessera-opt to re-derive its Target")
        config = ROCMExecutablePipeline(family="quant_fp", arch="gfx1201", input_level=ROCMInputLevel.TILE)
        cached = _run_opt(tool, tile, config.pass_pipeline(output=ROCMOutputLevel.TARGET))
        if len(_checked_targets) >= 64:
            _checked_targets.pop(next(iter(_checked_targets)))
        _checked_targets[tile] = cached
    return cached


@dataclass(frozen=True)
class NVFP4ExpansionPackage:
    graph_ir: str
    schedule_ir: str
    native: ROCMNativePackage

    @property
    def runtime_k(self):
        return self.native.descriptor.abi_id == NVFP4_EXPAND_RUNTIME_K_ABI

    @property
    def runtime_n(self):
        return self.native.descriptor.abi_id == NVFP4_EXPAND_RUNTIME_N_ABI

    def contract(self):
        """(names, n, k, offsets, digest, entry) of the checked static expansion.

        A runtime-K (or runtime-N) image records that extent as zero; its
        contract is the static Target it projects, which is re-derived from the
        Tile and never trusted."""
        target = (_checked_target(self.native.tile_ir) if self.runtime_k or self.runtime_n
                  else self.native.target_ir)
        return _contract(self.native.tile_ir, target)

    def validate(self):
        native = self.native
        image, desc = native.image, native.descriptor
        desc.validate_image(image)
        if (image.target != "rocm_gfx1201" or image.architecture != "gfx1201"
                or image.binary_format != "hsaco"
                or desc.abi_id not in (NVFP4_EXPAND_ABI, NVFP4_EXPAND_RUNTIME_K_ABI,
                                       NVFP4_EXPAND_RUNTIME_N_ABI)
                or image.target_ir_digest != _digest(native.target_ir)):
            raise ValueError("NVFP4 expansion native image/ABI/Target contract differs")
        info = self.contract()
        if re.findall(r'\bhash = "([0-9a-f]{64})"', self.schedule_ir) != [info[-2]]:
            raise ValueError("NVFP4 expansion Schedule record digest differs")
        entry = None
        if self.runtime_k or self.runtime_n:
            from .rocm_native import _directive_symbol, _shape_free_target_ir
            projection = _shape_free_target_ir(_checked_target(native.tile_ir), family="nvfp4_expand",
                                               directive=_DIRECTIVE, runtime_k=self.runtime_k,
                                               runtime_n=self.runtime_n)
            if native.target_ir != projection:
                raise ValueError("runtime-extent NVFP4 expansion image differs from its checked Target")
            entry = _directive_symbol(projection, _DIRECTIVE)
        expected = _descriptor(image, info, _digest(self.graph_ir), _digest(self.schedule_ir), _digest(native.tile_ir),
                               runtime_k=self.runtime_k, runtime_n=self.runtime_n, entry=entry)
        if desc != expected:
            raise ValueError("NVFP4 expansion descriptor differs from native contract")


def package_nvfp4_expansion(graph: str | GraphIRModule, *, runtime_k: bool = False,
                            runtime_n: bool = False):
    """Package the native expansion; ``runtime_k`` images its K-free projection.

    The runtime-K image is compiled from the identity projection of the
    checked static Target, so its kernel takes the K16 group count at launch
    while its descriptor still binds the compiled (capacity) shape."""
    if type(runtime_k) is not bool or type(runtime_n) is not bool or (runtime_k and runtime_n):
        raise ValueError("NVFP4 expansion takes at most one runtime extent, as a bool")
    source = graph if isinstance(graph, str) else graph.to_mlir(
        target="rocm_gfx1201", canonical=True)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("NVFP4 expansion requires matching tessera-opt")
    schedule = run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    entry = None
    if runtime_k or runtime_n:
        from .rocm_native import _directive_symbol, _shape_free_target_ir
        from .rocm_pipeline import ROCMInputLevel
        checked = _checked_target(tile)
        info = _contract(tile, checked)
        projections = len(info[3]) - 1
        if runtime_n and (info[1] % projections or list(info[3]) != [
                p * (info[1] // projections) for p in range(projections + 1)]):
            # The image derives each runtime boundary as p*N/P from its N scalar.
            raise ValueError("runtime-N NVFP4 expansion requires equally spaced projections")
        projection = _shape_free_target_ir(checked, family="nvfp4_expand", directive=_DIRECTIVE,
                                           runtime_k=runtime_k, runtime_n=runtime_n)
        target, backend, binary, compiler, toolchain, libraries, state = _compile_native_tile_ir(
            projection, directive=_DIRECTIVE, family="quant_fp", architecture="gfx1201",
            input_level=ROCMInputLevel.DIRECTIVE)
        if target != projection:
            raise ValueError("runtime-extent NVFP4 expansion Target changed during compilation")
        entry = _directive_symbol(projection, _DIRECTIVE)
    else:
        target, backend, binary, compiler, toolchain, libraries, state = _compile_native_tile_ir(
            tile, directive=_DIRECTIVE, family="quant_fp", architecture="gfx1201")
        info = _contract(tile, target)
    image = NativeImageArtifact(
        target="rocm_gfx1201", architecture="gfx1201", pipeline_name="tessera-rocm-executable",
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=_digest(target), binary_format="hsaco", payload=binary,
        entry_points=(NativeEntryPoint(entry or info[-1],
                                       NVFP4_EXPAND_RUNTIME_K_ABI if runtime_k else
                                       NVFP4_EXPAND_RUNTIME_N_ABI if runtime_n else NVFP4_EXPAND_ABI),),
        device_libraries=libraries, compile_state=state)
    desc = _descriptor(image, info, _digest(source), _digest(schedule), _digest(tile),
                       runtime_k=runtime_k, runtime_n=runtime_n, entry=entry)
    package = NVFP4ExpansionPackage(
        source, schedule, ROCMNativePackage(tile, target, backend, image, desc))
    package.validate()
    return package


def execute_nvfp4_expansion(package, codes, scales, projection_globals, *, event_samples=None):
    import ml_dtypes
    import numpy as np

    package.validate()
    _, n, k, offsets, _, _ = _contract(package.native.tile_ir, package.native.target_ir)
    codes, scales, globals_, _ = checked_expansion_inputs(
        codes, scales, projection_globals, row_offsets=offsets,
        numeric_policy=nvfp4_expansion_policy())
    if codes.shape != (n, k // 2):
        raise ValueError("NVFP4 expansion inputs differ from compiled shape")
    arrays = [np.ascontiguousarray(x) for x in (codes, scales, globals_)] + [
        np.empty((n, k), dtype=ml_dtypes.float8_e4m3fn),
        np.empty((k // 16, n), dtype=np.float32)]
    return _execute_arrays(package.native.image, package.native.descriptor, arrays,
                           event_samples=event_samples)


def build_nvfp4_expansion_graph(n: int, k: int, row_offsets, *, numeric_policy):
    """Build typed semantic Graph IR using the catalog's multi-result inference."""
    from .graph_ir import (
        GraphIRFunction,IRArg,IROp,tensor_ir_type,_infer_result_types)
    offsets=tuple(row_offsets)
    if (type(n) is not int or type(k) is not int or n <= 0 or k <= 0 or k % 16
            or n > (2**63 - 1) // k
            or len(offsets)<2 or any(type(x) is not int for x in offsets)
            or offsets[0]!=0 or offsets[-1]!=n
            or any(a>=b for a,b in zip(offsets,offsets[1:]))):
        raise ValueError("NVFP4 Graph requires positive static N/K16 and ordered boundaries")
    if numeric_policy != nvfp4_expansion_policy():
        raise ValueError("NVFP4 Graph requires the explicit expansion policy")
    types=[tensor_ir_type((n,k//2),"uint8"),
           tensor_ir_type((n,k//16),"fp8_e4m3"),
           tensor_ir_type((len(offsets)-1,),"fp64")]
    attrs={"row_offsets":list(offsets),"numeric_policy":dict(numeric_policy)}
    results=_infer_result_types("tessera.nvfp4_expand",types,attrs)
    operation=IROp(result="values, scale_plane",
        op_name="tessera.nvfp4_expand",
        operands=["%codes","%scales","%globals"],
        operand_types=[str(t) for t in types],
        result_type="("+", ".join(str(t) for t in results)+")",
        kwargs=attrs,inferred_type=results[0],inferred_types=tuple(results))
    fn=GraphIRFunction("expand",
        args=[IRArg(name,t) for name,t in zip(("codes","scales","globals"),types)],
        result_types=list(results),body=[operation],
        return_values=["%values","%scale_plane"],
        fn_attrs={"tessera.bindings":json.dumps(
            ["codes","scales","globals","values","scale_plane"])})
    return GraphIRModule([fn],module_attrs={
        "tessera.target":json.dumps("rocm_gfx1201"),"tessera.arch":json.dumps("gfx1201")})


def supports_nvfp4_expansion(module):
    """Recognize the semantic operation; native verification owns admission."""
    return (len(module.functions) == 1 and len(module.functions[0].body) == 1
            and module.functions[0].body[0].op_name == "tessera.nvfp4_expand")


def project_nvfp4_expansion_graph(module):
    from copy import deepcopy

    if not supports_nvfp4_expansion(module):
        raise ValueError("NVFP4 expansion requires one isolated semantic Graph operation")
    source = deepcopy(module)
    for key, value in (("tessera.target", "rocm_gfx1201"), ("tessera.arch", "gfx1201")):
        existing = source.module_attrs.get(key)
        if existing is not None and json.loads(existing) != value:
            raise ValueError("NVFP4 expansion Graph target/architecture conflicts")
        source.module_attrs[key] = json.dumps(value)
    fn = source.functions[0]
    op = fn.body[0]
    names = [arg.name for arg in fn.args] + op.result_names
    if (len(fn.args) != 3 or len(op.result_names) != 2 or len(set(names)) != 5
            or fn.return_values != ["%" + name for name in op.result_names]):
        raise ValueError("NVFP4 expansion argument/result ownership differs")
    existing = fn.fn_attrs.get("tessera.bindings")
    if existing is not None and json.loads(existing) != names:
        raise ValueError("NVFP4 expansion bindings conflict with operand/result lineage")
    fn.fn_attrs["tessera.bindings"] = json.dumps(names)
    return source


def expansion_runtime_artifact(package):
    from tessera.runtime import RuntimeArtifact

    package.validate()
    return RuntimeArtifact(
        graph_ir=package.graph_ir, schedule_ir=package.schedule_ir,
        tile_ir=package.native.tile_ir, target_ir=package.native.target_ir,
        native_image=package.native.image, launch_descriptor=package.native.descriptor,
        metadata={"target": "rocm_gfx1201", "compiler_path": "canonical_rocm_nvfp4_expand",
                  "executable": True, "execution_kind": "native_gpu"})


def validate_expansion_runtime_artifact(artifact):
    NVFP4ExpansionPackage(artifact.graph_ir, artifact.schedule_ir, ROCMNativePackage(
        artifact.tile_ir, artifact.target_ir, "", artifact.native_image,
        artifact.launch_descriptor)).validate()


def submit_nvfp4_expansion(image, desc, buffers, scalars, stream):
    """Common runtime checked host-buffer ABI; private device output storage."""
    import ml_dtypes
    import numpy as np

    if scalars or stream is not None:
        raise ValueError("NVFP4 expansion has no scalars and completes on the HIP default stream")
    desc.validate_image(image)
    if (image.target != "rocm_gfx1201" or image.architecture != "gfx1201"
            or image.binary_format != "hsaco" or desc.abi_id != NVFP4_EXPAND_ABI):
        raise ValueError("NVFP4 expansion requires its exact gfx1201 HSACO ABI")
    ordered = sorted(desc.buffers, key=lambda b: b.ordinal)
    names = tuple(b.name for b in ordered)
    info = desc.provenance
    n, k = info["shape_nk"]
    offsets = tuple(info["row_offsets"])
    if (type(n) is not int or type(k) is not int or n <= 0 or k <= 0 or k % 16
            or n > (2**63-1) // k or len(names) != 5 or len(set(names)) != 5
            or len(offsets) < 2 or any(type(x) is not int for x in offsets)
            or offsets[0] != 0 or offsets[-1] != n
            or any(a >= b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("NVFP4 expansion descriptor geometry/binding contract changed")
    contract = names, n, k, offsets, info["schedule_digest"], desc.entry_symbol
    expected = _descriptor(image, contract, info["graph_digest"],
                           info["schedule_ir_digest"], info["tile_ir_digest"])
    if desc != expected or set(buffers) != set(names):
        raise ValueError("NVFP4 expansion descriptor/bindings differ from native ABI")
    codes, scales, globals_, _ = checked_expansion_inputs(
        *(buffers[name] for name in names[:3]), row_offsets=offsets,
        numeric_policy=nvfp4_expansion_policy())
    if codes.shape != (n, k // 2):
        raise ValueError("NVFP4 expansion inputs differ from compiled shape")
    inputs = [codes, scales, globals_]
    outputs = [np.asarray(buffers[name]) for name in names[3:]]
    for output, shape, dtype in zip(outputs, ((n, k), (k // 16, n)),
                                    (ml_dtypes.float8_e4m3fn, np.float32)):
        if (output.shape != shape or output.dtype != np.dtype(dtype)
                or not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("NVFP4 expansion output shape/storage/writeability differs from ABI")
    for i, output in enumerate(outputs):
        if any(np.shares_memory(output, other) for other in inputs + outputs[:i]):
            raise ValueError("NVFP4 expansion outputs must not alias inputs or one another")
    arrays = [np.ascontiguousarray(x) for x in inputs] + outputs
    return _execute_arrays(image, desc, arrays)
