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


def _descriptor(image, info, graph, schedule, tile):
    names, n, k, offsets, digest, entry = info
    shapes = ((n, k // 2), (n, k // 16), (len(offsets) - 1,), (n, k), (k // 16, n))
    dtypes = ("uint8", "fp8_e4m3", "fp64", "fp8_e4m3", "fp32")
    return LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry, abi_id=NVFP4_EXPAND_ABI,
        buffers=tuple(BufferBinding(
            i, name, "input" if i < 3 else "output", dtypes[i], len(shapes[i]),
            "row_major", 8 if i == 2 else 4 if i == 4 else 1)
            for i, name in enumerate(names)),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", dim)
                           for name, shape in zip(names, shapes)
                           for axis, dim in enumerate(shape)),
        geometry=LaunchGeometry(grid=((n * (k // 16) + 255) // 256, 1, 1),
                                workgroup=(256, 1, 1)),
        ordering=OrderingSemantics(ordered_submission=True, residency="none",
                                  synchronization=("completion",)),
        provenance={
            "work_item": "ROCM-NVFP4-INGEST-1",
            "route": "GraphIR->ScheduleIR->TileIR->ROCm Target IR->LLVM->HSACO",
            "numeric_policy": nvfp4_expansion_policy(), "row_offsets": list(offsets),
            "shape_nk": [n, k], "schedule_digest": digest,
            "graph_digest": _digest(graph), "schedule_ir_digest": _digest(schedule),
            "tile_ir_digest": _digest(tile),
            "ownership": "private_outputs_distinct_readonly_inputs",
        })


@dataclass(frozen=True)
class NVFP4ExpansionPackage:
    graph_ir: str
    schedule_ir: str
    native: ROCMNativePackage

    def validate(self):
        native = self.native
        image, desc = native.image, native.descriptor
        desc.validate_image(image)
        if (image.target != "rocm_gfx1201" or image.architecture != "gfx1201"
                or image.binary_format != "hsaco" or desc.abi_id != NVFP4_EXPAND_ABI
                or image.target_ir_digest != _digest(native.target_ir)):
            raise ValueError("NVFP4 expansion native image/ABI/Target contract differs")
        info = _contract(native.tile_ir, native.target_ir)
        if re.findall(r'\bhash = "([0-9a-f]{64})"', self.schedule_ir) != [info[-2]]:
            raise ValueError("NVFP4 expansion Schedule record digest differs")
        expected = _descriptor(image, info, self.graph_ir, self.schedule_ir, native.tile_ir)
        if desc != expected:
            raise ValueError("NVFP4 expansion descriptor differs from native contract")


def package_nvfp4_expansion(graph: str | GraphIRModule):
    source = graph if isinstance(graph, str) else graph.to_mlir(
        target="rocm_gfx1201", canonical=True)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("NVFP4 expansion requires matching tessera-opt")
    schedule = run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    target, backend, binary, compiler, toolchain, libraries, state = _compile_native_tile_ir(
        tile, directive="tessera_rocm.nvfp4_expand", family="quant_fp", architecture="gfx1201")
    info = _contract(tile, target)
    image = NativeImageArtifact(
        target="rocm_gfx1201", architecture="gfx1201", pipeline_name="tessera-lower-to-rocm",
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=_digest(target), binary_format="hsaco", payload=binary,
        entry_points=(NativeEntryPoint(info[-1], NVFP4_EXPAND_ABI),),
        device_libraries=libraries, compile_state=state)
    desc = _descriptor(image, info, source, schedule, tile)
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
