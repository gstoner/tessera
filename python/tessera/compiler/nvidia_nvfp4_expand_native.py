"""Checked compiler-owned sm_120 FP4 expansion packages.

The NVIDIA sibling of ``rocm_nvfp4_expand_native`` (SM120-FP4-EXPAND-NATIVE-20261103).
Python serializes the same semantic Graph IR and validates ABI metadata; the
native Graph/Schedule/Tile passes and ``LowerTileToNVIDIA`` own all numerical
lowering. Static images only, for both the NVFP4 and the MXFP4 source.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ResourceRecord, ShapeGuard,
)
from .nvfp4_expansion import (
    checked_expansion_inputs, expansion_source, mxfp4_expansion_policy, nvfp4_expansion_policy,
)
from .nvidia_native import NVIDIANativePackage
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

NVFP4_EXPAND_SM120_ABI = "tessera.nvidia.sm120.nvfp4_expand.five_pointers.static.v1"
#: Same image from an MXFP4 source: E8M0 bytes [N,K/32] in place of E4M3 [N,K/16].
MXFP4_EXPAND_SM120_ABI = "tessera.nvidia.sm120.nvfp4_expand.five_pointers.mxfp4_static.v1"
_ABIS = (NVFP4_EXPAND_SM120_ABI, MXFP4_EXPAND_SM120_ABI)
_SOURCES = ("nvfp4_e2m1_e4m3_k16", "mxfp4_e2m1_e8m0_k32")
_WORKGROUP = 256


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _field(pattern, text):
    found = re.findall(pattern, text)
    if len(found) != 1:
        raise ValueError("sm_120 FP4 expansion Tile contract field is missing or repeated")
    return found[0]


def _contract(tile):
    """(names, n, k, offsets, digest, entry, mx) of the one checked Tile kernel."""
    if tile.count("tile.nvfp4_expand_kernel") != 1:
        raise ValueError("sm_120 FP4 expansion requires its isolated native producer")
    entry, arguments = _field(r"llvm\.func @(tessera_nvfp4_expand_[0-9a-f]{16})\(([^)]*)\)", tile)
    if (_field(r'\barch = "([^"]+)"', tile) != "sm_120"
            or _field(r'\btarget = "([^"]+)"', tile) != "nvidia_sm120"):
        raise ValueError("sm_120 FP4 expansion Tile names another architecture")
    if ([a.split(":", 1)[-1].strip() for a in arguments.split(",")] != ["!llvm.ptr"] * 5
            or "nvvm.kernel" not in tile):
        raise ValueError("sm_120 FP4 expansion Tile entry is not a five-pointer nvvm.kernel")
    source = _field(r'source_format = "([a-z0-9_]+)"', tile)
    if source not in _SOURCES:
        raise ValueError("sm_120 FP4 expansion Tile names no admitted source format")
    mx = source == _SOURCES[1]
    names = tuple(json.loads(_field(r"bindings = (\[[^\]]+\])", tile)))
    n = int(_field(r"\bn = (\d+) : i64", tile))
    k = int(_field(r"\bk = (\d+) : i64", tile))
    offsets = tuple(json.loads(_field(r"row_offsets = (\[[^\]]+\])", tile)))
    digest = _field(r'artifact_hash = "([0-9a-f]{64})"', tile)
    if (len(names) != 5 or len(set(names)) != 5
            or any(not isinstance(name, str) or not name for name in names)
            or n <= 0 or k <= 0 or k % (32 if mx else 16) or n > (2**63 - 1) // k
            or len(offsets) < 2 or any(type(x) is not int for x in offsets)
            or offsets[0] != 0 or offsets[-1] != n
            or any(a >= b for a, b in zip(offsets, offsets[1:]))
            or entry != "tessera_nvfp4_expand_" + digest[:16]):
        raise ValueError("sm_120 FP4 expansion shape/binding/projection contract changed")
    return names, n, k, offsets, digest, entry, mx


def _descriptor(image, info, graph_digest, schedule_ir_digest, tile_ir_digest):
    names, n, k, offsets, digest, entry, mx = info
    shapes = ((n, k // 2), (n, k // 32) if mx else (n, k // 16), (len(offsets) - 1,),
              (n, k), (k // 16, n))
    dtypes = ("uint8", "uint8" if mx else "fp8_e4m3", "fp64", "fp8_e4m3", "fp32")
    return LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry,
        abi_id=MXFP4_EXPAND_SM120_ABI if mx else NVFP4_EXPAND_SM120_ABI,
        buffers=tuple(BufferBinding(
            i, name, "input" if i < 3 else "output", dtypes[i], len(shapes[i]),
            "row_major", 8 if i == 2 else 4 if i == 4 else 1)
            for i, name in enumerate(names)),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", dim)
                           for name, shape in zip(names, shapes)
                           for axis, dim in enumerate(shape)),
        geometry=LaunchGeometry(grid=((n * (k // 16) + _WORKGROUP - 1) // _WORKGROUP, 1, 1),
                                workgroup=(_WORKGROUP, 1, 1)),
        ordering=OrderingSemantics(ordered_submission=True, residency="none",
                                  synchronization=("completion",)),
        provenance={
            "work_item": "ROCM-NVFP4-INGEST-1",
            "sync_key": "SM120-FP4-EXPAND-NATIVE-20261103",
            "route": "GraphIR->ScheduleIR->TileIR->LowerTileToNVIDIA->LLVM->PTX",
            "numeric_policy": mxfp4_expansion_policy() if mx else nvfp4_expansion_policy(),
            "row_offsets": list(offsets), "shape_nk": [n, k], "schedule_digest": digest,
            "graph_digest": graph_digest, "schedule_ir_digest": schedule_ir_digest,
            "tile_ir_digest": tile_ir_digest,
            "ownership": "private_outputs_distinct_readonly_inputs",
        })


@dataclass(frozen=True)
class SM120NVFP4ExpansionPackage:
    graph_ir: str
    schedule_ir: str
    native: NVIDIANativePackage

    @property
    def mxfp4(self):
        """Whether the checked source is MXFP4 (E8M0 K32 scales)."""
        return self.native.descriptor.abi_id == MXFP4_EXPAND_SM120_ABI

    def numeric_policy(self):
        return mxfp4_expansion_policy() if self.mxfp4 else nvfp4_expansion_policy()

    def contract(self):
        return _contract(self.native.tile_ir)

    def validate(self):
        native = self.native
        image, desc = native.image, native.descriptor
        desc.validate_image(image)
        info = self.contract()
        if (image.target != "nvidia_sm120" or image.architecture != "sm_120a"
                or image.binary_format != "ptx" or desc.abi_id not in _ABIS
                or image.target_ir_digest != _digest(native.target_ir)
                or self.mxfp4 != info[-1]
                or "tile.nvfp4_expand_kernel" in native.target_ir
                or f".visible .entry {info[5]}" not in native.backend_ir
                or image.payload != native.backend_ir.encode("ascii")):
            raise ValueError("sm_120 FP4 expansion native image/ABI/Target contract differs")
        if re.findall(r'\bhash = "([0-9a-f]{64})"', self.schedule_ir) != [info[4]]:
            raise ValueError("sm_120 FP4 expansion Schedule record digest differs")
        expected = _descriptor(image, info, _digest(self.graph_ir), _digest(self.schedule_ir),
                               _digest(native.tile_ir))
        if desc != expected:
            raise ValueError("sm_120 FP4 expansion descriptor differs from native contract")


def package_sm120_nvfp4_expansion(graph: str | GraphIRModule):
    """Package the static native sm_120 expansion of an ``nvidia_sm120`` Graph."""
    from .nvidia_native import _compile_tile_ir

    if isinstance(graph, GraphIRModule):
        attrs = graph.module_attrs
        if (json.loads(attrs.get("tessera.target", "null")) != "nvidia_sm120"
                or json.loads(attrs.get("tessera.arch", "null")) != "sm_120"):
            raise ValueError("sm_120 FP4 expansion requires an nvidia_sm120 / sm_120 Graph")
        source = graph.to_mlir(target="nvidia_sm120", canonical=True)
    elif isinstance(graph, str):
        source = graph
    else:
        raise TypeError("sm_120 FP4 expansion consumes a GraphIRModule or its MLIR text")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("sm_120 FP4 expansion requires matching tessera-opt")
    schedule = run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    info = _contract(tile)
    lowered, ptx, metrics, compiler, toolchain, libraries, state = _compile_tile_ir(tile, info[5])
    image = NativeImageArtifact(
        target="nvidia_sm120", architecture="sm_120a",
        pipeline_name="tessera-lower-to-nvidia-sm120",
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=_digest(lowered), binary_format="ptx", payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(info[5], MXFP4_EXPAND_SM120_ABI if info[-1]
                                       else NVFP4_EXPAND_SM120_ABI),),
        compile_state=state, device_libraries=libraries,
        resource_record=ResourceRecord(provenance="scheduled Tile IR; ptxas --arch=sm_120a -v",
                                       metrics=metrics))
    desc = _descriptor(image, info, _digest(source), _digest(schedule), _digest(tile))
    package = SM120NVFP4ExpansionPackage(
        source, schedule, NVIDIANativePackage(tile, lowered, ptx, image, desc))
    package.validate()
    return package


def execute_sm120_nvfp4_expansion(package, codes, scales, projection_globals):
    """Run a checked package on this host's sm_120 device; returns (values, scale_plane)."""
    import ml_dtypes
    import numpy as np

    from tessera import runtime

    package.validate()
    _, n, k, offsets, _, entry, mx = package.contract()
    # Checked before any device work: shape, finite scales, E8M0 code 255,
    # positive globals and representable rounded products.
    codes, scales, globals_, _ = checked_expansion_inputs(
        codes, scales, projection_globals, row_offsets=offsets,
        numeric_policy=package.numeric_policy())
    if codes.shape != (n, k // 2):
        raise ValueError("sm_120 FP4 expansion inputs differ from compiled shape")
    lib = runtime._load_nvidia_ptx_launch()
    if lib is None:
        raise RuntimeError("libtessera_nvidia_ptx_launch.so not loadable")
    if runtime._register_nvidia_ptx(lib, entry, package.native.backend_ir) != 0:
        raise RuntimeError(f"sm_120 FP4 expansion PTX registration failed for {entry}")
    inputs = [np.ascontiguousarray(codes),
              np.ascontiguousarray(scales).view(np.uint8),
              np.ascontiguousarray(globals_)]
    values = np.empty((n, k), dtype=np.uint8)
    plane = np.empty((k // 16, n), dtype=np.float32)
    arrays = inputs + [values, plane]
    buffers = (ctypes.c_void_p * 5)(*(a.ctypes.data for a in arrays))
    dims = (ctypes.c_int64 * 4)(n, k, 32 if mx else 16, len(offsets) - 1)
    rc = lib.tessera_nvidia_ptx_invoke(entry.encode(), buffers, 5, dims, 4)
    if rc:
        raise RuntimeError(f"sm_120 FP4 expansion invoke returned {runtime._nvidia_ptx_failure(lib, rc)}")
    return values.view(ml_dtypes.float8_e4m3fn), plane


def build_sm120_nvfp4_expansion_graph(n, k, row_offsets, *, numeric_policy):
    """The shared expansion Graph builder, targeted at ``nvidia_sm120``."""
    from .rocm_nvfp4_expand_native import build_nvfp4_expansion_graph

    expansion_source(numeric_policy)
    return build_nvfp4_expansion_graph(n, k, row_offsets, numeric_policy=numeric_policy,
                                       target="nvidia_sm120")
