"""Typed saved-LSE reverse role checks; no tensor arithmetic or compiler calls."""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .native_artifact import LaunchDescriptor


def _integer_policy(policy, key):
    values = policy.get(key, ())
    if not isinstance(values, (list, tuple)) or any(type(value) is not int for value in values):
        raise ValueError("typed checkpoint integer metadata differs: "+key)
    return tuple(values)


def _policy(descriptor):
    p = descriptor.provenance
    storage, output = p.get("storage"), p.get("output")
    if (storage not in ("f16", "bf16") or output not in (storage, "f32") or
            p.get("gradient_storage") != storage or p.get("accum") != "f32" or
            p.get("row_lse") != "f32[B,Hq,Sq]" or
            p.get("row_correction") != "probability_dot_v_f32_v1" or
            p.get("checkpoint_role") != "backward_load" or
            p.get("lse_checkpoint") != "saved" or
            p.get("mask_alignment") != "end_aligned_v1"):
        raise ValueError("typed backward storage and residual policy disagree")
    for key in ("bias", "bias_gradient", "lse_cotangent", "causal"):
        if type(p.get(key)) is not bool:
            raise ValueError("typed backward boolean policy disagrees: "+key)
    if p["bias_gradient"] and not p["bias"]:
        raise ValueError("typed backward bias gradient lacks a bias")
    scale = p.get("scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("typed backward scale requires finite positive f32")
    try:
        f32_scale = struct.unpack("!f", struct.pack("!f", scale))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("typed backward scale requires finite positive f32") from error
    if not math.isfinite(f32_scale) or f32_scale <= 0:
        raise ValueError("typed backward scale requires finite positive f32")
    if p.get("route") != "deterministic_direct" or p.get("deterministic") is not True or p.get("workspace_bytes") != 0:
        raise ValueError("typed backward requires its deterministic native route")
    return p


def validate_typed_backward_descriptor(descriptor: LaunchDescriptor) -> tuple[str, str]:
    from .nvidia_native import SM120_ATTN_BWD_LSE_TYPED_ABI, _checkpoint_identity
    from .attention_shape_contract import descriptor_attention_shapes
    if descriptor.abi_id != SM120_ATTN_BWD_LSE_TYPED_ABI:
        raise ValueError("typed backward requires its distinct ABI")
    p = _policy(descriptor)
    storage, output = p["storage"], p["output"]
    bias, bias_gradient, seeded = p["bias"], p["bias_gradient"], p["lse_cotangent"]
    compact = p.get("gradient_output") == "compact_v1"
    if p.get("gradient_output") not in ("complete_v1", "compact_v1"):
        raise ValueError("typed backward gradient output policy disagrees")
    activity = p.get("gradient_activity", ())
    if (not isinstance(activity, (list, tuple)) or (activity and (
            len(activity) != 3+int(bias_gradient) or not any(activity) or
            any(type(x) is not int or x not in (0, 1) for x in activity))) or
            (compact and not activity)):
        raise ValueError("typed backward gradient activity disagrees")
    roles = tuple(i for i in range(3+int(bias_gradient)) if not compact or activity[i])
    declared_roles = p.get("physical_gradient_roles")
    if not isinstance(declared_roles, (list, tuple)) or any(type(x) is not int for x in declared_roles) or tuple(declared_roles) != roles:
        raise ValueError("typed backward physical gradient roles disagree")
    expected_inactive = ("absent_v1" if compact else "zero_fill_v1") if activity else "none"
    if p.get("inactive_gradient") != expected_inactive:
        raise ValueError("typed backward inactive gradient policy disagrees")
    threads = p.get("gradient_block_threads")
    launch = p.get("gradient_launch")
    if (type(threads) is not int or threads not in ((64, 128) if compact else (128,)) or
            launch not in (("logical_v1", "packed_v1") if compact else ("logical_v1",))):
        raise ValueError("typed backward launch policy disagrees")
    dtype = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}
    expected = [(dtype[output], 4, "input")] + [(dtype[storage], 4, "input")]*3 + [(dtype[output], 4, "input")]
    if bias:
        expected += [("fp32", 4, "input")]
    expected += [("fp32", 3, "input")]
    if seeded:
        expected += [("fp32", 3, "input")]
    expected += [(dtype[storage] if role < 3 else "fp32", 4, "output") for role in roles]
    buffers = sorted(descriptor.buffers, key=lambda x: x.ordinal)
    if len(buffers) != len(expected) or len({x.name for x in buffers}) != len(buffers):
        raise ValueError("typed backward physical buffer count or names disagree")
    for index, (binding, (element, rank, direction)) in enumerate(zip(buffers, expected, strict=True)):
        if (binding.ordinal != index or binding.dtype != element or binding.rank != rank or
                binding.direction != direction or binding.layout != "row_major" or
                binding.alignment != (4 if element == "fp32" else 2)):
            raise ValueError("typed backward physical buffer storage disagrees")
    scalar_names = ("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv") + (
        ("BiasB", "BiasH", "BiasQ", "BiasK") if p.get("bias_shape") else ())
    scalars = sorted(descriptor.scalars, key=lambda x: x.ordinal)
    if (tuple(x.name for x in scalars) != scalar_names or any(
            x.ordinal != len(buffers)+i or x.dtype != "int64" for i, x in enumerate(scalars))):
        raise ValueError("typed backward scalar ABI disagrees")
    descriptor_attention_shapes(descriptor)
    if (descriptor.geometry.policy != f"sm120_attention_backward_lse_deterministic_direct_{threads}" or
            descriptor.workspace.bytes != 0 or descriptor.dynamic_local_memory_bytes or
            descriptor.dynamic_local_memory_expression is not None or
            not descriptor.ordering.ordered_submission or descriptor.ordering.residency != "none" or
            descriptor.ordering.synchronization != ("completion",)):
        raise ValueError("typed backward geometry or ownership policy disagrees")
    digest = p.get("schedule_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("typed backward Schedule digest is missing")
    prefix = "tessera_tile_attention_backward_lse_output_"
    if compact:
        mask = sum(1 << i for i, active in enumerate(activity) if active)
        prefix += f"compact_m{mask}_b{int(bias)}_g{int(bias_gradient)}_l{int(launch == 'logical_v1')}_t{threads}_"
    else:
        if bias_gradient:
            prefix += "bias_gradient_"
        if seeded:
            prefix += "cotangent_"
    entry = prefix+storage+"_out_"+output+"_"+digest[:10]+("_cotangent_" if seeded and compact else "")
    if descriptor.entry_symbol != entry:
        raise ValueError("typed backward entry differs from native storage/gradient policy")
    identity = _checkpoint_identity(_integer_policy(p, "shape"), p["scale"], p["causal"],
        bias=bias, bias_shape=_integer_policy(p, "bias_shape"), shape_bounds=_integer_policy(p, "shape_bounds"),
        storage=storage, output_storage=output)
    if p.get("checkpoint_contract") != identity:
        raise ValueError("typed backward checkpoint identity disagrees")
    return storage, output


def _native_dictionary(tile):
    match = re.search(r"tessera\.native_contract = ", tile)
    if match is None or match.end() >= len(tile) or tile[match.end()] != "{":
        raise ValueError("typed checkpoint native contract is missing")
    start = match.end()
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(tile)):
        c = tile[index]
        if escaped:
            escaped = False
        elif quoted and c == "\\":
            escaped = True
        elif c == '"':
            quoted = not quoted
        elif not quoted:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return tile[start:index+1]
    raise ValueError("typed checkpoint native dictionary is truncated")


def _verify_package(package, backward):
    from .native_artifact import LaunchDescriptor, NativeImageArtifact
    image = NativeImageArtifact.from_dict(package.image.to_dict())
    descriptor = LaunchDescriptor.from_dict(package.descriptor.to_dict())
    descriptor.validate_image(image)
    if image.target != "nvidia_sm120" or image.architecture != "sm_120a" or image.binary_format != "ptx":
        raise ValueError("typed checkpoint requires the SM120 PTX image")
    p = descriptor.provenance
    if (hashlib.sha256(package.target_ir.encode()).hexdigest() != image.target_ir_digest or
            hashlib.sha256(package.tile_ir.encode()).hexdigest() != p.get("tile_ir_digest")):
        raise ValueError("typed checkpoint retained IR digest disagrees")
    native = _native_dictionary(package.tile_ir)
    if (hashlib.sha256(native.encode()).hexdigest() != p.get("schedule_digest") or
            re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', package.tile_ir) != [p.get("schedule_digest")] or
            re.findall(r'llvm.func @([\w]+)\(', package.tile_ir) != [descriptor.entry_symbol]):
        raise ValueError("typed checkpoint native contract or entry digest disagrees")
    def string(key, default=None):
        found = re.search(r'(?<![\w.])'+re.escape(key)+r' = ("(?:[^"\\]|\\.)*")', native)
        return json.loads(found[1]) if found else default
    def array(key, default=()):
        found = re.search(r'(?<![\w.])'+re.escape(key)+r' = array<i64: ([^>]+)>', native)
        return tuple(map(int, found[1].split(","))) if found else default
    def names(key):
        found = re.search(r'(?<![\w.])'+re.escape(key)+r' = (\[[^\]]*\])', native)
        return tuple(json.loads(found[1])) if found else ()
    for key, expected in (("family", "attention_checkpoint_backward" if backward else "attention_checkpoint_forward"),
                          ("storage", p["storage"]), ("output_storage", p["output"]),
                          ("row_lse_storage", "f32"), ("accum", "f32"),
                          ("target", "nvidia_sm120"), ("arch", "sm_120"),
                          ("mask_alignment", "end_aligned_v1")):
        if string(key) != expected:
            raise ValueError("typed checkpoint native storage/policy differs: "+key)
    if backward and (string("gradient_storage") != p["gradient_storage"] or string("row_correction") != p["row_correction"]):
        raise ValueError("typed backward native gradient/residual storage differs")
    if array("shape") != _integer_policy(p, "shape") or array("shape_bounds") != _integer_policy(p, "shape_bounds"):
        raise ValueError("typed checkpoint native dimensions differ")
    if array("frontend_argument_indices") != _integer_policy(p, "frontend_argument_indices"):
        raise ValueError("typed checkpoint native frontend mapping differs")
    if array("bias_shape") != _integer_policy(p, "bias_shape"):
        raise ValueError("typed checkpoint physical bias differs")
    for key in ("causal", "bias", *(("lse_cotangent",) if backward else ())):
        found = re.search(r'(?<![\w.])'+re.escape(key)+r' = (true|false)', native)
        if (found is not None and found[1] == "true") is not p.get(key, False):
            raise ValueError("typed checkpoint native boolean differs: "+key)
    scale = re.search(r'(?<![\w.])scale = ([^ ,}]+) : f32', native)
    if scale is None:
        raise ValueError("typed checkpoint native scale is missing")
    actual = struct.unpack('>d', int(scale[1], 16).to_bytes(8, 'big'))[0] if scale[1].startswith('0x') else float(scale[1])
    if struct.pack('!f', actual) != struct.pack('!f', p['scale']):
        raise ValueError("typed checkpoint native scale differs")
    inputs = tuple(x.name for x in descriptor.buffers if x.direction == "input")
    outputs = tuple(x.name for x in descriptor.buffers if x.direction == "output")
    if names("arguments") != inputs or names("physical_results" if p.get("gradient_output") == "compact_v1" else "results") != outputs:
        raise ValueError("typed checkpoint native physical bindings differ")
    if backward:
        if len(names("results")) != 3+int(p["bias_gradient"] is True):
            raise ValueError("typed checkpoint native bias gradient role differs")
        if array("gradient_activity") != _integer_policy(p, "gradient_activity"):
            raise ValueError("typed checkpoint native gradient activity differs")
        for key, default in (("gradient_output", "complete_v1"), ("inactive_gradient", "none")):
            if string(key, default) != p[key]:
                raise ValueError("typed checkpoint native gradient policy differs: "+key)
        if p["gradient_output"] == "compact_v1":
            threads = re.search(r'gradient_block_threads = ([0-9]+) : i64', native)
            if string("gradient_launch") != p["gradient_launch"] or threads is None or int(threads[1]) != p["gradient_block_threads"]:
                raise ValueError("typed checkpoint native compact geometry differs")
    return descriptor


def checkpoint_pair_shapes(pair):
    """Validate the typed pair for the synchronous public owner, without tools."""
    from .typed_attention_checkpoint import validate_typed_saved_lse_descriptor
    from .attention_shape_contract import attention_dimensions, descriptor_attention_shapes
    from .nvidia_native import _checkpoint_identity
    storage, output = validate_typed_saved_lse_descriptor(pair.forward.descriptor)
    if validate_typed_backward_descriptor(pair.backward.descriptor) != (storage, output):
        raise ValueError("typed checkpoint producer and consumer storage differ")
    forward = _verify_package(pair.forward, False)
    backward = _verify_package(pair.backward, True)
    if validate_typed_backward_descriptor(backward) != (storage, output):
        raise ValueError("typed checkpoint producer and consumer storage differ")
    f, b = forward.provenance, backward.provenance
    identity = _checkpoint_identity(tuple(f['shape']), f['scale'], f['causal'],
        bias=f['bias'], bias_shape=tuple(f.get('bias_shape', ())), shape_bounds=tuple(f.get('shape_bounds', ())),
        storage=storage, output_storage=output)
    if pair.contract_digest != identity or b.get('checkpoint_contract') != identity:
        raise ValueError("typed checkpoint producer and consumer generations differ")
    fn = [x.name for x in forward.buffers]
    bn = [x.name for x in backward.buffers]
    biased = f['bias']
    if (fn[:3] != bn[1:4] or fn[3+int(biased)] != bn[4] or
            fn[4+int(biased)] != bn[5+int(biased)] or
            (biased and fn[3] != bn[5]) or
            f.get('frontend_argument_indices') != b.get('frontend_argument_indices')):
        raise ValueError("typed checkpoint saved/front-end roles differ")
    dims = attention_dimensions(f['shape'], f.get('shape_bounds', ()))
    descriptor_attention_shapes(forward)
    d, hq, hkv, sq, sk, width, value_width = dims
    return dims, ((d, hq, sq, width), (d, hkv, sk, width), (d, hkv, sk, value_width),
                  (d, hq, sq, value_width), (d, hq, sq))


def vjp_checkpoint_shapes(pair):
    if pair.forward.descriptor.provenance.get("storage") in ("f16", "bf16"):
        return checkpoint_pair_shapes(pair)
    from .resident_attention import checkpoint_shapes
    return checkpoint_shapes(pair)
