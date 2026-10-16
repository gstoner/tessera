"""Checked typed saved-LSE forward descriptor; no tensor computation."""
from __future__ import annotations

import math
import re
import struct
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .native_artifact import LaunchDescriptor


def validate_typed_saved_lse_descriptor(descriptor: LaunchDescriptor) -> tuple[str, str]:
    from .nvidia_native import (
        SM120_ATTN_LSE_TYPED_ABI, SM120_ATTN_LSE_BIAS_TYPED_ABI,
        SM120_ATTN_LSE_BCAST_TYPED_ABI, _checkpoint_identity,
    )
    abi = descriptor.abi_id
    if abi not in {SM120_ATTN_LSE_TYPED_ABI, SM120_ATTN_LSE_BIAS_TYPED_ABI,
                   SM120_ATTN_LSE_BCAST_TYPED_ABI}:
        raise ValueError("typed saved-LSE descriptor requires its forward ABI")
    bias = abi != SM120_ATTN_LSE_TYPED_ABI
    broadcast = abi == SM120_ATTN_LSE_BCAST_TYPED_ABI
    policy = descriptor.provenance
    storage, output = policy.get("storage"), policy.get("output")
    if (not isinstance(storage, str) or storage not in {"f16", "bf16"} or
            not isinstance(output, str) or output not in {storage, "f32"} or
            policy.get("accum") != "f32" or policy.get("row_lse") != "f32[B,Hq,Sq]" or
            policy.get("checkpoint_role") != "forward_save" or policy.get("lse_checkpoint") != "saved" or
            policy.get("bias") is not bias):
        raise ValueError("typed saved-LSE storage or result roles disagree")
    buffers = sorted(descriptor.buffers, key=lambda value: value.ordinal)
    dtype = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}
    expected = [(dtype[storage], 4, "input")] * 3
    if bias:
        expected += [("fp32", 4, "input")]
    expected += [(dtype[output], 4, "output"), ("fp32", 3, "output")]
    if len(buffers) != len(expected) or len({value.name for value in buffers}) != len(buffers):
        raise ValueError("typed saved-LSE buffer roles disagree")
    for index, (binding, (element, rank, direction)) in enumerate(zip(buffers, expected, strict=True)):
        if (binding.ordinal != index or binding.dtype != element or binding.rank != rank or
                binding.direction != direction or binding.layout != "row_major" or
                binding.alignment < (4 if element == "fp32" else 2)):
            raise ValueError("typed saved-LSE physical buffer storage disagrees")
    scalar_names: tuple[str, ...] = ("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv")
    if broadcast:
        scalar_names += ("BiasB", "BiasH", "BiasQ", "BiasK")
    scalars = sorted(descriptor.scalars, key=lambda value: value.ordinal)
    if (tuple(value.name for value in scalars) != scalar_names or any(
            value.ordinal != len(buffers)+index or value.dtype != "int64"
            for index, value in enumerate(scalars))):
        raise ValueError("typed saved-LSE scalar ABI disagrees")
    digest = policy.get("schedule_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("typed saved-LSE Schedule identity is missing")
    suffix = "_" if output == "f32" else "_out_" + output + "_"
    if descriptor.entry_symbol != "tessera_tile_attention_lse_" + storage + suffix + digest[:10]:
        raise ValueError("typed saved-LSE entry storage disagrees")
    shape = policy.get("shape")
    bounds = policy.get("shape_bounds", ())
    physical_bias = policy.get("bias_shape", ())
    if (not isinstance(shape, (list, tuple)) or len(shape) != 7 or
            any(type(value) is not int for value in shape) or
            not isinstance(bounds, (list, tuple)) or any(type(value) is not int for value in bounds) or
            not isinstance(physical_bias, (list, tuple)) or
            any(type(value) is not int for value in physical_bias) or
            bool(physical_bias) != broadcast):
        raise ValueError("typed saved-LSE shape and physical bias roles disagree")
    scale, causal = policy.get("scale"), policy.get("causal")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or
            not math.isfinite(scale) or scale <= 0 or type(causal) is not bool):
        raise ValueError("typed saved-LSE numeric policy disagrees")
    try:
        scale_f32 = struct.unpack("!f", struct.pack("!f", scale))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("typed saved-LSE scale is not finite positive f32") from error
    if not math.isfinite(scale_f32) or scale_f32 <= 0:
        raise ValueError("typed saved-LSE scale is not finite positive f32")
    identity = _checkpoint_identity(
        cast(tuple[int, ...], tuple(shape)), float(scale), causal,
        bias=bias, bias_shape=cast(tuple[int, ...], tuple(physical_bias)),
        shape_bounds=cast(tuple[int, ...], tuple(bounds)), storage=storage, output_storage=output)
    if policy.get("checkpoint_contract") != identity:
        raise ValueError("typed saved-LSE checkpoint identity disagrees")
    return storage, output

