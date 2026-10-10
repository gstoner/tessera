"""Frontend batch intent for compiler-owned static scaled-matmul packages.

Only semantic tensor types and batching intent are projected here. Native
Graph verification, Schedule derivation, Tile lowering and the checked ABI
own geometry, arithmetic and the single launch.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

from .graph_ir import GraphIRModule, tensor_ir_type
from .nvfp4_tensor import NVFP4Tensor


def _storage_dtype(value):
    # Tuple trace specs do not pass through ndarray dtype normalization.
    return {"float8_e4m3fn": "fp8_e4m3", "float8_e5m2": "fp8_e5m2"}.get(
        str(value.dtype), str(value.dtype))


def mixed_batch_policies(owner):
    policies = getattr(owner, "_frontend_batch_policies", ())
    return bool(policies and any(policy != policies[0] for policy in policies[1:]))


def normalize_mixed_batch_inputs(values, policies):
    """Insert missing map levels as aliasing singleton frontend views."""
    import numpy as np
    if not 4 <= len(values) <= 128 or not policies or any(len(p) != len(values) for p in policies):
        raise ValueError("native mixed maps require four operands and explicit level policies")
    extents = [None] * len(policies)
    shapes = []
    for role, value in enumerate(values):
        mapped = [p[role] is not None for p in policies]
        shape = tuple(value.shape)
        if len(shape) != sum(mapped) + 2:
            raise ValueError("native mixed map operand rank differs from its level policy")
        cursor = 0
        prefix = []
        for level, active in enumerate(mapped):
            if active:
                size = shape[cursor]
                cursor += 1
                if size <= 0 or (extents[level] is not None and extents[level] != size):
                    raise ValueError("native mixed map batch extents differ at one map level")
                extents[level] = size
                prefix.append(size)
            else:
                prefix.append(1)
        shapes.append(tuple(prefix) + shape[cursor:] if any(mapped) else shape)
    if any(extent is None for extent in extents):
        raise ValueError("native mixed map must map an operand at every level")
    result = []
    for value, shape in zip(values, shapes, strict=True):
        if not isinstance(value, np.ndarray):
            raise TypeError("native mixed maps require ndarray storage")
        view = value.reshape(shape)
        if not np.shares_memory(value, view):
            raise ValueError("native mixed map singleton projection must preserve storage")
        result.append(view)
    return result


def batch_specs(values: Sequence[Any], axes: Sequence[int | None], *, depth: int = 1, broadcast_prefix: bool = False):
    if type(depth) is not int or depth <= 0:
        raise ValueError("native scaled vmap requires a positive leading map depth")
    if not 4 <= len(values) <= 128 or len(axes) != len(values):
        raise ValueError("native NVFP4 vmap requires four operand axes or a composed scale frame")
    if any(axis is not None and (type(axis) is not int or axis != 0) for axis in axes):
        raise ValueError("native scaled vmap requires leading integer axes")
    coupled = tuple(axes) in {(0, None, 0, None), (0, 0, 0, 0), (None, 0, None, 0)}
    if not coupled and (
            not any(axis is not None for axis in axes) or
            any(axis is not None and (type(axis) is not int or axis != 0) for axis in axes) or
            any(isinstance(value, NVFP4Tensor) for value in values)):
        raise ValueError("native NVFP4 vmap requires leading matrix/scale batch axes")
    sizes = []
    specs = []
    for value, axis in zip(values, axes, strict=True):
        if isinstance(value, NVFP4Tensor):
            value.validate()
        shape = tuple(value.shape)
        if axis is not None:
            if len(shape) != depth + 2 or any(v <= 0 for v in shape[:depth]):
                raise ValueError("native scaled vmap requires positive leading batch extents")
            sizes.append(shape[:depth])
            shape = shape[depth:]
        elif len(shape) != 2:
            raise ValueError("native NVFP4 vmap requires rank-two shared operands")
        specs.append((shape, _storage_dtype(value)))
    if not broadcast_prefix and len(set(sizes)) != 1:
        raise ValueError("native NVFP4 vmap batch extents differ")
    return tuple(specs)


def project_batch(module: GraphIRModule, values: Sequence[Any], axes: Sequence[int | None], *, depth: int = 1, scale_transpose: bool = False, broadcast_prefix: bool = False) -> GraphIRModule:
    """Project frontend vmap intent without mutating the scalar trace."""
    batch_specs(values, axes, depth=depth, broadcast_prefix=broadcast_prefix)
    result = copy.deepcopy(module)
    if len(result.functions) != 1:
        raise ValueError("native NVFP4 vmap requires one semantic function")
    function = result.functions[0]
    if len(function.body) > 1:
        return _project_composed_batch(result, values, axes, depth=depth,
                                       scale_transpose=scale_transpose,
                                       broadcast_prefix=broadcast_prefix)
    if len(function.body) != 1 or len(function.args) != 4:
        raise ValueError("native NVFP4 vmap requires one scaled-matmul producer")
    op = function.body[0]
    from .rocm_typed_scaled_native import requests_typed_scaled
    typed = requests_typed_scaled(result)
    if (op.op_name != "tessera.scaled_matmul"
            or (not typed and op.kwargs.get("physical_contract") != "nvidia_sm120_nvfp4_blockscale_v1")
            or op.kwargs.get("batching") not in {None, "none"}
            or [name.removeprefix("%") for name in op.operands] != [arg.name for arg in function.args]
            or [name.removeprefix("%") for name in function.return_values] != op.result_names
            or len(function.result_types) != 1):
        raise ValueError("native NVFP4 vmap requires an unbatched direct scaled product")
    output = function.result_types[0]
    if output.rank != 2:
        raise ValueError("native NVFP4 vmap scalar result must have rank two")
    for arg, value in zip(function.args, values, strict=True):
        arg.ir_type = tensor_ir_type(tuple(value.shape), _storage_dtype(value))
    op.operand_types = [str(arg.ir_type) for arg in function.args]
    if broadcast_prefix:
        import numpy as np
        batch = np.broadcast_shapes(*(value.shape[:depth] for value, axis in zip(values, axes, strict=True) if axis is not None))
    else:
        batch = values[next(i for i, axis in enumerate(axes) if axis is not None)].shape[:depth]
    output = tensor_ir_type((*batch, *output.shape), output.dtype)
    op.inferred_type = output
    op.inferred_types = (output,)
    op.result_type = str(output)
    function.result_types = [output]
    coupled = tuple(axes) in {(0, None, 0, None), (0, 0, 0, 0), (None, 0, None, 0)}
    layout=op.kwargs.get("scale_layout",{})
    block=layout.get("block") if isinstance(layout,dict) else None
    k_axis=-2 if op.kwargs.get("transposeA",False) else -1
    partial=typed and isinstance(block,(list,tuple)) and len(block)==2 and type(block[1]) is int and block[1]>0 and int(function.args[0].ir_type.shape[k_axis])%block[1]!=0
    if broadcast_prefix or (typed and (op.kwargs.get("transposeA",False) or partial)):
        coupled = False
    op.kwargs["batching"] = (("shared_lhs" if axes[0] is None else
                             "independent_rhs" if axes[1] == 0 else "shared_rhs_rows")
                            if coupled else "broadcast")
    if typed:
        from .rocm_typed_scaled_native import supports_typed_scaled, supports_scale_transpose
        admission = supports_scale_transpose if scale_transpose else supports_typed_scaled
        if not admission(result):
            raise ValueError("native typed scaled vmap requires the exact matrix/scale contract")
    return result



def _project_composed_batch(module, values, axes, *, depth, scale_transpose, broadcast_prefix):
    """Project semantic batch types per product; native MLIR owns arithmetic."""
    from .rocm_typed_scaled_native import requests_composed_typed_scaled, supports_composed_scale_jvp
    if not requests_composed_typed_scaled(module):raise ValueError("native composed maps require typed scaled products and sums")
    function=module.functions[0]
    if len(function.args)!=len(values) or len(function.result_types)!=1 or function.result_types[0].rank!=2:
        raise ValueError("native composed maps require one scalar matrix result")
    import numpy as np
    prefixes=[tuple(value.shape[:depth]) for value,axis in zip(values,axes,strict=True) if axis is not None]
    if not prefixes:raise ValueError("native composed maps require a mapped operand")
    batch=np.broadcast_shapes(*prefixes) if broadcast_prefix else prefixes[0]
    output=tensor_ir_type((*batch,*function.result_types[0].shape),function.result_types[0].dtype)
    arguments={arg.name:(i,arg) for i,arg in enumerate(function.args)}
    source_types={arg.name:copy.deepcopy(arg) for arg in function.args}
    scale_roles=set()
    for op in function.body:
        if op.op_name=="tessera.scaled_matmul":
            if any(value.removeprefix("%") not in arguments for value in op.operands):
                raise ValueError("native composed maps require explicit product inputs")
            positions=[arguments[value.removeprefix("%")][0] for value in op.operands]
            member=copy.deepcopy(module);fn=member.functions[0]
            fn.args=[copy.deepcopy(source_types[value.removeprefix("%")]) for value in op.operands]
            fn.body=[copy.deepcopy(op)];fn.result_types=[copy.deepcopy(op.inferred_type)]
            fn.return_values=["%"+op.result_names[0]]
            projected=project_batch(member,[values[i] for i in positions],[axes[i] for i in positions],
                depth=depth,scale_transpose=scale_transpose,broadcast_prefix=broadcast_prefix)
            child=projected.functions[0].body[0]
            if str(child.inferred_type)!=str(output):
                raise ValueError("native composed map products must carry the same result batch")
            op.kwargs=copy.deepcopy(child.kwargs);op.operand_types=list(child.operand_types)
            scale_roles.update(positions[2:])
        elif op.op_name=="tessera.add":
            op.operand_types=[str(output)]*len(op.operands)
        else:
            raise ValueError("native composed maps require native product/sum SSA")
        op.inferred_type=output;op.inferred_types=(output,);op.result_type=str(output)
    for arg,value in zip(function.args,values,strict=True):
        arg.ir_type=tensor_ir_type(tuple(value.shape),_storage_dtype(value))
    function.result_types=[output]
    if not supports_composed_scale_jvp(module,tuple(sorted(scale_roles))):
        raise ValueError("native composed map requires the exact scale-product semantic contract")
    return module


def native_nvfp4_vmap(fn, in_axes, out_axes):
    return _native_scaled_vmap(fn, in_axes, out_axes, target="nvidia_sm120")


def native_typed_scaled_vmap(fn, in_axes, out_axes):
    """Project typed FP8/MXFP8 map intent into a compiler-owned batch."""
    return _native_scaled_vmap(fn, in_axes, out_axes, target="rocm_gfx1201")


def _native_scaled_vmap(fn, in_axes, out_axes, *, target):
    """Create an independent JIT owner; the scalar owner's caches stay intact."""
    from .jit import JitFn
    from .matmul_pipeline import normalize_target_kind
    if in_axes is None or (isinstance(in_axes, (tuple, list))
                           and len(in_axes) == len(fn.arg_names)
                           and all(axis is None for axis in in_axes)):
        return fn
    request = fn.differentiation_request
    from .rocm_typed_scaled_native import requests_composed_typed_scaled
    composed = requests_composed_typed_scaled(fn.graph_ir)
    if len(fn.graph_ir.functions)!=1:
        raise ValueError("native scaled vmap requires one semantic function")
    arguments=fn.graph_ir.functions[0].args
    scale_names={value.removeprefix("%") for op in fn.graph_ir.functions[0].body
                 if op.op_name=="tessera.scaled_matmul" for value in op.operands[2:]}
    scale_indices={i for i,arg in enumerate(arguments) if arg.name in scale_names and arg.ir_type.dtype=="fp32"}
    forward_scales = (
        target == "rocm_gfx1201" and request is not None
        and request.mode in {"forward", "reverse"} and request.wrt_indices
        and all(index in scale_indices for index in request.wrt_indices)
        and len(fn.graph_ir.functions) == 1
        and (len(fn.graph_ir.functions[0].body) == 1 or composed)
        and all(op.kwargs.get("scale_layout", {}).get("format") == "fp32"
                for op in fn.graph_ir.functions[0].body if op.op_name=="tessera.scaled_matmul")
    )
    if (normalize_target_kind(fn.target) != target
            or (request is not None and not forward_scales)
            or getattr(fn, "_bounded_lhs", None) is not None):
        raise ValueError(f"native scaled vmap requires a primal or admitted scale-JVP {target} JIT")
    if type(out_axes) is not int or out_axes != 0:
        raise ValueError("native NVFP4 vmap currently requires out_axes=0")
    if isinstance(in_axes, bool):
        raise ValueError("native NVFP4 vmap axes must be integers or None")
    axes = (in_axes,) * len(arguments) if type(in_axes) is int else tuple(in_axes) if in_axes is not None else (None,) * len(arguments)
    if any(axis is not None and type(axis) is not int for axis in axes):
        raise ValueError("native NVFP4 vmap axes must be integers or None")
    coupled = axes in {(0, None, 0, None), (0, 0, 0, 0), (None, 0, None, 0)}
    if not coupled and (
            len(axes) != len(arguments) or not any(axis is not None for axis in axes) or
            any(axis not in (None, 0) for axis in axes) or
            target != "rocm_gfx1201" or
            (request is not None and not forward_scales)):
        raise ValueError("native NVFP4 vmap requires leading matrix/scale batch axes")
    parent_axes = fn._frontend_batch_axes
    parent_depth = getattr(fn, "_frontend_batch_depth", 0) if parent_axes is not None else 0
    parent_policies = getattr(fn, "_frontend_batch_policies", ()) or ((parent_axes,) * parent_depth if parent_axes is not None else ())
    policies = (axes, *parent_policies)
    if parent_axes is not None and target != "rocm_gfx1201" and (
            not coupled or any(policy != axes for policy in parent_policies)):
        raise ValueError("native NVFP4 nested maps require matching coupled leading policies")
    depth = parent_depth + 1
    owner = JitFn(fn._fn, copy.deepcopy(fn._legacy_graph_ir or fn.graph_ir),
                  fn.inferred_effect, copy.deepcopy(fn.constraints),
                  deterministic=fn.deterministic, seed=fn.seed, target=fn.target,
                  source_origin=fn.source_origin, source_text=fn._frontend_source_text,
                  native_required=True,
                  differentiation_request=copy.deepcopy(request))
    # A leading map axis must not make scalar annotations disappear at the
    # ordinary call-time constraint gate. Keep M/N/K/S on their original
    # logical dimensions, with a fresh batch symbol shared by mapped inputs.
    arguments = copy.deepcopy(fn._constraint_ir_args)
    symbols = {name for arg in arguments for name in arg.dim_names}
    symbols.update(name for constraint in fn.constraints.constraints for name in constraint.dim_names())
    batch_symbol = "__tessera_vmap_batch"
    while batch_symbol in symbols:
        batch_symbol += "_"
    for role, (arg, axis) in enumerate(zip(arguments, axes, strict=True)):
        argument_depth = sum(policy[role] is not None for policy in parent_policies)
        if axis is None:
            continue
        if arg.dim_names:
            if len(arg.dim_names) != argument_depth + 2:
                raise ValueError("native scaled vmap annotations must retain each leading map")
            arg.dim_names = (batch_symbol, *arg.dim_names)
        if arg.ir_type.rank == argument_depth + 2:
            arg.ir_type = tensor_ir_type((batch_symbol, *arg.ir_type.shape),
                                         arg.ir_type.dtype, layout=arg.ir_type.layout)
    owner._constraint_ir_args = arguments
    owner._frontend_batch_policies = policies
    owner._frontend_batch_axes = tuple(0 if any(policy[role] is not None for policy in policies) else None for role in range(len(arguments)))
    owner._frontend_batch_depth = depth
    return owner


def certify_typed_batch_frontends(owner, values, *, rtol, atol):
    """Reference-only certification; never used as an execution backend."""
    import numpy as np
    from .frontend_authority import certify_frontends
    from .graph_ir import specialize_module_from_values
    from .reference_typed_scaled_matmul import reference_typed_scaled_matmul

    raw_values = values
    mixed = mixed_batch_policies(owner)
    if mixed:
        values = normalize_mixed_batch_inputs(values, owner._frontend_batch_policies)
    axes = owner._frontend_batch_axes
    depth = owner._frontend_batch_depth
    batch_specs(values, axes, depth=depth, broadcast_prefix=mixed)
    tracer_module, _ = owner._trace_frontend_capture(tuple(raw_values), {})
    from .rocm_typed_scaled_native import supports_typed_scaled, supports_scale_transpose
    request = owner.differentiation_request
    scale_transpose = request is not None and request.mode == "reverse"
    admission = supports_scale_transpose if scale_transpose else supports_typed_scaled
    from .rocm_typed_scaled_native import requests_composed_typed_scaled, supports_composed_scale_jvp
    composed=requests_composed_typed_scaled(tracer_module)
    scale_roles=tuple(i for i,arg in enumerate(tracer_module.functions[0].args) if arg.ir_type.dtype=="fp32")
    if not (supports_composed_scale_jvp(tracer_module,scale_roles) if composed else admission(tracer_module)):
        raise ValueError("mapped frontend differential requires exact typed scaled Graph")
    batch_shape = np.broadcast_shapes(*(value.shape[:depth] for value, axis in zip(values, axes, strict=True) if axis is not None)) if mixed else values[next(i for i, axis in enumerate(axes) if axis is not None)].shape[:depth]
    # This loop is the explicit independent eager map oracle, evaluated once
    # per certificate signature. Native execution uses the batched Graph.
    outputs = []
    first = None
    for plane in np.ndindex(batch_shape):
        scalar_values = tuple(value[tuple(0 if size == 1 else index for size, index in zip(value.shape[:depth], plane, strict=True))] if axis is not None else value
                              for value, axis in zip(values, axes, strict=True))
        if first is None:
            first = scalar_values
        outputs.append(owner._fn(*scalar_values))
    if first is None:
        raise ValueError("native batch certification requires a nonempty batch")
    legacy = specialize_module_from_values(
        owner._ensure_legacy_graph_ir(), dict(zip(owner.arg_names, first, strict=True)))
    legacy = project_batch(legacy, values, axes, depth=depth, scale_transpose=scale_transpose, broadcast_prefix=mixed)
    op = tracer_module.functions[0].body[0]
    if composed:
        semantic={("%"+arg.name):value for arg,value in zip(tracer_module.functions[0].args,values,strict=True)}
        for operation in tracer_module.functions[0].body:
            operands=[semantic[name] for name in operation.operands]
            computed=(reference_typed_scaled_matmul(*operands,**operation.kwargs)
                      if operation.op_name=="tessera.scaled_matmul" else operands[0]+operands[1])
            semantic["%"+operation.result_names[0]]=computed
        oracle=semantic[tracer_module.functions[0].return_values[0]]
    else:oracle = reference_typed_scaled_matmul(*values, **op.kwargs)
    return certify_frontends(
        legacy_module=legacy, tracer_module=tracer_module,
        legacy_outputs=(np.stack(outputs).reshape(*batch_shape, *outputs[0].shape),), tracer_outputs=(oracle,),
        rtol=rtol, atol=atol)
