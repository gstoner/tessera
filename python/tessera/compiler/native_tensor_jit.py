"""Transport-only signature and inspection helpers for native tensor programs."""
import numpy as np


def input_signature(values):
    """Describe compact physical roots without inspecting or copying their values."""
    if len(values) != 2:
        return None
    resident = any(hasattr(value, "__cuda_array_interface__") for value in values)
    if resident and not all(hasattr(value, "__cuda_array_interface__") for value in values):
        raise ValueError("native tensor JIT requires all resident or all host roots")
    if resident:
        if any(len(value.__cuda_array_interface__.get("shape", ())) != 2 for value in values):
            return None
        from .resident_nvidia_tensor import cuda_frontend_specs
        specs = cuda_frontend_specs(values, ranks=(2,))
    elif all(isinstance(value, np.ndarray) for value in values):
        specs = tuple((tuple(value.shape), value.dtype) for value in values)
    else:
        return None
    signature = []
    for value, (shape, dtype) in zip(values, specs, strict=True):
        if len(shape) != 2 or str(dtype) not in {"float16", "bfloat16"}:
            return None
        width = dtype.itemsize
        compact = (shape[1] * width, width)
        strides = (value.__cuda_array_interface__.get("strides") if resident else value.strides)
        if strides is not None and tuple(strides) != compact:
            return None
        signature.append((str(dtype), tuple(shape), compact))
    return resident, tuple(signature)


def lowering_artifacts(program):
    """Return the actual adjacent native stages, not fingerprints of leaf lists."""
    from .matmul_pipeline import LoweringArtifact
    stages = []
    for level, text, producer in (
        ("graph", program.source_ir, "frontend.tracer"),
        ("schedule", program.schedule_ir, "tessera-opt.tessera-graph-to-schedule"),
        ("tile", program.tile_ir, "tessera-opt.tessera-schedule-to-tile"),
        ("target", program.target_ir, "tessera-opt.tessera-lower-to-nvidia-sm120"),
    ):
        stages.append(LoweringArtifact(level, text, producer,
                                       stages[-1].output_digest if stages else None))
    return tuple(stages)
