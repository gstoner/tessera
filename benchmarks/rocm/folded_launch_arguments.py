"""Physical argument presentation for checked folded benchmark packages."""
import ctypes as ct
from tessera.compiler.native_artifact import HAND_EMITTED_HIP_PRODUCER

def folded_launch_values(package, pointers, arrays, shape):
    """Keep benchmark launches faithful to the image's checked physical ABI."""
    if len(pointers) != 5 or len(arrays) != 5:
        raise ValueError("folded launch requires five buffers")
    provenance = package.descriptor.provenance
    native = provenance.get("native_compiler_owned") is True
    layout = provenance.get("kernel_argument_layout")
    if native:
        if layout != "expanded_memref" or package.image.pipeline_name != "tessera-lower-to-rocm":
            raise ValueError("native folded benchmark requires expanded memref arguments")
        values = []
        for pointer, array in zip(pointers, arrays):
            values.extend([ct.c_void_p(pointer.value), ct.c_void_p(pointer.value),
                           ct.c_int64(0), ct.c_int64(array.size), ct.c_int64(1)])
    else:
        if layout is not None or package.image.pipeline_name != HAND_EMITTED_HIP_PRODUCER:
            raise ValueError("legacy folded benchmark requires its HIP pointer contract")
        values = [ct.c_void_p(pointer.value) for pointer in pointers]
    return values + [ct.c_int64(value) for value in shape]
