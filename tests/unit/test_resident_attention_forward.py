"""Host-independent ordinary resident forward metadata guards."""
import numpy as np
import pytest

from tessera.compiler.prepared_attention_forward import input_signature
from tests.unit.test_ordered_resident_tensor_dag import Buffer


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_forward_signature_reads_metadata_only(dtype):
    value = Buffer((1, 2, 3, 4), dtype)
    assert input_signature((value,)) == ((str(np.dtype(dtype)), (1, 2, 3, 4)),)


@pytest.mark.parametrize("field,value", [
    ("typestr", "<i4"), ("shape", (1, 2, 3)), ("strides", (100, 60, 20, 4)),
    ("stream", 0), ("data", (0, True)), ("version", 2),
])
def test_forward_invalid_resident_metadata_fails_before_native_work(field, value):
    root = Buffer((1, 2, 3, 4), np.float32)
    root.interface[field] = value
    with pytest.raises(ValueError):
        input_signature((root,))


def test_forward_mixed_roots_rejected_before_native_work():
    with pytest.raises(ValueError, match="all resident"):
        input_signature((Buffer((1, 2, 3, 4), np.float32), np.ones((1, 2, 3, 4), np.float32)))
