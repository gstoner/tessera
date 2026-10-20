"""Prepared bounded frames validate before loading a provider; no device evidence."""
from copy import deepcopy

import numpy as np
import pytest

from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse
from tests.unit.test_portable_bounded_source_attention_reverse import bounded_portable

pytestmark = pytest.mark.compiler_route


def test_bounded_frame_metadata_is_resolved_without_provider(bounded_portable, monkeypatch):
    from tessera import runtime as rt
    def forbidden(*args, **kwargs):
        pytest.fail("frame metadata reached a provider")
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    import ml_dtypes
    storage = bounded_portable["plan"]["buffers"][5]["storage"]
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    shapes = ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3), (1, 2, 3, 3), (1, 2, 3))
    values = tuple(np.zeros(shape, np.float32 if i == 4 else dtype) for i, shape in enumerate(shapes))
    with PreparedSourceAttentionReverse.from_manifest(deepcopy(bounded_portable)) as owner:
        frame = owner._frame(values)
        assert frame[3] == (1, 2, 1, 3, 5, 4, 3)
        assert frame[4] == tuple(x.nbytes for x in values)
        assert [shape for shape, _ in frame[5]] == list(shapes[:3])
        assert owner._lib is None and owner._handle == 0
        assert owner._api.endswith("bounded_") and not owner._times_api.endswith("bounded_")


@pytest.mark.parametrize("fault", ["query_cap", "key_cap", "key_value_rows", "head", "seed_rows", "dtype", "empty"])
def test_bad_bounded_frame_rejected_before_provider(bounded_portable, monkeypatch, fault):
    from tessera import runtime as rt
    def forbidden(*args, **kwargs):
        pytest.fail("invalid bounded frame acquired provider")
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", forbidden)
    import ml_dtypes
    storage = bounded_portable["plan"]["buffers"][5]["storage"]
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    shapes = [(1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3), (1, 2, 3, 3), (1, 2, 3)]
    if fault == "query_cap":
        shapes[0] = (1, 2, 8, 4)
    elif fault == "key_cap":
        shapes[1] = (1, 1, 12, 4)
    elif fault == "key_value_rows":
        shapes[2] = (1, 1, 4, 3)
    elif fault == "head":
        shapes[0] = (1, 3, 3, 4)
    elif fault == "seed_rows":
        shapes[4] = (1, 2, 2)
    elif fault == "empty":
        shapes[0] = (1, 2, 0, 4)
    values = [np.zeros(shape, np.float32 if i == 4 else dtype) for i, shape in enumerate(shapes)]
    if fault == "dtype":
        values[0] = values[0].astype(np.float64)
    with PreparedSourceAttentionReverse.from_manifest(bounded_portable) as owner:
        with pytest.raises(ValueError):
            owner(*values)
        assert owner._lib is None and owner._handle == 0 and owner.receipt is None
