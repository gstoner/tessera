"""Accepted launch containers must pin result generations without probing CUDA."""
from collections import UserDict
from types import MappingProxyType
import threading

import pytest
from tessera import runtime as rt
from tessera.compiler.native_tensor_result import NativeTensorResult, input_scope


class Batch:
    handle = 1

    def __init__(self):
        self.lock = threading.Lock()
        self.recorded = []

    def guard(self):
        pass

    def record(self, stream):
        assert self.lock.locked()
        self.recorded.append(stream)


@pytest.mark.parametrize("form", ["wrapped", "user_dict", "proxy_groups", "dict_subclass", "list_subclass", "cycles"])
def test_supported_containers_pin_before_consumption_and_record_before_unlock(form, monkeypatch):
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch",
                        lambda: pytest.fail("scope metadata touched CUDA"))
    batch = Batch()
    value = NativeTensorResult(batch, 0)
    wrapped = rt.NativeBufferValue(value, rt.BufferArgument("fp32", (1, 1), "row_major"))
    if form == "wrapped":
        args = {"x": wrapped}
    elif form == "user_dict":
        args = UserDict(x=value)
    elif form == "proxy_groups":
        args = MappingProxyType({"buffers": UserDict(x=wrapped), "scalars": UserDict(Rows=1)})
    elif form == "dict_subclass":
        class Arguments(dict):
            pass
        args = Arguments(x=value)
    elif form == "list_subclass":
        class Arguments(list):
            pass
        args = Arguments([value, wrapped])
    else:
        args = UserDict(x=wrapped)
        args["self"] = args
        nested = [value, args]
        nested.append(nested)
        args["nested"] = nested
    with input_scope(args, stream=23):
        assert batch.lock.locked()
        assert not batch.recorded
    assert not batch.lock.locked()
    assert batch.recorded == [23]
