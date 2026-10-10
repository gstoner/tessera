"""Native member timing units and admission must survive Python marshalling."""
import ctypes as c
from types import SimpleNamespace

import pytest

from tessera.compiler.native_scaled_program import PreparedScaledProgram


def test_member_profile_preserves_native_averages_and_generation():
    calls = []

    def profile(handle, repeats, members, generation, elapsed):
        calls.append((handle.value, repeats, members))
        c.cast(generation, c.POINTER(c.c_uint64))[0] = 9
        elapsed[0], elapsed[1] = 1.5, 2.5
        return 0

    owner = object.__new__(PreparedScaledProgram)
    owner.handle = c.c_uint64(7)
    owner._binding = SimpleNamespace(steps=(object(), object()))
    owner.lib = SimpleNamespace(tessera_rocm_program_profile_members=profile)
    assert owner.profile_members(repeats=4096) == (9, (1.5, 2.5))
    assert calls == [(7, 4096, 2)]
    for invalid in (0, -1, True, 65537, 1.5):
        with pytest.raises(ValueError, match="repetition"):
            owner.profile_members(repeats=invalid)
    assert len(calls) == 1
