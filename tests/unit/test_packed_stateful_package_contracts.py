"""Physical package values must survive the integer native ABI unchanged."""
import pytest

from tessera.compiler.nvidia_native import package_packed_decode
from tessera.compiler.ssm_replay import replay_state_descriptor


@pytest.mark.parametrize("field,value", [("rows", True), ("columns", 3.5),
    ("offset", -1), ("offset", 2**63), ("strides", (1, False)),
    ("scale_offset", -1), ("alignment", True)])
def test_packed_physical_fields_reject_lossy_values(field, value):
    kwargs = dict(logical="int4", rows=2, columns=3, source_bytes=8)
    kwargs[field] = value
    with pytest.raises(ValueError, match="packed"):
        package_packed_decode(**kwargs)


@pytest.mark.parametrize("field,value", [("batch", True), ("capacity", 3.5),
    ("async_slots", 2.5), ("channels", 2**63), ("state_dim", 2**62)])
@pytest.mark.parametrize("target", ["nvidia_sm120", "rocm_gfx1151", "apple_gpu", "x86"])
def test_replay_geometry_rejects_lossy_or_overflowing_state(field, value, target):
    kwargs = dict(target=target, batch=1, channels=4, state_dim=3, capacity=8, async_slots=2)
    kwargs[field] = value
    with pytest.raises(ValueError, match="ReplaySSM"):
        replay_state_descriptor(**kwargs)


@pytest.mark.parametrize("start,tokens", [(True, 1), (0, True), (0.5, 1), (0, 1.5)])
def test_replay_submission_rejects_noninteger_span(start, tokens):
    state = replay_state_descriptor(target="nvidia_sm120", batch=1, channels=4,
        state_dim=3, capacity=8, async_slots=2)
    with pytest.raises(ValueError, match="integer"):
        state.validate_span(start=start, tokens=tokens)


@pytest.mark.parametrize("field,value", [("start", 3.5), ("start", True),
    ("end", "10"), ("end", None)])
def test_paged_kv_rejects_coerced_bounds(field, value):
    from tessera.compiler.nvidia_native import supports_paged_kv_read
    from tests.unit.test_nvidia_e2e_spine import _paged_kv_module
    module = _paged_kv_module()
    module.functions[0].body[0].kwargs[field] = value
    assert not supports_paged_kv_read(module)


def test_paged_kv_rejects_an_unrelated_function_return():
    from tessera.compiler.nvidia_native import supports_paged_kv_read
    from tests.unit.test_nvidia_e2e_spine import _paged_kv_module
    module = _paged_kv_module()
    module.functions[0].return_values = ["%pages"]
    assert not supports_paged_kv_read(module)
