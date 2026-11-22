"""AR-DECODE-PROGRAM-1 S1a: in-place paged KV write on sm_120.

The kernel is compiled Graph -> Schedule -> Tile -> NVIDIA by the native
passes and checked against the S0 oracle's paged layout
(:func:`tessera.models.ar_decode.paged_kv_pages`). Every data-dependent bound
is checked on the device; a refused write must leave the caller's pages
untouched and report the refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_cuda_host_ready

PIPELINE = "tessera-nvidia-pipeline-sm120"
NAMES = ("pages", "page_table", "values", "position")


@pytest.fixture(autouse=True)
def _sm120():
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    from tessera import runtime as rt
    assert rt._nvidia_device_name() == "sm_120"


@pytest.fixture(scope="module")
def oracle():
    from tessera.models import ar_decode as A
    weights = A.tiny_fixture_weights()
    contract = A.tiny_fixture_contract()
    trace = A.reference_decode(A.AR_DECODE_TINY, weights, contract, A.TINY_PROMPT, 20)
    # Layer-0 keys in logical page order: [pages, page_size, Hkv, D].
    k_pages, _ = A.paged_kv_pages(A.AR_DECODE_TINY, contract, trace.state)[0]
    return contract, trace.state.position, k_pages.astype(np.float32)


_PACKAGES: dict = {}


def _package(dims):
    from tessera.compiler import nvidia_native as native
    if dims not in _PACKAGES:
        _PACKAGES[dims] = native.package_paged_kv_write(NAMES, dims, pipeline_name=PIPELINE)
    return _PACKAGES[dims]


def _write(pages, table, values, position):
    from tessera import runtime as rt
    P, PS, H, D = pages.shape
    package = _package((P, table.size, PS, H, D, values.shape[0]))
    status = np.zeros(1, np.int32)
    artifact = rt.RuntimeArtifact(
        tile_ir=package.tile_ir, target_ir=package.target_ir, metadata={"target": "nvidia_sm120"},
        native_image=package.image, launch_descriptor=package.descriptor)
    bindings = {"pages": pages, "page_table": table, "values": values,
                "position": np.array([position], np.int32), "status": status,
                "P": P, "LP": table.size, "PageSize": PS, "H": H, "D": D, "Tokens": values.shape[0]}
    result = rt.launch(rt.RuntimeArtifact.from_json(artifact.to_json()), bindings)
    return result, status


def _physical(logical_pages, table):
    out = np.zeros_like(logical_pages)
    for lp, phys in enumerate(table):
        out[phys] = logical_pages[lp]
    return out


def test_decode_writes_reproduce_the_oracle_layout(oracle):
    """Token-by-token (T=1) writes through a permuted table, bitwise vs the oracle."""
    contract, live, k_logical = oracle
    table = np.array([2, 0, 3, 1], np.int32)
    flat = k_logical.reshape(-1, *k_logical.shape[2:])
    pages = np.zeros_like(k_logical)
    for t in range(live):
        result, status = _write(pages, table, np.ascontiguousarray(flat[t:t + 1]), t)
        assert result.get("ok") is True, result
        assert result.get("execution_kind") == "native_gpu", result
        assert status[0] == 0
    np.testing.assert_array_equal(pages, _physical(k_logical, table))


def test_prefill_write_crosses_a_page_boundary(oracle):
    contract, live, k_logical = oracle
    table = np.array([1, 3, 0, 2], np.int32)
    flat = k_logical.reshape(-1, *k_logical.shape[2:])
    pages = np.zeros_like(k_logical)
    result, status = _write(pages, table, np.ascontiguousarray(flat[0:20]), 0)  # page 0 -> 1
    assert result.get("ok") is True and status[0] == 0, result
    np.testing.assert_array_equal(pages, _physical(k_logical, table))


def test_write_is_in_place_and_touches_only_its_rows(oracle):
    contract, live, k_logical = oracle
    table = np.array([0, 1, 2, 3], np.int32)
    sentinel = np.full_like(k_logical, 7.5)
    pages = sentinel.copy()
    row = np.ascontiguousarray(k_logical.reshape(-1, *k_logical.shape[2:])[17:18])
    result, status = _write(pages, table, row, 17)
    assert result.get("ok") is True and status[0] == 0, result
    expected = sentinel.copy()
    expected[1, 1] = row[0]  # logical 17 -> page 1, slot 1
    np.testing.assert_array_equal(pages, expected)


@pytest.mark.parametrize("case", ["negative_position", "past_capacity", "page_out_of_range",
                                  "negative_page"])
def test_refused_write_leaves_pages_untouched(oracle, case):
    contract, live, k_logical = oracle
    table = np.array([0, 1, 2, 3], np.int32)
    position = {"negative_position": -1, "past_capacity": 63}.get(case, 0)
    if case == "page_out_of_range":
        table = np.array([4, 1, 2, 3], np.int32)
    if case == "negative_page":
        table = np.array([-1, 1, 2, 3], np.int32)
    tokens = 2 if case == "past_capacity" else 1  # 63 + 2 > 64
    values = np.ones((tokens, *k_logical.shape[2:]), np.float32)
    pages = np.full_like(k_logical, 3.25)
    before = pages.copy()
    result, status = _write(pages, table, values, position)
    assert result.get("ok") is not True, result
    assert "refused" in str(result.get("reason", result)), result
    assert status[0] == 1
    np.testing.assert_array_equal(pages, before)
