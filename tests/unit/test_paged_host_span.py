"""Backing capacity checks for positive-stride native page buffers."""
import numpy as np
import pytest

from tessera.compiler.paged_host_span import checked_page_span


@pytest.mark.parametrize("view", [
    lambda x: x,
    lambda x: x[:, :, :, ::2],
    lambda x: x.transpose(3, 1, 2, 0),
    lambda x: x[1:, 1:, :, 1::2],
])
def test_page_span_preserves_pitch_permutation_and_offset(view):
    backing = np.arange(4*3*2*10, dtype=np.float32).reshape(4, 3, 2, 10)
    pages = view(backing)
    before = pages.copy()
    span, strides = checked_page_span(pages)
    assert strides == tuple(s//4 for s in pages.strides)
    assert span == 4 + sum((d-1)*s for d, s in zip(pages.shape, pages.strides, strict=True))
    np.testing.assert_array_equal(pages, before)


def test_f_contiguous_allocation_is_a_valid_capacity_certificate():
    pages = np.array(np.zeros((4, 3, 2, 5), np.float32), order="F")
    assert checked_page_span(pages) == (pages.nbytes, tuple(s//4 for s in pages.strides))


def test_readonly_byte_buffer_can_supply_pages():
    pages = np.frombuffer(bytes(4*3*2*5*4), dtype=np.float32).reshape(4, 3, 2, 5)
    assert not pages.flags.writeable
    assert checked_page_span(pages)[0] == pages.nbytes


@pytest.mark.parametrize("strides", [(120, 40, 20, -4), (120, 40, 20, 0), (120, 40, 20, 3)])
def test_negative_zero_or_fractional_element_strides_are_rejected(strides):
    owner = np.empty(128, np.float32)
    pages = np.lib.stride_tricks.as_strided(owner, shape=(2, 2, 2, 2), strides=strides)
    with pytest.raises(ValueError, match="whole-element strides"):
        checked_page_span(pages)


def test_forged_as_strided_extent_cannot_certify_its_own_capacity():
    owner = np.empty(8, np.float32)
    pages = np.lib.stride_tricks.as_strided(owner, shape=(4, 3, 2, 5), strides=(120, 40, 20, 4))
    with pytest.raises(ValueError, match="backing allocation capacity"):
        checked_page_span(pages)


def test_overflowing_extent_is_rejected_without_reading_it():
    owner = np.empty(1, np.float32)
    pages = np.lib.stride_tricks.as_strided(owner, shape=(2, 1, 1, 1), strides=(2**63-4, 4, 4, 4))
    with pytest.raises(ValueError, match="signed native extent"):
        checked_page_span(pages)
