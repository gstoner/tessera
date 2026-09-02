"""The write-only device-region contract, made executable.

`_rocm_dev_region` / `_rocm_dev_alloc` skip the host->device upload of an
output buffer. That is only sound when the kernel writes every element it
will later read back, and #690 established it for the dspark outputs by a
one-off manual experiment recorded in a docstring.

A claim checked once by hand is not a contract. `_ROCM_DEV_POISON` fills
each write-only region with a chosen byte before launch, so an adoption
that is NOT write-only shows up as a changed result instead of as silent
garbage on a machine nobody is watching. Every further adoption of the
pattern (there are ~100 remaining `_rocm_dev_in` call sites) should add its
own case here rather than inherit this evidence -- write-only is a property
of a specific kernel, not of the helper.

Three patterns, not one: 0x00 is what the old code uploaded, so it alone
would pass even for a kernel that reads its output buffer.
"""
from __future__ import annotations

import numpy as np
import pytest


POISONS = (0x00, 0xFF, 0xA5)


@pytest.mark.hardware_rocm
def test_selected_block_attention_outputs_are_write_only():
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("needs tessera-opt and a live AMD GPU")

    rng = np.random.default_rng(7007)
    B, Hq, Hkv, Sq, Sk, D, Dv, block, top_k = 1, 8, 2, 32, 256, 64, 64, 16, 8
    q = rng.standard_normal((B, Hq, Sq, D), dtype=np.float32) * 0.1
    k = rng.standard_normal((B, Hkv, Sk, D), dtype=np.float32) * 0.1
    v = rng.standard_normal((B, Hkv, Sk, Dv), dtype=np.float32) * 0.1
    sel = np.tile(np.arange(top_k, dtype=np.int64), (B, Hkv, Sq, 1))

    def run(tiled):
        return rt._rocm_selected_block_attention_native(
            q, k, v, sel, np, block_size=block, causal=False, tiled=tiled)

    reference = {tiled: run(tiled) for tiled in (False, True)}
    saved = rt._ROCM_DEV_POISON
    try:
        for poison in POISONS:
            rt._ROCM_DEV_POISON = poison
            for tiled in (False, True):
                got = run(tiled)
                assert np.array_equal(got, reference[tiled]), (
                    f"selected-block attention (tiled={tiled}) result changed "
                    f"when its output region was pre-filled with 0x{poison:02X}: "
                    "the kernel does not fully write its output, so skipping "
                    "the upload is unsound"
                )
    finally:
        rt._ROCM_DEV_POISON = saved


def test_poison_hook_reports_a_missing_hipmemset():
    """The hook fails loudly rather than silently not poisoning.

    A poison that quietly does nothing turns this whole file into a test
    that always passes -- the exact hollow-green shape it exists to catch.
    """
    from tessera import runtime as rt

    saved = rt._ROCM_DEV_POISON
    try:
        rt._ROCM_DEV_POISON = 0xFF
        with pytest.raises(RuntimeError, match="no hipMemset"):
            rt._rocm_dev_poison(object(), None, 128)
    finally:
        rt._ROCM_DEV_POISON = saved


@pytest.mark.hardware_rocm
def test_poison_actually_reaches_the_device_region():
    """Positive control: the poison must be OBSERVABLE.

    Without this, a poison that silently did nothing would make every
    write-only assertion above pass unconditionally -- a check that has
    evaluated nothing, reported green. Allocate a region under poison and
    read it straight back, with no kernel in between.
    """
    from tessera import runtime as rt

    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        pytest.skip("needs a live AMD GPU")

    probe = np.zeros(4096, dtype=np.uint8)
    saved = rt._ROCM_DEV_POISON
    try:
        rt._ROCM_DEV_POISON = 0xA5
        base, _ptrs, offsets = rt._rocm_dev_region(hip, [probe])
        try:
            rt._rocm_dev_unpack(hip, base, [probe], offsets)
        finally:
            hip.hipFree(base)
    finally:
        rt._ROCM_DEV_POISON = saved

    assert np.all(probe == 0xA5), (
        "poisoning a write-only region left "
        f"{int((probe != 0xA5).sum())} of {probe.size} bytes unset -- the "
        "write-only assertions in this file would pass vacuously"
    )


class _StubHip:
    """A HIP stand-in that lets each post-allocation step be made to fail.

    These paths need no GPU -- they are about what happens to an allocation
    the caller never receives -- so they run on every host, including CI,
    which has no AMD device at all.
    """

    def __init__(self, *, memcpy_rc: int = 0, has_memset: bool = True,
                 memset_rc: int = 0):
        self.freed: list = []
        self.allocated = 0
        self._memcpy_rc = memcpy_rc
        self._memset_rc = memset_rc
        if has_memset:
            self.hipMemset = self._memset

    def hipMalloc(self, _ref, _size):
        self.allocated += 1
        return 0

    def hipMemcpy(self, *_a):
        return self._memcpy_rc

    def _memset(self, *_a):
        return self._memset_rc

    def hipFree(self, ptr):
        self.freed.append(ptr)
        return 0


def test_pack_frees_when_the_upload_fails():
    """A failed host->device copy must not strand the allocation."""
    from tessera import runtime as rt

    hip = _StubHip(memcpy_rc=1)
    with pytest.raises(RuntimeError, match="host->device copy failed"):
        rt._rocm_dev_pack(hip, [np.zeros(16, dtype=np.float32)])
    assert hip.allocated == 1
    assert len(hip.freed) == 1, (
        "the caller never receives this base, so nothing else can free it"
    )


def test_region_frees_when_the_allocation_is_unusable():
    from tessera import runtime as rt

    hip = _StubHip()
    with pytest.raises(RuntimeError, match="null base pointer"):
        rt._rocm_dev_region(hip, [np.zeros(16, dtype=np.float32)])
    assert hip.allocated == 1 and len(hip.freed) == 1


@pytest.mark.parametrize("kwargs,match", [
    ({"has_memset": False}, "no hipMemset"),
    ({"memset_rc": 1}, "hipMemset failed"),
])
def test_alloc_frees_when_poisoning_fails(kwargs, match):
    """Poisoning is diagnostic; failing it must not cost device memory.

    Without this the poison hook would be self-defeating: enabling it to
    validate an adoption would leak a region per attempt, and a long
    validation run would exhaust the device it was meant to be checking.
    """
    from tessera import runtime as rt

    hip = _StubHip(**kwargs)
    saved = rt._ROCM_DEV_POISON
    try:
        rt._ROCM_DEV_POISON = 0xFF
        with pytest.raises(RuntimeError, match=match):
            rt._rocm_dev_alloc(hip, 128)
    finally:
        rt._ROCM_DEV_POISON = saved
    assert hip.allocated == 1 and len(hip.freed) == 1
