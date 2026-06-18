"""C1 — symmetric-heap placement model tests."""

from __future__ import annotations

import pytest

from tessera.symmetric_heap import (
    SYMMETRIC_MODES,
    SymmetricHeap,
    SymmetricShardSpec,
)


# ── construction + validation ───────────────────────────────────────────────
def test_basic_construction() -> None:
    h = SymmetricHeap(num_ranks=4, bytes_per_rank=1024, mesh_axis="dp")
    assert h.num_ranks == 4
    assert h.bytes_per_rank == 1024
    assert h.mesh_axis == "dp"
    assert h.alignment == 256
    assert h.total_bytes == 4096


def test_zero_bytes_is_valid() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=0)
    assert h.total_bytes == 0


def test_num_ranks_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_ranks"):
        SymmetricHeap(num_ranks=0, bytes_per_rank=256)


def test_negative_bytes_rejected() -> None:
    with pytest.raises(ValueError, match="bytes_per_rank"):
        SymmetricHeap(num_ranks=2, bytes_per_rank=-256)


def test_unaligned_bytes_rejected() -> None:
    with pytest.raises(ValueError, match="multiple of alignment"):
        SymmetricHeap(num_ranks=2, bytes_per_rank=300, alignment=256)


def test_non_power_of_two_alignment_rejected() -> None:
    with pytest.raises(ValueError, match="power of two"):
        SymmetricHeap(num_ranks=2, bytes_per_rank=300, alignment=300)


def test_custom_alignment_power_of_two() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=128, alignment=64)
    assert h.alignment == 64


# ── Iris remote-address translation ─────────────────────────────────────────
def test_remote_address_reduces_to_base_plus_offset() -> None:
    h = SymmetricHeap(num_ranks=3, bytes_per_rank=1024)
    bases = [0x1000, 0x9000, 0x20000]  # bases DIFFER per rank — only offset is symmetric
    # offset 64 on rank 0 as seen from rank 2 → bases[2] + 64
    assert h.remote_address(64, src_rank=0, dst_rank=2, heap_bases=bases) == 0x20000 + 64


def test_remote_address_round_trip_across_all_pairs() -> None:
    h = SymmetricHeap(num_ranks=4, bytes_per_rank=512)
    bases = [100, 5000, 99999, 7]
    for off in (0, 1, 255, 511):
        for i in range(4):
            for j in range(4):
                got = h.remote_address(off, src_rank=i, dst_rank=j, heap_bases=bases)
                assert got == bases[j] + off


def test_remote_address_self_is_identity() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=256)
    bases = [4096, 9000]
    assert h.remote_address(128, src_rank=1, dst_rank=1, heap_bases=bases) == 9000 + 128


def test_remote_address_bad_src_rank() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=256)
    with pytest.raises(ValueError, match="src_rank"):
        h.remote_address(0, src_rank=5, dst_rank=0, heap_bases=[0, 1])


def test_remote_address_bad_dst_rank() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=256)
    with pytest.raises(ValueError, match="dst_rank"):
        h.remote_address(0, src_rank=0, dst_rank=-1, heap_bases=[0, 1])


def test_remote_address_offset_past_bytes() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=256)
    with pytest.raises(ValueError, match="local_offset"):
        h.remote_address(256, src_rank=0, dst_rank=1, heap_bases=[0, 1])


def test_remote_address_negative_offset() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=256)
    with pytest.raises(ValueError, match="local_offset"):
        h.remote_address(-1, src_rank=0, dst_rank=1, heap_bases=[0, 1])


def test_remote_address_wrong_bases_length() -> None:
    h = SymmetricHeap(num_ranks=4, bytes_per_rank=256)
    with pytest.raises(ValueError, match="heap_bases length"):
        h.remote_address(0, src_rank=0, dst_rank=1, heap_bases=[0, 1])


# ── partition helpers ───────────────────────────────────────────────────────
def test_owner_rank_block_layout() -> None:
    h = SymmetricHeap(num_ranks=4, bytes_per_rank=1024)
    assert h.owner_rank(0) == 0
    assert h.owner_rank(1023) == 0
    assert h.owner_rank(1024) == 1
    assert h.owner_rank(4095) == 3


def test_owner_rank_out_of_range() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=256)
    with pytest.raises(ValueError, match="out of range"):
        h.owner_rank(512)


def test_owner_rank_zero_byte_heap() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=0)
    with pytest.raises(ValueError, match="zero-byte"):
        h.owner_rank(0)


def test_local_slice() -> None:
    h = SymmetricHeap(num_ranks=4, bytes_per_rank=1024)
    assert h.local_slice(0) == (0, 1024)
    assert h.local_slice(2) == (2048, 3072)


def test_local_slice_bad_rank() -> None:
    h = SymmetricHeap(num_ranks=2, bytes_per_rank=256)
    with pytest.raises(ValueError, match="rank"):
        h.local_slice(3)


def test_global_to_local() -> None:
    h = SymmetricHeap(num_ranks=4, bytes_per_rank=1024)
    assert h.global_to_local(0) == (0, 0)
    assert h.global_to_local(1024) == (1, 0)
    assert h.global_to_local(2050) == (2, 2)


# ── SymmetricShardSpec — partitioned ─────────────────────────────────────────
def test_shard_spec_default_mode_partitioned() -> None:
    spec = SymmetricShardSpec(heap=SymmetricHeap(num_ranks=4, bytes_per_rank=1024))
    assert spec.mode == "partitioned"
    assert not spec.is_replicated
    assert spec.logical_bytes == 4096


def test_shard_spec_partitioned_global_to_rank() -> None:
    spec = SymmetricShardSpec(
        heap=SymmetricHeap(num_ranks=4, bytes_per_rank=1024),
        mode="partitioned",
    )
    assert spec.global_to_rank(0) == (0, 0)
    assert spec.global_to_rank(1024) == (1, 0)
    assert spec.global_to_rank(3500) == (3, 3500 - 3072)


def test_shard_spec_partitioned_shard_bytes() -> None:
    spec = SymmetricShardSpec(heap=SymmetricHeap(num_ranks=4, bytes_per_rank=1024))
    for r in range(4):
        assert spec.shard_bytes(r) == 1024


def test_shard_spec_shard_bytes_bad_rank() -> None:
    spec = SymmetricShardSpec(heap=SymmetricHeap(num_ranks=2, bytes_per_rank=256))
    with pytest.raises(ValueError, match="rank"):
        spec.shard_bytes(9)


# ── SymmetricShardSpec — replicated ──────────────────────────────────────────
def test_shard_spec_replicated_holds_full_bank() -> None:
    spec = SymmetricShardSpec(
        heap=SymmetricHeap(num_ranks=4, bytes_per_rank=1024),
        mode="replicated",
    )
    assert spec.is_replicated
    # logical bank is one bank's worth — replicated across ranks
    assert spec.logical_bytes == 1024
    # every rank physically holds the full bank
    for r in range(4):
        assert spec.shard_bytes(r) == 1024


def test_shard_spec_replicated_global_to_rank() -> None:
    spec = SymmetricShardSpec(
        heap=SymmetricHeap(num_ranks=4, bytes_per_rank=1024),
        mode="replicated",
    )
    # any rank serves a replicated read; by convention rank 0, offset unchanged
    assert spec.global_to_rank(500) == (0, 500)


def test_shard_spec_replicated_offset_out_of_range() -> None:
    spec = SymmetricShardSpec(
        heap=SymmetricHeap(num_ranks=4, bytes_per_rank=1024),
        mode="replicated",
    )
    with pytest.raises(ValueError, match="out of range"):
        spec.global_to_rank(1024)


def test_shard_spec_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        SymmetricShardSpec(
            heap=SymmetricHeap(num_ranks=2, bytes_per_rank=256),
            mode="sharded",
        )


def test_symmetric_modes_constant() -> None:
    assert SYMMETRIC_MODES == ("replicated", "partitioned")


# ── as_metadata_dict round-trips ─────────────────────────────────────────────
def test_heap_metadata_dict() -> None:
    h = SymmetricHeap(num_ranks=4, bytes_per_rank=1024, mesh_axis="dp", alignment=128)
    md = h.as_metadata_dict()
    assert md == {
        "num_ranks": 4,
        "bytes_per_rank": 1024,
        "mesh_axis": "dp",
        "alignment": 128,
        "total_bytes": 4096,
    }
    # reconstruct from the metadata
    h2 = SymmetricHeap(
        num_ranks=md["num_ranks"],
        bytes_per_rank=md["bytes_per_rank"],
        mesh_axis=md["mesh_axis"],
        alignment=md["alignment"],
    )
    assert h2 == h


def test_shard_spec_metadata_dict() -> None:
    spec = SymmetricShardSpec(
        heap=SymmetricHeap(num_ranks=2, bytes_per_rank=512, mesh_axis="ep"),
        mode="replicated",
    )
    md = spec.as_metadata_dict()
    assert md["mode"] == "replicated"
    assert md["logical_bytes"] == 512
    assert md["heap"]["num_ranks"] == 2
    assert md["heap"]["mesh_axis"] == "ep"
