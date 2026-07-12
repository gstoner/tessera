"""Architecture-neutral device execution-proof vocabulary."""

from __future__ import annotations

from pathlib import Path

from tessera.compiler import backend_manifest as bm


ROOT = Path(__file__).resolve().parents[2]


def test_device_execution_proof_definitions_are_stable() -> None:
    assert bm.DEVICE_EXECUTION_PROOF_DEFINITIONS == {
        "device_verified_jit": (
            "compiler-generated target binary, launched on the exact target and "
            "numerically verified; no stable public C ABI required"
        ),
        "device_verified_abi": (
            "shipped stable C ABI runtime symbol, launched on the exact target and "
            "numerically verified"
        ),
    }


def test_every_device_verified_row_has_numerical_evidence() -> None:
    for op_name, entries in bm.all_manifests().items():
        for entry in entries:
            assert entry.status not in {"compiled", "hardware_verified"}, op_name
            if entry.status not in {
                bm.DEVICE_VERIFIED_JIT_STATUS,
                bm.DEVICE_VERIFIED_ABI_STATUS,
            }:
                continue
            assert entry.execute_compare_fixture, (op_name, entry.target)
            assert (ROOT / entry.execute_compare_fixture).is_file(), (
                op_name, entry.target, entry.execute_compare_fixture)
            if entry.status == bm.DEVICE_VERIFIED_ABI_STATUS:
                assert entry.runtime_symbol, (op_name, entry.target)


def test_statuses_do_not_promote_reference_targets() -> None:
    summary = bm.manifest_summary()
    assert set(summary["cpu"]) == {"reference"}
    assert not ({"device_verified_jit", "device_verified_abi"}
                & set(summary["apple_cpu"]))


def test_execution_proof_vocabulary_is_used_across_native_architectures() -> None:
    summary = bm.manifest_summary()
    assert summary["x86"]["device_verified_jit"] > 0
    assert summary["apple_gpu"]["device_verified_jit"] > 0
    assert summary["apple_gpu"]["device_verified_abi"] > 0
    assert summary["rocm"]["device_verified_jit"] > 0
    assert summary["rocm"]["device_verified_abi"] > 0
    assert summary["nvidia_sm120"]["device_verified_abi"] > 0
