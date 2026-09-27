"""EVIDENCE-PACKET-1: the shared evidence envelope.

Every committed measurement packet reads through `read_evidence_packet`, or is
refused for the reason it already had; doctored packets refuse at the family
validator or the envelope, never pass. Sync `EVIDENCE-PACKET-1-2026-09-27`.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from tessera.compiler import evidence_envelope as ee
from tessera.compiler import profiler_nvidia_evidence as nvidia
from tessera.compiler import profiler_rocm_evidence as rocm
from tessera.compiler import profiler_x86_evidence as x86
from tessera.compiler.profiler_x86_evidence import digest_json

ROOT = Path(__file__).resolve().parents[2]

_ROCM = "benchmarks/baselines/gfx1151_ssd_calibrated_pairs_interleaved_20260926/0-cooperative-calibration.json"
_NVIDIA = "benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/0-cooperative-calibration.json"
_X86_V2 = "benchmarks/baselines/x86_zen5_profiler_packet_20260926_princess_luna.json"
_X86_V1 = "benchmarks/baselines/x86_zen5_profiler_packet_2026_08_06.json"

#: Committed packets that failed their family validator before the envelope
#: existed (they derive `DEVICE_CLOCK_WINDOW_TOO_SHORT` they do not state; see
#: test_x86_evidence_vocabulary._KNOWN_INVALID). Shrink-only.
_KNOWN_INVALID = frozenset({
    "benchmarks/baselines/gfx1151_ssd_calibrated_pairs_interleaved_20260926/"
    "diagnostics/launches_probe/cooperative-100-calibration.json",
    "benchmarks/baselines/gfx1201_ssd_calibrated_pairs_20260926/superseded/"
    "second_6e6904dc_short_window/0-cooperative-calibration.json",
})


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def _reseal(packet: dict) -> dict:
    """Re-sign a doctored packet as a forger would, so the digest is not what
    refuses it."""
    if "timing" in packet:
        packet["timing_sha256"] = digest_json(packet["timing"])
    if "benchmark" in packet:
        packet["benchmark_sha256"] = digest_json(packet["benchmark"])
    packet.pop("packet_sha256", None)
    packet["packet_sha256"] = digest_json(packet)
    return packet


# --------------------------------------------------------------------------
# Committed evidence.
# --------------------------------------------------------------------------

def test_every_committed_packet_reads_or_keeps_its_old_refusal() -> None:
    by_family: dict[str, int] = {}
    failures, stale = [], set(_KNOWN_INVALID)
    for path in sorted((ROOT / "benchmarks").rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        rel = path.relative_to(ROOT).as_posix()
        for packet in ee.iter_packets(data):
            try:
                env = ee.read_evidence_packet(packet)
            except ValueError as exc:
                if rel in _KNOWN_INVALID and "DEVICE_CLOCK_WINDOW_TOO_SHORT" in str(exc):
                    stale.discard(rel)
                    continue
                failures.append(f"{rel}: {exc}")
                continue
            by_family[env.family] = by_family.get(env.family, 0) + 1
    assert not failures, failures
    assert not stale, f"now valid; remove from _KNOWN_INVALID: {sorted(stale)}"
    # Every family has committed evidence the envelope actually read.
    assert set(by_family) == set(ee.evidence_families()), by_family
    assert sum(by_family.values()) > 180, by_family


@pytest.mark.parametrize("rel,family,route,eligible", [
    (_ROCM, "rocm_profiler", "device_clock_witness", True),
    (_NVIDIA, "nvidia_device_clock", "device_clock_witness", True),
    (_X86_V2, "x86_profiler", "tsc_witness", True),
    (_X86_V1, "x86_profiler", "profiler_correlated", False),
])
def test_committed_packet_projects_every_envelope_field(rel, family, route, eligible) -> None:
    env = ee.read_evidence_packet(_load(rel))
    assert (env.family, env.admission_route, env.eligible_for_promotion) == (family, route, eligible)
    assert env.artifacts and all(len(d) == 64 for _, d in env.artifacts)
    assert len(env.source_commit) == 40
    assert env.execution_environment == "wsl2"
    assert env.timing_domain in ee.evidence_families()[family].timing_domains
    if eligible:
        assert env.clock_valid and env.image_bound_to_sample and env.sample_ids
        assert not env.refusal_causes and not env.worktree_dirty
    else:
        assert env.refusal_causes
    json.dumps(env.to_dict())  # serializable, for recorders that stamp it


def test_x86_verdict_is_a_refusal_cause_even_without_a_tag() -> None:
    env = ee.read_evidence_packet(_load(_X86_V1))
    assert "benchmark_verdict=retain" in env.refusal_causes
    assert env.compiler_identity  # compiler + toolchain fingerprints per row


def test_gpu_families_have_no_compiler_identity_yet() -> None:
    """Recorded, not required: the GPU packets name their image, ISA and
    semantic digests but no compiler build. The plan record keeps this open."""
    assert ee.read_evidence_packet(_load(_ROCM)).compiler_identity == ()
    assert ee.read_evidence_packet(_load(_NVIDIA)).compiler_identity == ()


# --------------------------------------------------------------------------
# Doctored evidence: refuse at the family or the envelope, never pass.
# --------------------------------------------------------------------------

def test_unknown_schema_refuses() -> None:
    doctored = _load(_ROCM)
    doctored["schema"] = "tessera.profiler_rocm_packet.v9"
    with pytest.raises(ee.EvidenceEnvelopeError, match="EVIDENCE_ENVELOPE_SCHEMA_UNKNOWN"):
        ee.read_evidence_packet(doctored)
    with pytest.raises(ee.EvidenceEnvelopeError, match="must be a JSON object"):
        ee.read_evidence_packet([])  # type: ignore[arg-type]


def test_a_pinned_family_refuses_another_familys_valid_packet() -> None:
    with pytest.raises(ee.EvidenceEnvelopeError, match="expected a rocm_profiler packet"):
        ee.read_evidence_packet(_load(_NVIDIA), family="rocm_profiler")


@pytest.mark.parametrize("value", ["MISSING", "false", 0, None])
def test_rocm_worktree_state_must_be_a_bool(value) -> None:
    """Before 2026-09-27 `if source.get("worktree_dirty")` read each of these
    as a clean tree, so the packet derived no SOURCE_WORKTREE_DIRTY."""
    doctored = _load(_ROCM)
    if value == "MISSING":
        del doctored["source"]["worktree_dirty"]
    else:
        doctored["source"]["worktree_dirty"] = value
    _reseal(doctored)
    with pytest.raises(rocm.ROCmProfilerPacketError, match="worktree_dirty as a bool"):
        ee.read_evidence_packet(doctored)


@pytest.mark.parametrize("fact", ["wsl", "virtualized", "worktree_dirty"])
def test_x86_environment_facts_must_be_bools(fact) -> None:
    doctored = _load(_X86_V2)
    del doctored["environment"][fact]
    _reseal(doctored)
    with pytest.raises(x86.X86ProfilerPacketError, match=f"state '{fact}' as a bool"):
        ee.read_evidence_packet(doctored)


def test_nvidia_already_refused_a_missing_worktree_state() -> None:
    doctored = _load(_NVIDIA)
    del doctored["source"]["worktree_dirty"]
    _reseal(doctored)
    with pytest.raises(ValueError):
        ee.read_evidence_packet(doctored)


def test_a_non_digest_artifact_identity_refuses() -> None:
    doctored = _load(_ROCM)
    doctored["instrumentation_comparison"]["instrumented"]["isa_sha256"] = "isa-probe"
    _reseal(doctored)
    with pytest.raises(ee.EvidenceEnvelopeError, match="EVIDENCE_ENVELOPE_INCOMPLETE.*instrumented_isa"):
        ee.read_evidence_packet(doctored)


def test_x86_row_without_compiler_identity_refuses() -> None:
    doctored = _load(_X86_V2)
    del doctored["benchmark"]["rows"][0]["compile"]["toolchain_fingerprint"]
    _reseal(doctored)
    with pytest.raises(ee.EvidenceEnvelopeError, match="toolchain_fingerprint"):
        ee.read_evidence_packet(doctored)


def test_undeclared_timing_domain_refuses() -> None:
    doctored = _load(_X86_V2)
    doctored["benchmark"]["timing_domain"] = "kernel_elapsed_ms"
    _reseal(doctored)
    with pytest.raises(ee.EvidenceEnvelopeError, match="timing domain 'kernel_elapsed_ms'"):
        ee.read_evidence_packet(doctored)


# --------------------------------------------------------------------------
# Envelope invariants, checked directly: each must refuse even if a family
# validator (today's or a future one) let the packet through.
# --------------------------------------------------------------------------

def _family_and_env(rel: str):
    env = ee.read_evidence_packet(_load(rel))
    return ee.evidence_families()[env.family], env


@pytest.mark.parametrize("change,match", [
    (dict(eligible_for_promotion=False), "names no cause"),
    (dict(refusal_causes=("SOURCE_WORKTREE_DIRTY",), ineligibility_reasons=("SOURCE_WORKTREE_DIRTY",)),
     "names no cause"),
    (dict(clock_valid=False), "invalid clock"),
    (dict(image_bound_to_sample=False), "not named by its timing sample"),
    (dict(worktree_dirty=True), "dirty worktree"),
    (dict(sample_ids=()), "names no timing sample"),
    (dict(diagnostic_gaps=("SOURCE_WORKTREE_DIRTY",)), "dirty worktree as a non-blocking gap"),
    (dict(admission_route="profiler_correlated"), "admission route"),
    (dict(ineligibility_reasons=("NOBODY_DECLARED",), refusal_causes=("NOBODY_DECLARED",),
          eligible_for_promotion=False), "unknown"),
    (dict(source_commit="abc"), "full sha"),
    (dict(packet_sha256="x"), "packet digest"),
    (dict(execution_environment="laptop"), "execution environment"),
    (dict(artifacts=()), "no measured artifact"),
])
def test_each_invariant_refuses(change, match) -> None:
    family, env = _family_and_env(_NVIDIA)
    ee._check_invariants(family, env)  # the control
    with pytest.raises(ee.EvidenceEnvelopeError, match=match):
        ee._check_invariants(family, dataclasses.replace(env, **change))


def test_require_promotable_names_every_cause() -> None:
    ee.require_promotable(ee.read_evidence_packet(_load(_ROCM)))
    with pytest.raises(ee.EvidenceEnvelopeError, match="VIRTUALIZED_HOST.*benchmark_verdict=retain"):
        ee.require_promotable(ee.read_evidence_packet(_load(_X86_V1)))


def test_cli_reads_committed_and_refuses_doctored(tmp_path, capsys) -> None:
    assert ee.main([str(ROOT / _ROCM), str(ROOT / _X86_V1)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "retain:" in out
    doctored = _load(_ROCM)
    doctored["ineligibility_reasons"] = ["NOBODY_DECLARED"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_reseal(doctored)))
    assert ee.main([str(bad)]) == 1


# --------------------------------------------------------------------------
# Drift gates: every family schema is registered, and consumers read through
# the envelope rather than calling a family validator directly.
# --------------------------------------------------------------------------

def test_every_packet_schema_constant_is_registered() -> None:
    constants = {
        x86.X86_PROFILER_PACKET_SCHEMA_VERSION, x86.X86_PROFILER_PACKET_SCHEMA_V1,
        rocm.ROCM_PROFILER_PACKET_SCHEMA_VERSION,
        nvidia.NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION,
    }
    registered = {s for f in ee.evidence_families().values() for s in f.schemas}
    assert constants == registered


def test_each_family_route_and_vocabulary_is_its_own() -> None:
    families = ee.evidence_families()
    assert families["x86_profiler"].vocabulary is x86.X86_REASON_VOCABULARY
    assert families["rocm_profiler"].vocabulary is rocm.ROCM_REASON_VOCABULARY
    assert families["nvidia_device_clock"].vocabulary is nvidia.NVIDIA_REASON_VOCABULARY
    assert families["nvidia_device_clock"].routes == {nvidia.ROUTE_DEVICE_CLOCK}


_FAMILY_VALIDATORS = {
    "validate_x86_profiler_packet", "validate_rocm_profiler_packet",
    "validate_nvidia_device_clock_packet",
}
#: Modules allowed to call a family validator by name: each family module
#: (its builder self-checks) and the envelope that dispatches to them.
_ALLOWED = {
    "python/tessera/compiler/evidence_envelope.py",
    "python/tessera/compiler/profiler_x86_evidence.py",
    "python/tessera/compiler/profiler_rocm_evidence.py",
    "python/tessera/compiler/profiler_nvidia_evidence.py",
}


def test_no_production_consumer_bypasses_the_envelope() -> None:
    offenders = []
    for path in sorted((ROOT / "python" / "tessera").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in _FAMILY_VALIDATORS:
                    offenders.append(f"{rel}:{node.lineno} calls {name}")
    assert not offenders, offenders


def test_iter_packets_finds_nested_packets_once() -> None:
    packet = _load(_ROCM)
    bundle = {"runs": [{"calibration": packet}, copy.deepcopy(packet)], "meta": {"schema": "other"}}
    assert len(ee.iter_packets(bundle)) == 2
