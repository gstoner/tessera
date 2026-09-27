"""Host-free: evidence-packet ineligibility vocabularies are declared and gated.

X86-EVIDENCE-VOCAB-1. The eleven x86 tags decide `eligible_for_promotion` on an
x86 measurement, which makes each one a semantic key under Decision #21a.
Until 2026-09-20 nothing enumerated them: the producer appended bare string
literals, the validator checked only that each was a `str`, and no consumer
could know the vocabulary was eleven items or that a twelfth had been added. A
reader handling a subset therefore treated an unknown reason as NO reason --
promoting a result that something had declined to vouch for.

2026-09-27: the same shape was found in three siblings -- the ROCm profiler
packet, the NVIDIA device-clock packet and the x86 PMU event map, whose
validator did not even re-derive its reasons -- and all four now declare their
vocabulary through `tessera.compiler.evidence_reasons.ReasonVocabulary`.

They are deliberately NOT diagnostic codes and must not be answered by
registering them in `diagnostic_codes.py`; `TIMING_PROOF_INCOMPLETE` was
misfiled that way once already. Tags that ARE registered diagnostics (the
shared device-clock window/witness refusals) are borrowed by `pass_origin`,
never redeclared.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import pytest

from tessera.compiler import (
    profiler_nvidia_evidence as nvidia,
    profiler_rocm_evidence as rocm,
    profiler_timing as timing,
    profiler_x86_event_map as event_map,
    profiler_x86_evidence as x86,
    target_perf,
)
from tessera.compiler.diagnostic_codes import code_lookup
from tessera.compiler.evidence_reasons import ReasonVocabulary, reason_tag
from tessera.compiler.profiler_x86_evidence import (
    X86_INELIGIBILITY_REASONS,
    X86ProfilerPacketError,
    build_x86_profiler_packet,
    digest_json,
    validate_x86_profiler_packet,
    x86_reason_tag,
)

ROOT = Path(__file__).resolve().parents[2]
_TAG = re.compile(r"[A-Z][A-Z0-9_]{2,}")


def _string_heads(node: ast.AST) -> list[str]:
    """The literal tag an expression starts with: `"TAG"`, `"TAG:" + x`,
    `f"TAG:{x}"`, or each element of a list/tuple literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_heads(node.left)
    if isinstance(node, ast.JoinedStr) and node.values:
        return _string_heads(node.values[0])
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [h for elt in node.elts for h in _string_heads(elt)]
    return []


def _emitted_tags(path: Path, lists: frozenset[str] = frozenset({"reasons", "gaps"})) -> set[str]:
    """Every literal tag a producer puts on a reasons/gaps list, by AST.

    Sees `.append(...)`, `.extend([...])`, `.insert(i, ...)`, `x += [...]` and
    list-literal assignments, so a tag emitted in a new style cannot slip past
    the declaration (the first version of this gate saw only
    `reasons.append("TAG"`).
    """
    tags: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        heads: list[str] = []
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in lists \
                and node.func.attr in {"append", "extend", "insert"} and node.args:
            heads = _string_heads(node.args[-1])
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                and node.target.id in lists:
            heads = _string_heads(node.value)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in lists for t in targets) and node.value:
                heads = _string_heads(node.value)
        for head in heads:
            m = _TAG.match(head)
            if m:
                tags.add(m.group(0))
    return tags


def _function_codes(path: Path, function: str) -> set[str]:
    """String constants shaped like a code inside one function's body."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {c.value for c in ast.walk(node)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)
                    and re.fullmatch(r"DEVICE_CLOCK_[A-Z_]+", c.value)}
    raise AssertionError(f"{function} not found in {path}")


def _module(module) -> tuple[str, Path]:
    return module.__name__, Path(module.__file__)


#: vocabulary -> (producer's dotted name, producer source, the call through
#: which the producer's output meets the declaration). The calibration corpus
#: is produced by a benchmark recorder and consumed by `target_perf`, which
#: declares its vocabulary.
_FAMILIES = {
    "x86": (x86.X86_REASON_VOCABULARY, *_module(x86), "validate_x86_profiler_packet(packet)"),
    "x86_event_map": (event_map.X86_EVENT_MAP_VOCABULARY, *_module(event_map),
                      "validate_x86_event_map(body)"),
    "rocm": (rocm.ROCM_REASON_VOCABULARY, *_module(rocm), "validate_rocm_profiler_packet(packet)"),
    "nvidia": (nvidia.NVIDIA_REASON_VOCABULARY, *_module(nvidia),
               "validate_nvidia_device_clock_packet(packet)"),
    "calibration_corpus": (target_perf.CALIBRATION_CORPUS_VOCABULARY, "calibrate_gfx1151",
                           ROOT / "benchmarks/calibration/calibrate_gfx1151.py",
                           "CALIBRATION_CORPUS_VOCABULARY.require_known(reasons"),
}


def _produced(vocab: ReasonVocabulary, path: Path) -> set[str]:
    """Tags the producer can emit: its own literals plus each borrowed code."""
    return _emitted_tags(path) | set(vocab.registered)


def test_x86_vocabulary_is_still_eleven() -> None:
    """The queue item's count, re-measured from the producer, not copied."""
    produced = _emitted_tags(Path(x86.__file__))
    assert len(produced) == 11 == len(X86_INELIGIBILITY_REASONS), sorted(produced)


def test_the_producer_scan_sees_every_emission_style(tmp_path: Path) -> None:
    src = tmp_path / "p.py"
    src.write_text(
        'reasons.append("A_TAG")\n'
        'reasons.append("B_TAG:" + x)\n'
        'reasons.append(f"C_TAG:{x}")\n'
        'reasons.extend(["D_TAG", "E_TAG"])\n'
        'reasons += ["F_TAG"]\n'
        'gaps.insert(0, "G_TAG")\n'
        'reasons = ["H_TAG"]\n'
        'other.append("NOT_A_REASON")\n')
    assert _emitted_tags(src) == {f"{c}_TAG" for c in "ABCDEFGH"}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_vocabulary_matches_its_producer(family: str) -> None:
    """A tag cannot appear, or linger, without the declaration moving with it."""
    vocab, _name, path, _call = _FAMILIES[family]
    assert _emitted_tags(path), (
        f"{family}: the producer scan matched nothing; its spelling has drifted")
    problems = vocab.drift(_produced(vocab, path))
    assert not problems, f"{family} vocabulary drift: {problems}"


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_borrowed_codes_are_registered_and_actually_produced(family: str) -> None:
    """Each borrowed code is registered, and the producer emits it itself or
    calls the function the registry says emits it."""
    vocab, name, path, _call = _FAMILIES[family]
    source = path.read_text()
    own = _emitted_tags(path)
    for code in vocab.registered:
        entry = code_lookup(code)
        assert entry is not None, f"{family}: {code} is not registered"
        if code in own:
            continue
        function = entry.pass_origin.rsplit(".", 1)[1]
        assert f"{function}(" in source, (
            f"{family} borrows {code} ({entry.pass_origin}) but never calls {function}")


def test_the_shared_timing_code_lists_match_their_functions() -> None:
    """The borrowed tuples are exactly what the timing functions return."""
    path = Path(timing.__file__)
    assert _function_codes(path, "device_clock_window_refusals") == \
        set(timing.DEVICE_CLOCK_WINDOW_REFUSAL_CODES)
    assert _function_codes(path, "witness_refusal_codes") == \
        set(timing.DEVICE_CLOCK_WITNESS_REFUSAL_CODES)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_producer_checks_its_output_against_the_declaration(family: str) -> None:
    """In-package builders validate what they build; the out-of-package
    recorder checks its tags explicitly before writing."""
    _vocab, _name, path, call = _FAMILIES[family]
    assert call in path.read_text(), f"{family}: {path.name} no longer calls {call}"


def test_route_classes_read_the_declaration() -> None:
    """The environment/gap classifications are consumers of the vocabulary."""
    assert x86._ENVIRONMENT_TAGS <= set(X86_INELIGIBILITY_REASONS)
    assert rocm._ENVIRONMENT_REASONS <= set(rocm.ROCM_PROFILER_REASONS)
    assert x86._SUPERSEDED_BY_WITNESS.isdisjoint(X86_INELIGIBILITY_REASONS), (
        "superseded clock proofs are TIMING_PROOF_INCOMPLETE details, not tags")


@pytest.mark.parametrize("tag", sorted(X86_INELIGIBILITY_REASONS))
def test_declared_tag_survives_the_detail_split(tag: str) -> None:
    assert x86_reason_tag(f"{tag}:some,detail") == tag == reason_tag(tag)


def test_drift_reports_each_way_a_vocabulary_can_lie() -> None:
    vocab = ReasonVocabulary(
        "synthetic",
        {"A_TAG": "a tag whose meaning is long enough to act on",
         "B_TAG": "a tag nobody produces and nobody reserved either",
         "C_TAG": "a tag reserved for a producer not yet written"},
        reserved={"C_TAG": "held for the owning queue item's future producer"})
    problems = vocab.drift({"A_TAG", "Z_TAG:detail"})
    assert problems == [
        "Z_TAG: produced but not declared",
        "B_TAG: declared but never produced (reserve it with a reason, or delete it)",
    ]
    assert vocab.drift({"A_TAG", "B_TAG"}) == []
    assert "C_TAG: reserved but produced; drop the reservation" in \
        vocab.drift({"A_TAG", "B_TAG", "C_TAG"})


# ─── Fail closed, on real packets ───────────────────────────────────────────


def _real_x86_packet() -> dict:
    """A packet the builder itself produced, so the fixture cannot drift."""
    return build_x86_profiler_packet(
        benchmark={
            "schema": "tessera.compiler.e2e_real4.x86_matmul.v1",
            "architecture": "zen5-avx512",
            "ratchet": {"kind": "production_non_regression", "limit": 1.10},
            "rows": [
                {"shape_class": shape_class, "correctness": {"passed": True},
                 "timing": {"production_samples_ms": [5.0, 5.0, 5.0],
                            "scheduled_samples_ms": [5.0, 5.0, 5.0],
                            "non_regression_10pct": True}}
                for shape_class in ("aligned", "ragged")
            ],
            "verdict": "promote",
        },
        timing_status={
            name: True for name in (
                "avx512_visible", "monotonic_raw_valid", "rdtscp_valid",
                "invariant_tsc", "affinity_stable", "clock_agreement_valid",
                "perf_event_open", "perf_sample_valid",
            )
        },
        cpu={
            "vendor_id": "AuthenticAMD", "cpu_family": 26,
            "model_name": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
            "flags": ["avx512f"],
        },
        environment={
            "wsl": True, "virtualized": True,
            "source_commit": "a" * 40, "worktree_dirty": False,
        },
        sampling=None,
    )


def test_validator_refuses_an_undeclared_reason() -> None:
    """Built from a REAL packet and then spoiled, so the test cannot pass by
    tripping an earlier check -- the first version asserted on a stub that was
    rejected for its schema, which would have passed for the wrong reason."""
    packet = _real_x86_packet()
    assert packet["ineligibility_reasons"], "fixture should already be ineligible"
    packet["ineligibility_reasons"] = [*packet["ineligibility_reasons"],
                                       "SOMETHING_NOBODY_DECLARED"]
    packet["packet_sha256"] = digest_json(
        {k: v for k, v in packet.items() if k != "packet_sha256"})
    with pytest.raises(X86ProfilerPacketError, match="unknown x86 ineligibility reason"):
        validate_x86_profiler_packet(packet)


def test_validator_accepts_the_same_packet_unspoiled() -> None:
    validate_x86_profiler_packet(_real_x86_packet())


def _committed(relpath: str) -> dict:
    return json.loads((ROOT / relpath).read_text())


def _resign(packet: dict, field: str) -> None:
    packet[field] = digest_json({k: v for k, v in packet.items() if k != field})


_EVENT_MAP = "benchmarks/baselines/x86_zen5_event_map_2026_08_07.json"
_ROCM = "benchmarks/baselines/gfx1151_ssd_calibrated_pairs_interleaved_20260926/0-cooperative-calibration.json"
_NVIDIA = "benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/0-cooperative-calibration.json"


def test_committed_packets_validate() -> None:
    """The control for every doctored case below."""
    event_map.validate_x86_event_map(_committed(_EVENT_MAP))
    rocm.validate_rocm_profiler_packet(_committed(_ROCM))
    nvidia.validate_nvidia_device_clock_packet(_committed(_NVIDIA))


def test_event_map_refuses_an_undeclared_reason() -> None:
    doctored = _committed(_EVENT_MAP)
    doctored["ineligibility_reasons"].append("SOMETHING_NOBODY_DECLARED")
    _resign(doctored, "event_map_sha256")
    with pytest.raises(event_map.X86EventMapError, match="unknown x86 event map"):
        event_map.validate_x86_event_map(doctored)


def test_event_map_refuses_a_dropped_reason() -> None:
    """Before 2026-09-27 the event-map validator only type-checked its reasons,
    so a stored map could drop one and still validate."""
    doctored = _committed(_EVENT_MAP)
    assert "VIRTUALIZED_HOST" in doctored["ineligibility_reasons"]
    doctored["ineligibility_reasons"].remove("VIRTUALIZED_HOST")
    _resign(doctored, "event_map_sha256")
    with pytest.raises(event_map.X86EventMapError, match="stored inputs derive"):
        event_map.validate_x86_event_map(doctored)


def test_event_map_refuses_promotion_it_does_not_derive() -> None:
    doctored = _committed(_EVENT_MAP)
    doctored["ineligibility_reasons"] = []
    doctored["eligible_for_promotion"] = True
    _resign(doctored, "event_map_sha256")
    with pytest.raises(event_map.X86EventMapError):
        event_map.validate_x86_event_map(doctored)


@pytest.mark.parametrize("field", ["ineligibility_reasons", "diagnostic_gaps"])
def test_rocm_packet_refuses_an_undeclared_tag(field: str) -> None:
    doctored = copy.deepcopy(_committed(_ROCM))
    doctored[field] = [*doctored.get(field, []), "SOMETHING_NOBODY_DECLARED"]
    doctored["eligible_for_promotion"] = False
    _resign(doctored, "packet_sha256")
    with pytest.raises(rocm.ROCmProfilerPacketError, match="unknown ROCm profiler packet"):
        rocm.validate_rocm_profiler_packet(doctored)


def test_nvidia_packet_refuses_an_undeclared_tag() -> None:
    doctored = _committed(_NVIDIA)
    doctored["ineligibility_reasons"] = ["SOMETHING_NOBODY_DECLARED"]
    doctored["eligible_for_promotion"] = False
    _resign(doctored, "packet_sha256")
    with pytest.raises(nvidia.NVIDIADeviceClockPacketError,
                       match="unknown NVIDIA device-clock packet"):
        nvidia.validate_nvidia_device_clock_packet(doctored)


#: Committed packets that already failed their validator before 2026-09-27:
#: diagnostic/superseded ROCm packets recorded before the per-window rule,
#: which now derive `DEVICE_CLOCK_WINDOW_TOO_SHORT` that they do not state.
#: Retained history, never admitted evidence; shrink-only.
_KNOWN_INVALID = frozenset({
    "benchmarks/baselines/gfx1151_ssd_calibrated_pairs_interleaved_20260926/"
    "diagnostics/launches_probe/cooperative-100-calibration.json",
    "benchmarks/baselines/gfx1201_ssd_calibrated_pairs_20260926/superseded/"
    "second_6e6904dc_short_window/0-cooperative-calibration.json",
})
_VALIDATORS = {
    "tessera.profiler_x86_packet.": validate_x86_profiler_packet,
    "tessera.profiler_x86_event_map.": event_map.validate_x86_event_map,
    "tessera.profiler_rocm_packet.": rocm.validate_rocm_profiler_packet,
    "tessera.profiler_nvidia_device_clock_packet.": nvidia.validate_nvidia_device_clock_packet,
}


def _packets(obj, out: list) -> None:
    if isinstance(obj, dict):
        schema = obj.get("schema")
        for prefix, validator in _VALIDATORS.items():
            if isinstance(schema, str) and schema.startswith(prefix):
                out.append((validator, obj))
        for value in obj.values():
            _packets(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _packets(value, out)


def test_every_committed_packet_still_validates() -> None:
    """The vocabulary gates must not strand recorded evidence."""
    checked, failures, stale = 0, [], set(_KNOWN_INVALID)
    for path in sorted((ROOT / "benchmarks").rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        found: list = []
        _packets(data, found)
        rel = path.relative_to(ROOT).as_posix()
        for validator, packet in found:
            checked += 1
            try:
                validator(packet)
            except ValueError as exc:
                # Pinned to the known reason: a known-invalid packet that starts
                # failing for a NEW reason is a new failure.
                if rel in _KNOWN_INVALID and "DEVICE_CLOCK_WINDOW_TOO_SHORT" in str(exc):
                    stale.discard(rel)
                    continue
                failures.append(f"{rel}: {exc}")
    assert checked > 150, f"only {checked} committed packets found; the walk has drifted"
    assert not failures, failures
    assert not stale, f"now valid; remove from _KNOWN_INVALID: {sorted(stale)}"
