"""Host-free: the x86 promotion-ineligibility vocabulary is declared and gated.

These eleven tags decide `eligible_for_promotion` on an x86 measurement, which
makes each one a semantic key under Decision #21a. Until 2026-09-20 nothing
enumerated them: the producer appended bare string literals, the validator
checked only that each was a `str`, and no consumer could know the vocabulary
was eleven items or that a twelfth had been added. A reader handling a subset
therefore treated an unknown reason as NO reason — promoting a result that
something had declined to vouch for.

They are deliberately NOT diagnostic codes and must not be answered by
registering them in `diagnostic_codes.py`; `TIMING_PROOF_INCOMPLETE` was
misfiled that way once already, and was only visible to the shape scan at all
because it concatenates a colon.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler.profiler_x86_evidence import (
    build_x86_profiler_packet,
    digest_json,
    X86_INELIGIBILITY_REASONS,
    X86ProfilerPacketError,
    validate_x86_profiler_packet,
    x86_reason_tag,
)

_SOURCE = Path(__file__).resolve().parents[2] / "python/tessera/compiler/profiler_x86_evidence.py"
#: `reasons.append("TAG")` and `reasons.append("TAG:" + detail)`.
_APPENDED = re.compile(r'reasons\.append\(\s*"([A-Z0-9_]+)')


def test_every_produced_tag_is_declared() -> None:
    """The gate the queue item asked for: a twelfth tag cannot appear silently."""
    produced = set(_APPENDED.findall(_SOURCE.read_text()))
    assert produced, "the producer scan matched nothing; its spelling has drifted"
    undeclared = sorted(produced - set(X86_INELIGIBILITY_REASONS))
    assert not undeclared, (
        f"these tags are appended by the producer but declared nowhere: "
        f"{undeclared}. Add each to X86_INELIGIBILITY_REASONS with a meaning — "
        f"a consumer cannot handle a reason it has never heard of, and an "
        f"unhandled reason reads as no reason."
    )


def test_no_declared_tag_is_dead() -> None:
    """A declared tag nobody produces is Decision #29's unconsumed declaration."""
    produced = set(_APPENDED.findall(_SOURCE.read_text()))
    orphans = sorted(set(X86_INELIGIBILITY_REASONS) - produced)
    assert not orphans, (
        f"declared but never produced: {orphans}. Either the producer lost a "
        f"branch or the entry should go; a vocabulary that lists reasons which "
        f"cannot occur misleads exactly like one that omits reasons that can."
    )


def test_every_tag_states_a_meaning() -> None:
    vague = sorted(t for t, why in X86_INELIGIBILITY_REASONS.items() if len(why) < 25)
    assert not vague, (
        f"these tags have no usable meaning: {vague}. The point of the registry "
        f"is that a consumer can act on the reason, not just match the string.")


@pytest.mark.parametrize("tag", sorted(X86_INELIGIBILITY_REASONS))
def test_declared_tag_survives_the_detail_split(tag: str) -> None:
    assert x86_reason_tag(f"{tag}:some,detail") == tag
    assert x86_reason_tag(tag) == tag


def _real_packet() -> dict:
    """A packet the builder itself produced, so the fixture cannot drift."""
    return build_x86_profiler_packet(
        benchmark={
            "schema": "tessera.compiler.e2e_real4.x86_matmul.v1",
            "architecture": "zen5-avx512",
            "rows": [
                {"shape_class": "aligned", "correctness": {"passed": True}},
                {"shape_class": "ragged", "correctness": {"passed": True}},
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
    """Fail CLOSED. A validator that only type-checks is how this went unseen.

    Built from a REAL packet and then spoiled, so the test cannot pass by
    tripping an earlier check -- the first version of it asserted on a stub and
    was rejected for its schema, which would have "passed" for the wrong
    reason had the match been looser.
    """
    packet = _real_packet()
    assert packet["ineligibility_reasons"], "fixture should already be ineligible"
    packet["ineligibility_reasons"] = [*packet["ineligibility_reasons"],
                                       "SOMETHING_NOBODY_DECLARED"]
    packet["packet_sha256"] = digest_json(
        {k: v for k, v in packet.items() if k != "packet_sha256"})
    with pytest.raises(X86ProfilerPacketError, match="unknown x86 ineligibility reason"):
        validate_x86_profiler_packet(packet)


def test_validator_accepts_the_same_packet_unspoiled() -> None:
    """The control: without the bogus tag the very same packet validates."""
    validate_x86_profiler_packet(_real_packet())
