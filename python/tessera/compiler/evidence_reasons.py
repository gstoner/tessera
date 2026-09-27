"""Declared promotion-ineligibility vocabularies for evidence packets.

X86-EVIDENCE-VOCAB-1 (docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md).

An evidence packet's ``ineligibility_reasons`` decide whether a measurement may
be promoted, which makes each tag a semantic key under Decision #21a: a reader
that meets a tag it has never heard of must stop, not treat it as no reason.
The x86 packet's eleven tags were the first vocabulary declared (2026-09-20);
the ROCm and NVIDIA device-clock packets and the x86 PMU event map had the same
shape -- bare string literals appended by a producer, enumerated nowhere -- and
are declared here the same way (2026-09-27).

These tags are **not diagnostic codes** and are never registered in
`diagnostic_codes.py`: they classify a measurement, they are not raised at a
user. A packet may *also* carry tags that are registered diagnostics (the
shared device-clock window and witness refusals in `profiler_timing`, NVIDIA's
`DEVICE_CLOCK_PART_UNVALIDATED`); a vocabulary lists those codes explicitly
in ``registered`` and takes their meaning from the registry, so no tag is ever
declared twice. They are named one by one rather than by ``pass_origin``, so a
later diagnostic registered under the same origin cannot silently become an
admissible ineligibility tag (review, 2026-09-27).

A tag may carry a ``:detail`` suffix (``TIMING_PROOF_INCOMPLETE:a,b``); the part
before the first colon is the tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


def reason_tag(reason: str) -> str:
    """The tag half of a reason, discarding any ``:detail`` suffix."""
    return reason.split(":", 1)[0]


@dataclass(frozen=True)
class ReasonVocabulary:
    """One packet family's complete set of ineligibility tags.

    ``declared`` maps each packet-local tag to its meaning. ``registered``
    names the `diagnostic_codes.py` codes that may also appear, one by one.
    ``reserved`` maps a declared tag that no producer emits yet to the reason
    it is kept; the drift test accepts an unproduced tag only there.
    """

    owner: str
    declared: Mapping[str, str]
    registered: tuple[str, ...] = ()
    reserved: Mapping[str, str] = field(default_factory=dict)

    def borrowed(self) -> dict[str, str]:
        """The registered codes this family borrows, with the registry's summary.

        Raises when one is not registered: a vocabulary cannot vouch for a
        meaning nobody declared.
        """
        from .diagnostic_codes import code_lookup

        out: dict[str, str] = {}
        for code in self.registered:
            entry = code_lookup(code)
            if entry is None:
                raise ValueError(f"{self.owner}: {code} is not a registered diagnostic")
            out[code] = entry.summary
        return out

    def meanings(self) -> dict[str, str]:
        """Every tag this family may carry -> what it means."""
        return {**self.borrowed(), **self.declared}

    def unknown(self, reasons: Iterable[str]) -> list[str]:
        known = self.meanings()
        return sorted({reason_tag(r) for r in reasons} - set(known))

    def drift(self, produced: Iterable[str]) -> list[str]:
        """Every way ``produced`` (the tags a producer can emit) and this
        declaration disagree; empty when they match.

        * a produced tag that is not declared (a consumer would refuse it);
        * a declared tag nothing produces, unless ``reserved`` says why
          (Decision #29: a vocabulary listing reasons that cannot occur
          misleads exactly like one omitting reasons that can);
        * a reserved tag that is in fact produced, or reserved but undeclared;
        * a tag declared here that is also a registered diagnostic.
        """
        produced = {reason_tag(p) for p in produced}
        problems: list[str] = []
        meanings = self.meanings()
        for tag in sorted(produced - set(meanings)):
            problems.append(f"{tag}: produced but not declared")
        for tag in sorted(set(self.declared) - produced - set(self.reserved)):
            problems.append(f"{tag}: declared but never produced (reserve it with a reason, or delete it)")
        for tag in sorted(set(self.reserved) & produced):
            problems.append(f"{tag}: reserved but produced; drop the reservation")
        for tag in sorted(set(self.reserved) - set(self.declared)):
            problems.append(f"{tag}: reserved but not declared")
        for tag, why in sorted(self.reserved.items()):
            if len(why) < 25:
                problems.append(f"{tag}: reservation states no usable reason")
        from .diagnostic_codes import all_codes

        for tag in sorted(set(self.declared) & set(all_codes())):
            problems.append(f"{tag}: declared here AND registered as a diagnostic code")
        for tag, meaning in sorted(self.declared.items()):
            if len(meaning) < 25:
                problems.append(f"{tag}: meaning too short to act on")
        return problems

    def require_known(self, reasons: Iterable[str], error: type[Exception],
                      what: str = "ineligibility reason", *, code: str | None = None) -> None:
        """Raise ``error`` naming every undeclared tag (Decision #21a).

        ``code``, when given, is a registered diagnostic the message starts with.
        """
        unknown = self.unknown(reasons)
        if unknown:
            raise error(
                (f"{code}: " if code else "")
                + f"unknown {self.owner} {what}(s) {unknown}; declare each in its "
                f"vocabulary with a meaning, or the packet's readers will treat an "
                f"unknown reason as no reason and promote a result something "
                f"declined to vouch for")


__all__ = ["ReasonVocabulary", "reason_tag"]
