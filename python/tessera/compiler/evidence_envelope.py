"""One evidence envelope every measurement-packet consumer reads through.

EVIDENCE-PACKET-1 (docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md), sync
``EVIDENCE-PACKET-1-2026-09-27``.

Three packet families measure performance today -- the x86 Zen 5 profiler
packet, the ROCm (gfx1151/gfx1201) profiler packet and the NVIDIA sm_120
device-clock packet -- and each grew its own schema and validator. Each
validator re-derives its own eligibility, and that stays: the family owns
*what* its measurement proves. What no family owned was the *shape every
consumer relies on*: which artifact was measured, in which timing domain,
whether that clock was valid, on which environment and revision, from which
sample, and whether promotion is allowed and why not. A consumer had to know
each family's field layout, and the families disagreed on the fail-closed
direction of the same fact: the NVIDIA packet refuses a missing
``worktree_dirty`` (``is not False``), while the ROCm and x86 packets read a
missing one as a clean tree (``if source.get("worktree_dirty")``) -- a
packet that simply omitted the field derived no ``SOURCE_WORKTREE_DIRTY``.

:func:`read_evidence_packet` is the single entry point. It

1. dispatches on the packet's ``schema`` to a registered family, refusing
   any schema no family registers (``EVIDENCE_ENVELOPE_SCHEMA_UNKNOWN``);
2. runs that family's own validator unchanged (its errors propagate as they
   always have, so existing refusals keep their text);
3. projects the packet onto :class:`EvidenceEnvelope`, refusing any missing
   or malformed field instead of defaulting it (Decision #21a;
   ``EVIDENCE_ENVELOPE_INCOMPLETE``);
4. checks the invariants every family must satisfy
   (``EVIDENCE_ENVELOPE_CONTRADICTED``): promotion eligibility holds exactly
   when the packet names no refusal cause, an eligible packet's clock is
   valid, its measured image is bound to its timing sample, its tree was
   clean and every tag is declared in its family's vocabulary.

Nothing here makes a packet *more* promotable than its family validator
says; every rule only refuses. The PMU event map
(``profiler_x86_event_map``) is an input catalog the x86 packet consumes, not
a measurement, and is deliberately not a family. The calibration corpus
(``target_perf``) and the CUDA activity-window calibration
(``profiler_cuda_window``) are measurement records that are not yet
registered; see the plan record for what remains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .evidence_reasons import ReasonVocabulary, reason_tag

EVIDENCE_ENVELOPE_SCHEMA = "tessera.evidence_envelope.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")

#: Execution environments a packet may name. ``bare_metal`` is the only one
#: that is not virtualized; the envelope never infers one from a hostname.
EXECUTION_ENVIRONMENTS = frozenset({"bare_metal", "wsl2", "vm", "container"})


class EvidenceEnvelopeError(ValueError):
    """Raised when a packet cannot be read as trustworthy evidence."""


@dataclass(frozen=True)
class EvidenceEnvelope:
    """The fields every consumer of a measurement packet may rely on.

    ``artifacts`` and ``compiler_identity`` are ``(role, sha256)`` pairs.
    ``refusal_causes`` is every reason the packet may not promote: its
    ineligibility tags plus any family-specific cause that is not a tag (the
    x86 benchmark's own ``retain``/``reject`` verdict). ``diagnostic_gaps``
    are recorded environment gaps that do not block the packet's route.
    """

    family: str
    schema: str
    work_item: str
    architecture: str
    source_commit: str
    worktree_dirty: bool
    execution_environment: str
    artifacts: tuple[tuple[str, str], ...]
    compiler_identity: tuple[tuple[str, str], ...]
    timing_domain: str
    clock_valid: bool
    image_bound_to_sample: bool
    sample_ids: tuple[str, ...]
    admission_route: str
    eligible_for_promotion: bool
    ineligibility_reasons: tuple[str, ...]
    refusal_causes: tuple[str, ...]
    diagnostic_gaps: tuple[str, ...]
    packet_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_ENVELOPE_SCHEMA,
            "family": self.family,
            "packet_schema": self.schema,
            "work_item": self.work_item,
            "architecture": self.architecture,
            "source_commit": self.source_commit,
            "worktree_dirty": self.worktree_dirty,
            "execution_environment": self.execution_environment,
            "artifacts": [list(a) for a in self.artifacts],
            "compiler_identity": [list(c) for c in self.compiler_identity],
            "timing_domain": self.timing_domain,
            "clock_valid": self.clock_valid,
            "image_bound_to_sample": self.image_bound_to_sample,
            "sample_ids": list(self.sample_ids),
            "admission_route": self.admission_route,
            "eligible_for_promotion": self.eligible_for_promotion,
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "refusal_causes": list(self.refusal_causes),
            "diagnostic_gaps": list(self.diagnostic_gaps),
            "packet_sha256": self.packet_sha256,
        }


@dataclass(frozen=True)
class EvidenceFamily:
    """One registered packet family.

    ``project`` maps a packet that its own ``validate`` accepted onto the
    envelope's raw fields; it must raise ``EvidenceEnvelopeError`` (via the
    helpers below) rather than default a missing one.
    """

    name: str
    schemas: tuple[str, ...]
    validate: Callable[[Mapping[str, Any]], None]
    vocabulary: ReasonVocabulary
    routes: frozenset[str]
    timing_domains: frozenset[str]
    project: Callable[[Mapping[str, Any]], dict[str, Any]]


# --------------------------------------------------------------------------
# Field helpers: absence is a refusal, never a default (Decision #21a).
# --------------------------------------------------------------------------

def _incomplete(family: str, what: str) -> EvidenceEnvelopeError:
    return EvidenceEnvelopeError(f"EVIDENCE_ENVELOPE_INCOMPLETE: {family} packet {what}")


def _mapping(obj: Any, key: str, family: str) -> Mapping[str, Any]:
    value = obj.get(key) if isinstance(obj, Mapping) else None
    if not isinstance(value, Mapping):
        raise _incomplete(family, f"has no {key!r} object")
    return value


def _string(obj: Mapping[str, Any], key: str, family: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise _incomplete(family, f"has no {key!r} string")
    return value


def _bool(obj: Mapping[str, Any], key: str, family: str) -> bool:
    value = obj.get(key)
    if type(value) is not bool:
        raise _incomplete(family, f"states {key!r} as {value!r}, not a bool")
    return value


def _digest(value: Any, role: str, family: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _incomplete(family, f"names artifact {role!r} by {value!r}, not a sha256")
    return value


def _tags(obj: Mapping[str, Any], key: str, family: str, *, required: bool = True) -> tuple[str, ...]:
    if key not in obj and not required:
        return ()
    value = obj.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise _incomplete(family, f"states {key!r} as {value!r}, not a list of tags")
    return tuple(value)


# --------------------------------------------------------------------------
# Family projections.
# --------------------------------------------------------------------------

def _gpu_images(payload: Mapping[str, Any], family: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    comparison = _mapping(payload, "instrumentation_comparison", family)
    return (_mapping(comparison, "uninstrumented", family),
            _mapping(comparison, "instrumented", family))


def _gpu_common(payload: Mapping[str, Any], family: str,
                clock_slots: Mapping[str, str]) -> dict[str, Any]:
    source = _mapping(payload, "source", family)
    timing = _mapping(payload, "timing", family)
    clean, probe = _gpu_images(payload, family)
    artifacts = [
        ("image", _digest(clean.get("image_sha256"), "image", family)),
        ("isa", _digest(clean.get("isa_sha256"), "isa", family)),
        ("semantic", _digest(clean.get("semantic_sha256"), "semantic", family)),
        ("instrumented_image", _digest(probe.get("image_sha256"), "instrumented_image", family)),
        ("instrumented_isa", _digest(probe.get("isa_sha256"), "instrumented_isa", family)),
    ]
    sample_digests = _mapping(timing, "artifact_digests", family)
    for role in sorted(sample_digests):
        artifacts.append((f"sample.{role}", _digest(sample_digests[role], f"sample.{role}", family)))
    domain = _string(clean, "clock_source", family)
    slot = clock_slots.get(domain)
    clocks = _mapping(timing, "clocks", family)
    clock = clocks.get(slot) if slot is not None else None
    clock_valid = isinstance(clock, Mapping) and clock.get("valid") is True
    route = _string(payload, "admission_route", family)
    if route == "device_clock_witness":
        device = clocks.get("device_wall_clock_ns")
        clock_valid = clock_valid and isinstance(device, Mapping) and device.get("valid") is True
    reasons = _tags(payload, "ineligibility_reasons", family)
    return {
        "work_item": _string(payload, "work_item", family),
        "architecture": _string(payload, "architecture", family),
        "source_commit": _string(source, "source_commit", family),
        "worktree_dirty": _bool(source, "worktree_dirty", family),
        "execution_environment": _string(timing, "execution_environment", family),
        "artifacts": tuple(artifacts),
        "compiler_identity": (),
        "timing_domain": domain,
        "clock_valid": clock_valid,
        "image_bound_to_sample": clean.get("image_sha256") in set(sample_digests.values()),
        "sample_ids": (_string(timing, "sample_id", family),),
        "admission_route": route,
        "eligible_for_promotion": _bool(payload, "eligible_for_promotion", family),
        "ineligibility_reasons": reasons,
        "refusal_causes": reasons,
        "diagnostic_gaps": _tags(payload, "diagnostic_gaps", family, required=False),
        "packet_sha256": _string(payload, "packet_sha256", family),
    }


_ROCM_CLOCK_SLOTS = {
    "hip_event": "hip_event_ns",
    "device_wall_clock": "device_wall_clock_ns",
    "rocprofiler_activity": "profiler_activity_ns",
}
_NVIDIA_CLOCK_SLOTS = {"cuda_event": "cuda_event_ns"}


def _project_rocm(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _gpu_common(payload, "rocm_profiler", _ROCM_CLOCK_SLOTS)


def _project_nvidia(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = _gpu_common(payload, "nvidia_device_clock", _NVIDIA_CLOCK_SLOTS)
    if out["diagnostic_gaps"]:
        # The NVIDIA device-clock route has no environment split: a gap list
        # there is something its validator never derives.
        raise EvidenceEnvelopeError(
            "EVIDENCE_ENVELOPE_CONTRADICTED: nvidia_device_clock packet carries "
            "diagnostic gaps its route never derives")
    return out


#: x86 clock proofs the tprof probe reports; the tsc_witness route supersedes
#: them with per-row re-derived samples (``profiler_x86_evidence``).
_X86_CLOCK_PROOFS = ("monotonic_raw_valid", "rdtscp_valid", "invariant_tsc",
                     "affinity_stable", "clock_agreement_valid")


def _project_x86(payload: Mapping[str, Any]) -> dict[str, Any]:
    from .profiler_x86_evidence import (
        X86_PROFILER_PACKET_SCHEMA_V1, X86_ROUTE_PROFILER, X86_ROUTE_TSC)
    family = "x86_profiler"
    environment = _mapping(payload, "environment", family)
    benchmark = _mapping(payload, "benchmark", family)
    timing_status = _mapping(payload, "timing_status", family)
    wsl = _bool(environment, "wsl", family)
    virtualized = _bool(environment, "virtualized", family)
    execution = "wsl2" if wsl else ("vm" if virtualized else "bare_metal")
    rows = benchmark.get("rows")
    if not isinstance(rows, list) or not rows:
        raise _incomplete(family, "has no benchmark rows")
    artifacts: list[tuple[str, str]] = []
    compiler: list[tuple[str, str]] = []
    samples: list[str] = []
    bound = True
    for index, row in enumerate(rows):
        compile_record = _mapping(row, "compile", family)
        digests = _mapping(compile_record, "digests", family)
        image = _digest(digests.get("image"), f"rows[{index}].image", family)
        artifacts.append((f"rows[{index}].image", image))
        artifacts.append((f"rows[{index}].production_image", _digest(
            digests.get("production_image"), f"rows[{index}].production_image", family)))
        for key in ("compiler_fingerprint", "toolchain_fingerprint"):
            compiler.append((f"rows[{index}].{key}",
                             _digest(compile_record.get(key), f"rows[{index}].{key}", family)))
        witness = row.get("timing_witness")
        if isinstance(witness, Mapping):
            samples.append(_string(witness, "sample_id", family))
            witnessed = witness.get("artifact_digests")
            if not isinstance(witnessed, Mapping) or image not in set(witnessed.values()):
                bound = False
        else:
            bound = False
    legacy = payload.get("schema") == X86_PROFILER_PACKET_SCHEMA_V1
    # A v1 packet predates admission routes and its validator forbids one; it
    # is read as profiler-correlated, the only route v1 could have used.
    route = X86_ROUTE_PROFILER if legacy else _string(payload, "admission_route", family)
    clock_valid = route == X86_ROUTE_TSC or all(
        timing_status.get(field) is True for field in _X86_CLOCK_PROOFS)
    reasons = _tags(payload, "ineligibility_reasons", family)
    verdict = _string(payload, "verdict", family)
    causes = reasons + (() if verdict == "promote" else (f"benchmark_verdict={verdict}",))
    return {
        "work_item": _string(payload, "work_item", family),
        "architecture": _string(payload, "architecture", family),
        "source_commit": _string(payload, "source_commit", family),
        "worktree_dirty": _bool(environment, "worktree_dirty", family),
        "execution_environment": execution,
        "artifacts": tuple(artifacts),
        "compiler_identity": tuple(compiler),
        "timing_domain": _string(benchmark, "timing_domain", family),
        "clock_valid": clock_valid,
        "image_bound_to_sample": bound,
        "sample_ids": tuple(samples),
        "admission_route": route,
        "eligible_for_promotion": _bool(payload, "eligible_for_promotion", family),
        "ineligibility_reasons": reasons,
        "refusal_causes": causes,
        "diagnostic_gaps": _tags(payload, "diagnostic_gaps", family, required=not legacy),
        "packet_sha256": _string(payload, "packet_sha256", family),
    }


def _families() -> tuple[EvidenceFamily, ...]:
    # Imported lazily: the family modules never import this one, and a
    # consumer that only needs the envelope types should not pay for them.
    from . import profiler_nvidia_evidence as nvidia
    from . import profiler_rocm_evidence as rocm
    from . import profiler_x86_evidence as x86

    return (
        EvidenceFamily(
            name="x86_profiler",
            schemas=(x86.X86_PROFILER_PACKET_SCHEMA_VERSION, x86.X86_PROFILER_PACKET_SCHEMA_V1),
            validate=x86.validate_x86_profiler_packet,
            vocabulary=x86.X86_REASON_VOCABULARY,
            routes=frozenset({x86.X86_ROUTE_PROFILER, x86.X86_ROUTE_TSC}),
            timing_domains=frozenset({"host_wall_operation_total"}),
            project=_project_x86,
        ),
        EvidenceFamily(
            name="rocm_profiler",
            schemas=(rocm.ROCM_PROFILER_PACKET_SCHEMA_VERSION,),
            validate=rocm.validate_rocm_profiler_packet,
            vocabulary=rocm.ROCM_REASON_VOCABULARY,
            routes=frozenset({rocm.ROUTE_PROFILER, rocm.ROUTE_DEVICE_CLOCK}),
            timing_domains=frozenset(_ROCM_CLOCK_SLOTS),
            project=_project_rocm,
        ),
        EvidenceFamily(
            name="nvidia_device_clock",
            schemas=(nvidia.NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION,),
            validate=nvidia.validate_nvidia_device_clock_packet,
            vocabulary=nvidia.NVIDIA_REASON_VOCABULARY,
            routes=frozenset({nvidia.ROUTE_DEVICE_CLOCK}),
            timing_domains=frozenset(_NVIDIA_CLOCK_SLOTS),
            project=_project_nvidia,
        ),
    )


def evidence_families() -> dict[str, EvidenceFamily]:
    """Every registered family by name."""
    return {family.name: family for family in _families()}


def family_for_schema(schema: Any) -> EvidenceFamily | None:
    for family in _families():
        if schema in family.schemas:
            return family
    return None


def _contradicted(family: str, what: str) -> EvidenceEnvelopeError:
    return EvidenceEnvelopeError(f"EVIDENCE_ENVELOPE_CONTRADICTED: {family} packet {what}")


def read_evidence_packet(payload: Mapping[str, Any], *, family: str | None = None) -> EvidenceEnvelope:
    """Validate ``payload`` through its family and the shared envelope.

    ``family``, when given, pins which family the caller accepts: a valid
    packet of another family is refused rather than silently read.
    """
    if not isinstance(payload, Mapping):
        raise EvidenceEnvelopeError(
            "EVIDENCE_ENVELOPE_INCOMPLETE: an evidence packet must be a JSON object")
    schema = payload.get("schema")
    owner = family_for_schema(schema)
    if owner is None:
        raise EvidenceEnvelopeError(
            f"EVIDENCE_ENVELOPE_SCHEMA_UNKNOWN: no evidence family registers schema "
            f"{schema!r}; register it in evidence_envelope before any consumer reads it")
    if family is not None and owner.name != family:
        raise EvidenceEnvelopeError(
            f"EVIDENCE_ENVELOPE_SCHEMA_UNKNOWN: expected a {family} packet, got "
            f"{owner.name} ({schema!r})")
    # The family's own derivation first; its refusals keep their exact text.
    owner.validate(payload)
    raw = owner.project(payload)
    envelope = EvidenceEnvelope(family=owner.name, schema=str(schema), **raw)
    _check_invariants(owner, envelope)
    return envelope


def _check_invariants(family: EvidenceFamily, env: EvidenceEnvelope) -> None:
    name = family.name
    if _COMMIT.fullmatch(env.source_commit) is None:
        raise _incomplete(name, f"names source commit {env.source_commit!r}, not a full sha")
    if _SHA256.fullmatch(env.packet_sha256) is None:
        raise _incomplete(name, "carries no sha256 packet digest")
    if env.execution_environment not in EXECUTION_ENVIRONMENTS:
        raise _incomplete(
            name, f"names execution environment {env.execution_environment!r}; declared: "
            f"{sorted(EXECUTION_ENVIRONMENTS)}")
    if env.timing_domain not in family.timing_domains:
        raise _incomplete(
            name, f"measures in timing domain {env.timing_domain!r}, which the family "
            f"does not declare ({sorted(family.timing_domains)})")
    if env.admission_route not in family.routes:
        raise _contradicted(name, f"names admission route {env.admission_route!r}; "
                                  f"declared: {sorted(family.routes)}")
    if not env.artifacts:
        raise _incomplete(name, "names no measured artifact")
    # The family validators already refuse undeclared tags; repeated here so a
    # family registered later cannot skip it.
    family.vocabulary.require_known(env.ineligibility_reasons, EvidenceEnvelopeError,
                                    code="EVIDENCE_ENVELOPE_CONTRADICTED")
    family.vocabulary.require_known(env.diagnostic_gaps, EvidenceEnvelopeError,
                                    "diagnostic gap", code="EVIDENCE_ENVELOPE_CONTRADICTED")
    if env.eligible_for_promotion == bool(env.refusal_causes):
        raise _contradicted(
            name, f"states eligible_for_promotion={env.eligible_for_promotion} beside "
                  f"refusal causes {list(env.refusal_causes)}; a packet promotes exactly "
                  f"when it names no cause")
    if env.eligible_for_promotion:
        # Each of these is a reason no family can waive: an eligible packet
        # must say which image it measured, on a valid clock, from a clean tree.
        if not env.clock_valid:
            raise _contradicted(name, "is promotion-eligible on an invalid clock")
        if not env.image_bound_to_sample:
            raise _contradicted(
                name, "is promotion-eligible but its measured image is not named by "
                      "its timing sample")
        if env.worktree_dirty:
            raise _contradicted(name, "is promotion-eligible from a dirty worktree")
        if not env.sample_ids:
            raise _contradicted(name, "is promotion-eligible but names no timing sample")
        if any(reason_tag(g) == "SOURCE_WORKTREE_DIRTY" for g in env.diagnostic_gaps):
            raise _contradicted(name, "records a dirty worktree as a non-blocking gap")


def require_promotable(envelope: EvidenceEnvelope) -> None:
    """Raise unless the envelope allows promotion, naming every cause."""
    if not envelope.eligible_for_promotion:
        raise EvidenceEnvelopeError(
            f"{envelope.family} packet {envelope.packet_sha256[:12]} may not promote: "
            + ", ".join(envelope.refusal_causes))


def main(argv: list[str] | None = None) -> int:
    """``python -m tessera.compiler.evidence_envelope PACKET.json...``

    Reads every packet found in each file (bundles nest them) and prints one
    envelope line per packet; exits 1 if any refuses.
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="print full envelopes as JSON")
    args = parser.parse_args(argv)
    failed = 0
    for path in args.paths:
        for packet in iter_packets(json.loads(path.read_text())):
            try:
                env = read_evidence_packet(packet)
            except ValueError as exc:
                failed += 1
                print(f"REFUSED {path}: {exc}")
                continue
            if args.json:
                print(json.dumps(env.to_dict(), sort_keys=True))
            else:
                state = "promotable" if env.eligible_for_promotion else (
                    "retain: " + ",".join(env.refusal_causes))
                print(f"OK {path}: {env.family} {env.architecture} {env.admission_route} "
                      f"{env.timing_domain} {state}")
    return 1 if failed else 0


def iter_packets(obj: Any) -> list[Mapping[str, Any]]:
    """Every object in ``obj`` whose ``schema`` a family registers, outermost
    first; a registered packet's own sub-objects are not searched."""
    found: list[Mapping[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if family_for_schema(node.get("schema")) is not None:
                found.append(node)
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    return found


__all__ = [
    "EVIDENCE_ENVELOPE_SCHEMA",
    "EXECUTION_ENVIRONMENTS",
    "EvidenceEnvelope",
    "EvidenceEnvelopeError",
    "EvidenceFamily",
    "evidence_families",
    "family_for_schema",
    "iter_packets",
    "read_evidence_packet",
    "require_promotable",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
