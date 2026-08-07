"""Fail-closed symbol correlation for Linux perf and AMD IBS samples."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SYMBOL_SAMPLING_SCHEMA_VERSION = "tessera.profiler_symbol_sampling.v1"


class SymbolSamplingError(ValueError):
    """Raised when samples cannot be tied to the requested image and symbol."""


@dataclass(frozen=True)
class SymbolRange:
    name: str
    start: int
    size: int
    symbol_type: str

    @property
    def end(self) -> int:
        return self.start + self.size

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "symbol_type": self.symbol_type,
        }


@dataclass(frozen=True)
class PerfSample:
    ip: int
    symbol: str
    dso: str
    period: int
    event: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ip": self.ip,
            "symbol": self.symbol,
            "dso": self.dso,
            "period": self.period,
            "event": self.event,
        }


_NM_LINE = re.compile(r"^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([A-Za-z])\s+(.+?)\s*$")
_PERF_LINE = re.compile(r"^\s*(\S+)\s+([0-9a-fA-F]+)\s+(\S+)\s+\((.+)\)\s+(\d+)\s*$")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_symbol_name(name: str) -> str:
    """Remove perf's offset decoration while preserving the linkage name."""

    return name.split("+0x", 1)[0]


def parse_nm_symbols(text: str) -> list[SymbolRange]:
    symbols: list[SymbolRange] = []
    for line in text.splitlines():
        match = _NM_LINE.match(line)
        if match is None:
            continue
        start, size, symbol_type, name = match.groups()
        parsed_size = int(size, 16)
        if parsed_size <= 0:
            continue
        symbols.append(
            SymbolRange(
                name=name,
                start=int(start, 16),
                size=parsed_size,
                symbol_type=symbol_type,
            )
        )
    return symbols


def parse_perf_script(text: str) -> tuple[list[PerfSample], list[str]]:
    """Parse output requested in perf's canonical event/ip/sym/dso/period order."""

    samples: list[PerfSample] = []
    rejected: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _PERF_LINE.match(line)
        if match is None:
            rejected.append(line)
            continue
        event, ip, symbol, dso, period = match.groups()
        samples.append(
            PerfSample(
                ip=int(ip, 16),
                symbol=normalize_symbol_name(symbol),
                dso=dso,
                period=int(period),
                event=event,
            )
        )
    return samples, rejected


def select_symbol(symbols: Iterable[SymbolRange], name: str) -> SymbolRange:
    matches = [symbol for symbol in symbols if symbol.name == name]
    if len(matches) != 1:
        raise SymbolSamplingError(f"expected exactly one static symbol {name!r}, found {len(matches)}")
    return matches[0]


def ibs_capability(*, cpu: Mapping[str, Any], perf_list_text: str) -> dict[str, Any]:
    vendor = str(cpu.get("vendor_id", ""))
    family = cpu.get("cpu_family")
    flags = set(cpu.get("flags", []))
    catalog = perf_list_text.lower()
    zen5 = vendor == "AuthenticAMD" and family == 26 and "avx512f" in flags
    events = {
        "ibs_op": "ibs_op" in catalog,
        "ibs_fetch": "ibs_fetch" in catalog,
    }
    return {
        "exact_zen5_family": zen5,
        "events": events,
        "eligible": zen5 and any(events.values()),
        "reason": (
            "available"
            if zen5 and any(events.values())
            else "requires AMD family 26 plus advertised ibs_op/ibs_fetch event"
        ),
    }


def build_symbol_sampling_artifact(
    *,
    binary: Path,
    build_id: str,
    target_symbol: SymbolRange,
    event: str,
    samples: Iterable[PerfSample],
    rejected_lines: Iterable[str],
    cpu: Mapping[str, Any],
    ibs: Mapping[str, Any],
    perf_version: str,
    perf_record_command: Iterable[str],
    perf_returncode: int,
    event_map: Mapping[str, Any] | None = None,
    affinity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = binary.resolve()
    sample_rows = list(samples)
    rejected = list(rejected_lines)
    target_dso = resolved.name
    correlated = [
        sample for sample in sample_rows if Path(sample.dso).name == target_dso and sample.symbol == target_symbol.name
    ]
    total_period = sum(sample.period for sample in sample_rows)
    correlated_period = sum(sample.period for sample in correlated)
    payload = {
        "schema": SYMBOL_SAMPLING_SCHEMA_VERSION,
        "binary": str(resolved),
        "binary_sha256": digest_file(resolved),
        "build_id": build_id,
        "target_symbol": target_symbol.to_dict(),
        "correlation": {
            "method": "perf_dso_symbolizer_plus_elf_static_range",
            "aslr_safe": True,
            "dso_basename": target_dso,
            "samples": len(correlated),
            "period": correlated_period,
            "coverage": correlated_period / total_period if total_period else 0.0,
        },
        "event": event,
        "sample_count": len(sample_rows),
        "total_period": total_period,
        "rejected_script_lines": rejected,
        "cpu": dict(cpu),
        "ibs_capability": dict(ibs),
        "perf": {
            "version": perf_version,
            "record_command": list(perf_record_command),
            "returncode": perf_returncode,
        },
        "event_map": dict(event_map) if event_map is not None else None,
        "affinity": dict(affinity or {}),
        "eligible_for_regression": perf_returncode == 0 and bool(correlated),
        "eligible_for_promotion": False,
    }
    validate_symbol_sampling_artifact(payload)
    return payload


def validate_symbol_sampling_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SYMBOL_SAMPLING_SCHEMA_VERSION:
        raise SymbolSamplingError("unsupported symbol-sampling schema")
    for field in ("binary", "binary_sha256", "build_id", "event"):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            raise SymbolSamplingError(f"symbol sample requires {field}")
    symbol = payload.get("target_symbol")
    if not isinstance(symbol, Mapping):
        raise SymbolSamplingError("symbol sample requires target_symbol")
    if not isinstance(symbol.get("name"), str) or not symbol.get("name"):
        raise SymbolSamplingError("target symbol requires name")
    if not all(isinstance(symbol.get(field), int) for field in ("start", "end", "size")):
        raise SymbolSamplingError("target symbol requires integer range")
    if symbol["start"] < 0 or symbol["size"] <= 0 or symbol["end"] != symbol["start"] + symbol["size"]:
        raise SymbolSamplingError("target symbol range is inconsistent")
    correlation = payload.get("correlation")
    if not isinstance(correlation, Mapping):
        raise SymbolSamplingError("symbol sample requires correlation")
    if correlation.get("method") != "perf_dso_symbolizer_plus_elf_static_range":
        raise SymbolSamplingError("unsupported symbol correlation method")
    if correlation.get("aslr_safe") is not True:
        raise SymbolSamplingError("symbol correlation must be ASLR-safe")
    coverage = correlation.get("coverage")
    if not isinstance(coverage, (int, float)) or not 0.0 <= coverage <= 1.0:
        raise SymbolSamplingError("symbol coverage must be in [0, 1]")
    if payload.get("eligible_for_regression") and not correlation.get("samples"):
        raise SymbolSamplingError("regression-eligible sampling needs correlated samples")
    if payload.get("eligible_for_promotion"):
        raise SymbolSamplingError("sampling artifact alone is never promotion-eligible")
    event_map = payload.get("event_map")
    if event_map is not None:
        from .profiler_x86_event_map import validate_x86_event_map

        if not isinstance(event_map, Mapping):
            raise SymbolSamplingError("symbol-sampling event_map must be an object")
        validate_x86_event_map(event_map)


__all__ = [
    "PerfSample",
    "SYMBOL_SAMPLING_SCHEMA_VERSION",
    "SymbolRange",
    "SymbolSamplingError",
    "build_symbol_sampling_artifact",
    "digest_file",
    "ibs_capability",
    "normalize_symbol_name",
    "parse_nm_symbols",
    "parse_perf_script",
    "select_symbol",
    "validate_symbol_sampling_artifact",
]
