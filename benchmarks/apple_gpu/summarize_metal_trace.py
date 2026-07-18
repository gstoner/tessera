#!/usr/bin/env python3
"""Summarize exported Instruments Metal System Trace tables.

The input files are XML tables produced by ``xctrace export``.  The summary
keeps only evidence emitted by the requested process and joins spill events to
the exact command-buffer labels retained by the Apple runtime.  Missing GPU
counter/occupancy data stays null; it is never inferred from pipeline limits.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


_COMMITTED_LABEL = re.compile(r'Committed "\s*(.*?)\s*" with')


class TraceTable:
    def __init__(self, path: Path):
        self.root = ET.parse(path).getroot()
        self.ids = {
            element.attrib["id"]: element
            for element in self.root.iter()
            if "id" in element.attrib
        }

    def resolve(self, element: ET.Element) -> ET.Element:
        seen: set[str] = set()
        while "ref" in element.attrib:
            ref = element.attrib["ref"]
            if ref in seen or ref not in self.ids:
                break
            seen.add(ref)
            element = self.ids[ref]
        return element

    def value(self, element: ET.Element) -> str:
        element = self.resolve(element)
        if element.text and element.text.strip():
            return element.text.strip()
        return element.attrib.get("fmt", "")

    def fmt(self, element: ET.Element) -> str:
        return self.resolve(element).attrib.get("fmt", self.value(element))

    def rows(self) -> list[ET.Element]:
        return list(self.root.iter("row"))


def _integer(table: TraceTable, element: ET.Element) -> int:
    value = table.value(element).replace(",", "")
    return int(value or "0", 0)


def _process_matches(table: TraceTable, element: ET.Element, process: str) -> bool:
    return table.fmt(element).startswith(f"{process} (")


def _command_buffers(
    path: Path, process: str
) -> tuple[dict[int, set[str]], dict[str, Any], dict[str, list[tuple[int, int]]]]:
    table = TraceTable(path)
    labels_by_id: dict[int, set[str]] = defaultdict(set)
    submissions_by_label: dict[str, int] = defaultdict(int)
    intervals_by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in table.rows():
        cols = list(row)
        if len(cols) < 15 or not _process_matches(table, cols[7], process):
            continue
        command_buffer_id = _integer(table, cols[14])
        narrative = table.fmt(cols[10])
        match = _COMMITTED_LABEL.search(narrative)
        label = match.group(1).strip() if match else "unlabeled"
        labels_by_id[command_buffer_id].add(label)
        submissions_by_label[label] += 1
        intervals_by_label[label].append((_integer(table, cols[0]), _integer(table, cols[1])))
    return labels_by_id, {
        label: {"submission_count": count}
        for label, count in sorted(submissions_by_label.items())
    }, intervals_by_label


def _spills(
    path: Path, process: str, labels_by_id: dict[int, set[str]], routes: dict[str, Any]
) -> dict[str, Any]:
    table = TraceTable(path)
    total_events = 0
    total_bytes = 0
    unattributed_events = 0
    for row in table.rows():
        cols = list(row)
        if len(cols) < 5 or not _process_matches(table, cols[4], process):
            continue
        command_buffer_id = _integer(table, cols[1])
        spilled_bytes = _integer(table, cols[3])
        labels = labels_by_id.get(command_buffer_id, set())
        if len(labels) != 1:
            label = "unattributed"
            unattributed_events += 1
        else:
            label = next(iter(labels))
        route = routes.setdefault(label, {"submission_count": 0})
        route["spill_event_count"] = route.get("spill_event_count", 0) + 1
        route["spilled_bytes_total"] = route.get("spilled_bytes_total", 0) + spilled_bytes
        route["spilled_bytes_max_event"] = max(
            route.get("spilled_bytes_max_event", 0), spilled_bytes
        )
        total_events += 1
        total_bytes += spilled_bytes

    for route in routes.values():
        route.setdefault("spill_event_count", 0)
        route.setdefault("spilled_bytes_total", 0)
        route.setdefault("spilled_bytes_max_event", 0)
        route["spill_evidence"] = "metal_system_trace_graphics_compiler_spill_events"
    return {
        "event_count": total_events,
        "spilled_bytes_total": total_bytes,
        "unattributed_event_count": unattributed_events,
    }


def _compiler_activity(path: Path, process: str) -> dict[str, Any]:
    table = TraceTable(path)
    by_kind: dict[str, dict[str, int]] = defaultdict(
        lambda: {"event_count": 0, "duration_ns_total": 0, "duration_ns_max": 0}
    )
    for row in table.rows():
        cols = list(row)
        if len(cols) < 6 or not _process_matches(table, cols[5], process):
            continue
        kind = table.fmt(cols[4])
        # Strip the process suffix while retaining the compiler operation.
        kind = re.sub(r"\s*\(\s*[^()]+\(\d+\)\s*\)\s*$", "", kind).strip()
        kind = " ".join(kind.split())
        duration_ns = _integer(table, cols[1])
        entry = by_kind[kind]
        entry["event_count"] += 1
        entry["duration_ns_total"] += duration_ns
        entry["duration_ns_max"] = max(entry["duration_ns_max"], duration_ns)
    return dict(sorted(by_kind.items()))


def _shader_inventory(path: Path | None, process: str) -> dict[str, Any]:
    if path is None:
        return {"available": False, "shader_count": 0, "shaders": []}
    table = TraceTable(path)
    shaders = []
    for row in table.rows():
        cols = list(row)
        if len(cols) < 10 or not _process_matches(table, cols[8], process):
            continue
        shaders.append({
            "name": table.fmt(cols[1]),
            "pipeline": None if cols[3].tag == "sentinel" else table.fmt(cols[3]),
            "shader_type": table.fmt(cols[7]),
        })
    return {"available": True, "shader_count": len(shaders), "shaders": shaders}


def _counter_capabilities(toc: Path | None) -> dict[str, Any]:
    if toc is None:
        return {
            "counter_profile": None,
            "shader_profiler": None,
            "occupancy_available": False,
            "occupancy_unavailable_reason": "trace_toc_not_provided",
        }
    root = ET.parse(toc).getroot()
    tables = [
        element
        for element in root.iter("table")
        if element.attrib.get("schema") == "metal-gpu-counter-profile"
    ]
    profile = tables[0].attrib.get("counter-profile") if tables else None
    shader_profiler = tables[0].attrib.get("shader-profiler") if tables else None
    available = profile not in (None, "0") and shader_profiler not in (None, "0")
    return {
        "counter_profile": int(profile) if profile is not None else None,
        "shader_profiler": int(shader_profiler) if shader_profiler is not None else None,
        "occupancy_available": available,
        "occupancy": None,
        "occupancy_unavailable_reason": (
            None if available else "metal_system_trace_recorded_without_gpu_counter_profile"
        ),
    }


def _counter_id(path: Path, name: str) -> int | None:
    table = TraceTable(path)
    for row in table.rows():
        cols = list(row)
        if len(cols) >= 3 and table.fmt(cols[2]) == name:
            return _integer(table, cols[1])
    return None


def _counter_samples(path: Path, counter_id: int) -> list[tuple[int, float]]:
    """Stream one counter from a potentially very large xctrace XML export."""
    refs: dict[str, str] = {}
    samples: list[tuple[int, float]] = []

    def value(element: ET.Element) -> str:
        ref = element.attrib.get("ref")
        if ref is not None:
            return refs.get(ref, "")
        return (element.text or "").strip() or element.attrib.get("fmt", "")

    for _event, element in ET.iterparse(path, events=("end",)):
        element_id = element.attrib.get("id")
        if element_id is not None:
            refs[element_id] = value(element)
        if element.tag != "row":
            continue
        cols = list(element)
        if len(cols) >= 3:
            try:
                row_counter_id = int(value(cols[1]).replace(",", ""), 0)
                if row_counter_id == counter_id:
                    samples.append((
                        int(value(cols[0]).replace(",", ""), 0),
                        float(value(cols[2]).replace(",", "")),
                    ))
            except ValueError:
                pass
        element.clear()
    return samples


def _attach_compute_occupancy(
    routes: dict[str, Any],
    captures: list[tuple[dict[str, list[tuple[int, int]]], Path, Path]],
) -> dict[str, Any]:
    if not captures:
        return {"available": False, "counter_name": "Compute Occupancy"}
    route_values: dict[str, list[float]] = defaultdict(list)
    total_sample_count = 0
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    counter_ids: set[int] = set()
    for intervals, counter_info, counter_values in captures:
        counter_id = _counter_id(counter_info, "Compute Occupancy")
        if counter_id is None:
            continue
        counter_ids.add(counter_id)
        samples = sorted(_counter_samples(counter_values, counter_id))
        if samples:
            first_timestamp = (samples[0][0] if first_timestamp is None
                               else min(first_timestamp, samples[0][0]))
            last_timestamp = (samples[-1][0] if last_timestamp is None
                              else max(last_timestamp, samples[-1][0]))
        total_sample_count += len(samples)
        timestamps = [timestamp for timestamp, _value in samples]
        for label, route_intervals in intervals.items():
            for start, duration in route_intervals:
                left = bisect.bisect_left(timestamps, start)
                right = bisect.bisect_right(timestamps, start + duration)
                route_values[label].extend(
                    value for _timestamp, value in samples[left:right]
                )
    if not counter_ids:
        return {
            "available": False,
            "counter_name": "Compute Occupancy",
            "unavailable_reason": "counter_not_reported_by_device",
        }
    for label, values in route_values.items():
        if not values:
            continue
        route = routes.setdefault(label, {"submission_count": 0})
        ordered = sorted(values)
        nonzero = [value for value in ordered if value > 0.0]
        p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        route.setdefault("gpu_counters", {})["compute_occupancy_percent"] = {
            "sample_count": len(values),
            "minimum": ordered[0],
            "median": statistics.median(ordered),
            "p95": ordered[p95_index],
            "maximum": ordered[-1],
            "nonzero_sample_count": len(nonzero),
            "nonzero_median": statistics.median(nonzero) if nonzero else None,
            "evidence": "xctrace Metal GPU Counters; samples time-correlated to command-buffer interval",
        }
    return {
        "available": True,
        "counter_ids": sorted(counter_ids),
        "counter_name": "Compute Occupancy",
        "capture_count": len(captures),
        "sample_count": total_sample_count,
        "attributed_sample_count": sum(len(values) for values in route_values.values()),
        "first_sample_timestamp_ns": first_timestamp,
        "last_sample_timestamp_ns": last_timestamp,
    }


def summarize(
    command_buffers: Path,
    spills: Path,
    compiler: Path,
    *,
    process: str,
    device: str,
    shaders: Path | None = None,
    toc: Path | None = None,
    counter_command_buffers: Path | list[Path] | None = None,
    counter_info: Path | list[Path] | None = None,
    counter_values: Path | list[Path] | None = None,
) -> dict[str, Any]:
    labels_by_id, routes, _ = _command_buffers(command_buffers, process)
    spill_summary = _spills(spills, process, labels_by_id, routes)
    def paths(value: Path | list[Path] | None) -> list[Path]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    counter_cbs = paths(counter_command_buffers)
    counter_infos = paths(counter_info)
    counter_value_files = paths(counter_values)
    if not (len(counter_cbs) == len(counter_infos) == len(counter_value_files)):
        raise ValueError("counter command-buffer, info, and value inputs must have equal counts")
    captures = []
    for command_buffer_path, info_path, value_path in zip(
        counter_cbs, counter_infos, counter_value_files
    ):
        _labels, _routes, intervals = _command_buffers(command_buffer_path, process)
        captures.append((intervals, info_path, value_path))
    occupancy = _attach_compute_occupancy(routes, captures)
    counter_capabilities = _counter_capabilities(toc)
    if occupancy.get("available"):
        counter_capabilities.update({
            "occupancy_available": True,
            "occupancy": "route_time_correlated_samples",
            "occupancy_unavailable_reason": None,
            "compute_occupancy": occupancy,
        })
    return {
        "schema_version": 1,
        "evidence_source": (
            "xctrace Metal System Trace + Metal GPU Counters"
            if captures else "xctrace Metal System Trace"
        ),
        "process": process,
        "device_family": device,
        "routes": dict(sorted(routes.items())),
        "spill_summary": spill_summary,
        "compiler_activity": _compiler_activity(compiler, process),
        "shader_inventory": _shader_inventory(shaders, process),
        "gpu_counters": counter_capabilities,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-buffers", type=Path, required=True)
    parser.add_argument("--spills", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--shaders", type=Path)
    parser.add_argument("--toc", type=Path)
    parser.add_argument("--counter-command-buffers", type=Path, action="append")
    parser.add_argument("--counter-info", type=Path, action="append")
    parser.add_argument("--counter-values", type=Path, action="append")
    parser.add_argument("--process", default="python")
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        args.command_buffers,
        args.spills,
        args.compiler,
        process=args.process,
        device=args.device,
        shaders=args.shaders,
        toc=args.toc,
        counter_command_buffers=args.counter_command_buffers,
        counter_info=args.counter_info,
        counter_values=args.counter_values,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
