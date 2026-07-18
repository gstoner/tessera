"""Portable contract checks for the Metal System Trace evidence summarizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "apple_gpu" / "summarize_metal_trace.py"


def _module():
    spec = importlib.util.spec_from_file_location("apple_metal_trace_summary", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(path: Path, rows: str) -> Path:
    path.write_text(f"<trace-query-result><node>{rows}</node></trace-query-result>")
    return path


def test_summary_joins_spills_to_exact_label_and_keeps_occupancy_null(tmp_path):
    module = _module()
    command_buffers = _table(
        tmp_path / "command-buffers.xml",
        """
        <row>
          <start-time>1</start-time><duration>2</duration><event/><gpu/>
          <sentinel/><uint32>1</uint32><duration>2</duration>
          <process id="proc" fmt="python (42)"><pid>42</pid></process>
          <thread/><event/><narrative id="note" fmt='Committed &quot; tessera.softmax.msl.f32 &quot; with 1 encoders'/>
          <narrative/><uint32>1</uint32><depth/><metal-command-buffer-id id="cb">99</metal-command-buffer-id>
        </row>
        <row>
          <start-time>2</start-time><duration>2</duration><event/><gpu/>
          <sentinel/><uint32>1</uint32><duration>2</duration>
          <process ref="proc"/><thread/><event/><narrative ref="note"/>
          <narrative/><uint32>2</uint32><depth/><metal-command-buffer-id ref="cb"/>
        </row>
        """,
    )
    spills = _table(
        tmp_path / "spills.xml",
        """
        <row><start-time>3</start-time><metal-command-buffer-id>99</metal-command-buffer-id>
          <uint64>100</uint64><size-in-bytes>64</size-in-bytes><process ref="spill-proc"/></row>
        <row><start-time>4</start-time><metal-command-buffer-id>99</metal-command-buffer-id>
          <uint64>101</uint64><size-in-bytes ref="bytes"/><process id="spill-proc" fmt="python (42)"/></row>
        <size-in-bytes id="bytes">64</size-in-bytes>
        """,
    )
    compiler = _table(
        tmp_path / "compiler.xml",
        """
        <row><start-time>1</start-time><duration>250</duration><plane/><compiler/>
          <formatted-label fmt="Compile Compute shader ( python (42) )"/>
          <process fmt="python (42)"/><priority/><depth/></row>
        """,
    )
    toc = tmp_path / "toc.xml"
    toc.write_text(
        '<trace-toc><table schema="metal-gpu-counter-profile" '
        'counter-profile="0" shader-profiler="0"/></trace-toc>'
    )

    report = module.summarize(
        command_buffers,
        spills,
        compiler,
        process="python",
        device="apple7",
        toc=toc,
    )

    route = report["routes"]["tessera.softmax.msl.f32"]
    assert route["submission_count"] == 2
    assert route["spill_event_count"] == 2
    assert route["spilled_bytes_total"] == 128
    assert report["spill_summary"]["unattributed_event_count"] == 0
    assert report["compiler_activity"]["Compile Compute shader"]["duration_ns_total"] == 250
    assert report["gpu_counters"]["occupancy"] is None
    assert not report["gpu_counters"]["occupancy_available"]


def test_summary_streams_genuine_compute_occupancy_samples(tmp_path):
    module = _module()
    command_buffers = _table(
        tmp_path / "command-buffers.xml",
        """
        <row><start-time>100</start-time><duration>50</duration><event/><gpu/><sentinel/>
          <uint32>1</uint32><duration>50</duration><process fmt="python (42)"/>
          <thread/><event/><narrative fmt='Committed &quot; tessera.gemm.mps.f32 &quot; with 1 encoders'/>
          <narrative/><uint32>1</uint32><depth/><metal-command-buffer-id>99</metal-command-buffer-id></row>
        """,
    )
    spills = _table(tmp_path / "spills.xml", "")
    compiler = _table(tmp_path / "compiler.xml", "")
    counter_info = _table(
        tmp_path / "counter-info.xml",
        """
        <row><event-time>0</event-time><uint32>24</uint32>
          <gpu-counter-name>Compute Occupancy</gpu-counter-name><uint64>100</uint64>
          <uint64>1</uint64><string>real device counter</string><uint32>0</uint32>
          <string>Percentage</string><uint32>1</uint32><boolean>1</boolean><uint32>10</uint32></row>
        """,
    )
    counter_values = _table(
        tmp_path / "counter-values.xml",
        """
        <row><event-time id="ts">125</event-time><uint32 id="counter">24</uint32>
          <fixed-decimal>37.5</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>
        <row><event-time ref="ts"/><uint32 ref="counter"/>
          <fixed-decimal>0.0</fixed-decimal><uint64>1</uint64><uint32>1</uint32><uint32>0</uint32></row>
        """,
    )
    toc = tmp_path / "toc.xml"
    toc.write_text(
        '<trace-toc><table schema="metal-gpu-counter-profile" '
        'counter-profile="3" shader-profiler="1"/></trace-toc>'
    )

    report = module.summarize(
        command_buffers,
        spills,
        compiler,
        process="python",
        device="apple7",
        toc=toc,
        counter_command_buffers=command_buffers,
        counter_info=counter_info,
        counter_values=counter_values,
    )

    occupancy = report["routes"]["tessera.gemm.mps.f32"]["gpu_counters"][
        "compute_occupancy_percent"
    ]
    assert occupancy["sample_count"] == 2
    assert occupancy["maximum"] == 37.5
    assert occupancy["nonzero_sample_count"] == 1
    assert report["gpu_counters"]["occupancy_available"]
