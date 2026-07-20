#!/usr/bin/env python3
"""Build a clean, structured data archive from AMD RDNA ISA reference PDFs.

Source PDFs (AMD "RDNA Instruction Set Architecture: Reference Guide" series +
the Micro Engine Scheduler spec) have a clean text layer and full bookmark
outlines, so extraction is programmatic rather than OCR/vision based.

For each ISA doc this emits, under <out>/<key>/:
  - meta.json          doc-level metadata (title, pages, source hash)
  - sections.json      the bookmark outline flattened to {id,title,page_start,page_end}
  - sections/*.md      cleaned per-section text (running header/footer stripped)
  - instructions.json  structured instruction DB parsed from "Chapter 16. Instructions"
  - encodings.json     microcode field layouts parsed from "Chapter 15. Microcode Formats"

Plus, at the archive root:
  - MANIFEST.json              every doc + section + page range
  - cross_version/instruction_matrix.{json,md}  per-instruction presence/opcode across RDNA3/3.5/4

The Micro Engine Scheduler doc has no Ch15/16 instruction surface; it gets
meta + sections only.

Re-run with:  python3 docs/reference/isa/rdna/tools/build_archive.py
Requires:     pdftotext (poppler), pypdf.   No network, no API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypdf import PdfReader

# --- configuration ---------------------------------------------------------

# Source PDFs. Edit these paths if the originals move.
SOURCES = {
    "rdna3": {
        "pdf": "rdna3-shader-instruction-set-architecture-feb-2023_0.pdf",
        "name": "RDNA3",
        "title": '"RDNA3" Instruction Set Architecture: Reference Guide',
        "kind": "isa",
    },
    "rdna35": {
        "pdf": "rdna35_instruction_set_architecture.pdf",
        "name": "RDNA3.5",
        "title": '"RDNA3.5" Instruction Set Architecture: Reference Guide',
        "kind": "isa",
    },
    "rdna4": {
        "pdf": "rdna4-instruction-set-architecture.pdf",
        "name": "RDNA4",
        "title": '"RDNA4" Instruction Set Architecture: Reference Guide',
        "kind": "isa",
    },
    "mes": {
        "pdf": "micro_engine_scheduler.pdf",
        "name": "Micro Engine Scheduler",
        "title": "Micro Engine Scheduler Specification",
        "kind": "spec",
    },
}

SOURCE_DIRS = tuple(
    path
    for path in (
        Path(os.environ["TESSERA_RDNA_ISA_SOURCE_DIR"]).expanduser()
        if os.environ.get("TESSERA_RDNA_ISA_SOURCE_DIR")
        else None,
        Path.home() / "projects/AMD_GPU_ISA_DOCS",
        Path.home() / "Downloads",
    )
    if path is not None
)
OUT_DIR = Path(__file__).resolve().parents[1]  # docs/reference/isa/rdna

# --- text extraction helpers ----------------------------------------------


def pdftotext(pdf: Path, first: int, last: int) -> str:
    """Extract a 1-based inclusive page range with column layout preserved."""
    out = subprocess.run(
        ["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s[:80] or "section"


@dataclass
class Section:
    id: str
    title: str
    level: int
    page_start: int  # 1-based
    page_end: int = 0  # 1-based inclusive, filled in later


def read_outline(reader: PdfReader) -> list[Section]:
    """Flatten the bookmark tree to an ordered list with 1-based page starts."""
    flat: list[Section] = []

    def walk(items, level=0):
        for it in items:
            if isinstance(it, list):
                walk(it, level + 1)
                continue
            try:
                pg = reader.get_destination_page_number(it) + 1  # -> 1-based
            except Exception:
                pg = flat[-1].page_start if flat else 1
            flat.append(Section(id="", title=str(it.title).strip(), level=level, page_start=pg))

    walk(reader.outline)
    # Tight page_end: this heading's own pages, up to the next heading that
    # starts on a strictly later page (a child sharing the start page does not
    # extend or truncate it). Containers get their full extent via
    # chapter_full_span() below, used only for wholesale chapter parsing.
    npages = len(reader.pages)
    for i, sec in enumerate(flat):
        end = npages
        for nxt in flat[i + 1 :]:
            if nxt.page_start > sec.page_start:
                end = nxt.page_start - 1
                break
        sec.page_end = max(sec.page_start, end)
    # stable unique ids
    seen: dict[str, int] = {}
    for sec in flat:
        base = slugify(sec.title)
        n = seen.get(base, 0)
        seen[base] = n + 1
        sec.id = base if n == 0 else f"{base}-{n}"
    return flat


def chapter_full_span(flat: list[Section], sec: Section, npages: int) -> tuple[int, int]:
    """Full page extent of a container heading: from its start to just before
    the next heading at the same-or-shallower level (or doc end)."""
    end = npages
    started = False
    for s in flat:
        if s is sec:
            started = True
            continue
        if started and s.level <= sec.level and s.page_start > sec.page_start:
            end = s.page_start - 1
            break
    return sec.page_start, max(sec.page_start, end)


def clean_lines(raw: str, doc_title: str) -> list[str]:
    """Drop the repeating running header and page-number footer lines."""
    header_pat = re.compile(re.escape(doc_title.split(":")[0]))  # '"RDNA4" Instruction Set Architecture'
    footer_pat = re.compile(r"\d+\s+of\s+\d+\s*$")
    out: list[str] = []
    for ln in raw.splitlines():
        s = ln.rstrip()
        stripped = s.strip()
        if not stripped:
            out.append("")
            continue
        if header_pat.fullmatch(stripped) or stripped == doc_title:
            continue
        if footer_pat.search(stripped):
            # footer is either bare "163 of 697" or "16.1. SOP2 Instructions   207 of 697"
            continue
        out.append(s)
    # collapse 3+ blank lines to 1
    collapsed: list[str] = []
    blank = 0
    for ln in out:
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                collapsed.append("")
        else:
            blank = 0
            collapsed.append(ln)
    # trim leading/trailing blanks
    while collapsed and collapsed[0] == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return collapsed


# --- instruction reference parser (Chapter 16) -----------------------------

# An instruction entry header looks like:  "S_ADD_CO_U32                 0"
# (UPPER_SNAKE name, >=2 spaces of right-padding, then an integer opcode).
INSTR_HEADER = re.compile(r"^([A-Z][A-Z0-9_]+)\s{2,}(\d{1,4})\s*$")
# Family subsection heading: "16.1. SOP2 Instructions"
FAMILY_HEADING = re.compile(r"^16\.(\d+)\.\s+(.+?)\s+Instructions\s*$")


@dataclass
class Instruction:
    name: str
    opcode: int
    family: str
    description: str
    pseudocode: str
    notes: str
    page: int


def parse_instructions(reader: PdfReader, pdf: Path, flat: list[Section],
                       sec_index: dict[str, Section], doc_title: str) -> list[Instruction]:
    chap = sec_index.get("chapter-16-instructions")
    if chap is None:
        return []
    first, last = chapter_full_span(flat, chap, len(reader.pages))
    raw = pdftotext(pdf, first, last)
    lines = clean_lines(raw, doc_title)

    instrs: list[Instruction] = []
    family = "?"
    cur: Instruction | None = None
    body: list[str] = []

    def flush():
        nonlocal cur, body
        if cur is None:
            return
        cur.description, cur.pseudocode, cur.notes = split_body(body)
        instrs.append(cur)
        cur, body = None, []

    for ln in lines:
        fam = FAMILY_HEADING.match(ln.strip())
        if fam:
            flush()
            family = fam.group(2).strip()
            continue
        m = INSTR_HEADER.match(ln)
        if m and looks_like_opcode_entry(m.group(1)):
            flush()
            cur = Instruction(
                name=m.group(1), opcode=int(m.group(2)), family=family,
                description="", pseudocode="", notes="", page=chap.page_start,
            )
            continue
        if cur is not None:
            body.append(ln)
    flush()
    return instrs


def looks_like_opcode_entry(name: str) -> bool:
    # Filter section-number/reference false positives; real opcodes have a prefix.
    if len(name) < 3:
        return False
    return bool(re.match(r"^(S_|V_|DS_|BUFFER_|IMAGE_|TBUFFER_|FLAT_|GLOBAL_|SCRATCH_|EXP|"
                         r"DUAL_|GET_|SAMPLE)", name)) or name.isupper()


def split_body(body: list[str]) -> tuple[str, str, str]:
    """Separate an entry body into (description, pseudocode, notes)."""
    # Find 'Notes' marker (its own line).
    notes_idx = next((i for i, l in enumerate(body) if l.strip() == "Notes"), None)
    notes = ""
    if notes_idx is not None:
        notes = "\n".join(x.strip() for x in body[notes_idx + 1 :]).strip()
        body = body[:notes_idx]
    # Pseudocode = the trailing indented block (lines starting with >=2 spaces),
    # which in these docs always follows the prose description.
    desc_lines, code_lines = [], []
    in_code = False
    for ln in body:
        is_code = bool(ln) and ln.startswith("  ")
        if is_code:
            in_code = True
            code_lines.append(ln.rstrip())
        elif in_code and ln.strip() == "":
            code_lines.append("")  # keep internal blank lines of a code block
        else:
            if in_code and ln.strip():
                # prose resumed after code (rare) -> treat as part of notes-ish tail
                code_lines.append(ln.rstrip())
            else:
                desc_lines.append(ln.strip())
    description = " ".join(x for x in desc_lines if x).strip()
    pseudocode = "\n".join(code_lines).strip("\n")
    return description, pseudocode, notes


# --- microcode format parser (Chapter 15) ----------------------------------

# Format subsection: "15.1.1. SOP2"  /  "15.3.4. VOP3"
FORMAT_HEADING = re.compile(r"^15\.\d+\.\d+\.\s+([A-Z][A-Z0-9_]+)\s*$")
# Field row: "SSRC0        [7:0]      Source 0. First operand..."
FIELD_ROW = re.compile(r"^([A-Z][A-Z0-9_]+)\s+\[(\d+):(\d+)\]\s+(.*\S)\s*$")
FIELD_ROW_1BIT = re.compile(r"^([A-Z][A-Z0-9_]+)\s+\[(\d+)\]\s+(.*\S)\s*$")


@dataclass
class EncFormat:
    format: str
    page: int
    fields: list[dict] = field(default_factory=list)


def parse_encodings(reader: PdfReader, pdf: Path, flat: list[Section],
                    sec_index: dict[str, Section], doc_title: str) -> list[EncFormat]:
    chap = sec_index.get("chapter-15-microcode-formats")
    if chap is None:
        return []
    first, last = chapter_full_span(flat, chap, len(reader.pages))
    raw = pdftotext(pdf, first, last)
    lines = clean_lines(raw, doc_title)

    formats: list[EncFormat] = []
    cur: EncFormat | None = None
    for ln in lines:
        h = FORMAT_HEADING.match(ln.strip())
        if h:
            cur = EncFormat(format=h.group(1), page=chap.page_start)
            formats.append(cur)
            continue
        if cur is None:
            continue
        m = FIELD_ROW.match(ln)
        if m:
            cur.fields.append({"name": m.group(1), "bits": f"[{m.group(2)}:{m.group(3)}]",
                               "desc": m.group(4).strip()})
            continue
        m = FIELD_ROW_1BIT.match(ln)
        if m:
            cur.fields.append({"name": m.group(1), "bits": f"[{m.group(2)}]",
                               "desc": m.group(3).strip()})
    # keep only formats that actually had a field table
    return [f for f in formats if f.fields]


# --- Micro Engine Scheduler API parser -------------------------------------

OPCODE_ENUM = re.compile(r"^(MES_SCH_API_[A-Z0-9_]+)\s*=\s*(\d+)\s*,?\s*$")
UNION_NAME = re.compile(r"\bunion\s+(MESAPI__[A-Z0-9_]+)")
# section titles that are API commands
API_TITLE = re.compile(r"^(MES_SCH_API_|MESAPI_MISC__|MES_API_QUERY_MES__)[A-Z0-9_]+$")


def parse_mes_api(reader: PdfReader, pdf: Path, flat: list[Section],
                  sec_index: dict[str, Section], doc_title: str) -> dict:
    """Extract the MES command surface: the opcode enum + one record per API
    command (purpose, page span, C union name, KMD->FW direction)."""
    api_chap = sec_index.get("mes-api")
    opcodes: dict[str, int] = {}
    if api_chap is not None:
        first, last = chapter_full_span(flat, api_chap, len(reader.pages))
        for ln in clean_lines(pdftotext(pdf, first, last), doc_title):
            m = OPCODE_ENUM.match(ln.strip())
            if m and not m.group(1).endswith("_MAX"):
                opcodes.setdefault(m.group(1), int(m.group(2)))

    code_prefixes = ("union", "struct", "enum", "{", "}", "uint", "int", "//", "•")
    commands = []
    for sec in flat:
        title = sec.title.strip()
        if not API_TITLE.match(title):
            continue
        lines = clean_lines(pdftotext(pdf, sec.page_start, sec.page_end), doc_title)
        # Anchor to the heading repeated inside the body so we don't pick up the
        # previous section's trailing content (tight spans can share a page).
        start = next((i for i, l in enumerate(lines) if l.strip() == title), 0)
        body = lines[start + 1 :]
        purpose, union = "", ""
        for ln in body:
            if not union:
                um = UNION_NAME.search(ln)
                if um:
                    union = um.group(1)
            s = ln.strip()
            if s and not purpose and not s.startswith(code_prefixes) and not s.endswith((";", "{", "}")):
                purpose = s
        commands.append({
            "name": sec.title.strip(),
            "opcode": opcodes.get(sec.title.strip()),
            "union": union,
            "purpose": purpose,
            "page_start": sec.page_start,
            "page_end": sec.page_end,
            "file": f"sections/{sec.id}.md",
        })
    return {"frame_size_dwords": 64, "opcodes": opcodes, "commands": commands}


# --- driver ----------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def find_source_pdf(filename: str, source_dirs=None) -> Path:
    """Resolve one source independently so partial preferred mirrors work."""
    directories = SOURCE_DIRS if source_dirs is None else tuple(source_dirs)
    for directory in directories:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(Path(directory)) for directory in directories)
    raise FileNotFoundError(f"{filename} not found in: {searched}")


def build_doc(key: str, cfg: dict) -> dict:
    from pypdf import PdfReader

    try:
        pdf = find_source_pdf(cfg["pdf"])
    except FileNotFoundError as exc:
        print(f"  !! missing source: {exc}", file=sys.stderr)
        return {}
    reader = PdfReader(str(pdf))
    sections = read_outline(reader)
    sec_index = {s.id: s for s in sections}
    doc_out = OUT_DIR / key
    (doc_out / "sections").mkdir(parents=True, exist_ok=True)

    # per-section markdown (leaf-ish: write every section's own page span)
    sec_records = []
    for s in sections:
        rec = {"id": s.id, "title": s.title, "level": s.level,
               "page_start": s.page_start, "page_end": s.page_end,
               "file": f"sections/{s.id}.md"}
        sec_records.append(rec)
        raw = pdftotext(pdf, s.page_start, s.page_end)
        text = "\n".join(clean_lines(raw, cfg["title"]))
        md = f"# {s.title}\n\n> {cfg['name']} ISA — pages {s.page_start}–{s.page_end}\n\n{text}\n"
        (doc_out / "sections" / f"{s.id}.md").write_text(md)

    meta = {"key": key, "name": cfg["name"], "title": cfg["title"], "kind": cfg["kind"],
            "source_pdf": cfg["pdf"], "pages": len(reader.pages),
            "sha256": sha256(pdf), "section_count": len(sections)}
    (doc_out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (doc_out / "sections.json").write_text(json.dumps(sec_records, indent=2) + "\n")

    result = {"meta": meta, "sections": sec_records, "instructions": [], "encodings": []}

    if cfg["kind"] == "isa":
        instrs = parse_instructions(reader, pdf, sections, sec_index, cfg["title"])
        encs = parse_encodings(reader, pdf, sections, sec_index, cfg["title"])
        (doc_out / "instructions.json").write_text(
            json.dumps([asdict(i) for i in instrs], indent=2) + "\n")
        (doc_out / "encodings.json").write_text(
            json.dumps([asdict(e) for e in encs], indent=2) + "\n")
        result["instructions"] = instrs
        result["encodings"] = encs
        print(f"  {cfg['name']}: {len(sections)} sections, "
              f"{len(instrs)} instructions, {len(encs)} encoding formats")
    elif key == "mes":
        api = parse_mes_api(reader, pdf, sections, sec_index, cfg["title"])
        (doc_out / "api.json").write_text(json.dumps(api, indent=2) + "\n")
        result["mes_api"] = api
        print(f"  {cfg['name']}: {len(sections)} sections, "
              f"{len(api['opcodes'])} opcodes, {len(api['commands'])} API commands")
    else:
        print(f"  {cfg['name']}: {len(sections)} sections")
    return result


def build_cross_version(results: dict[str, dict]) -> None:
    isa_keys = [k for k in ("rdna3", "rdna35", "rdna4") if results.get(k)]
    by_name: dict[str, dict] = {}
    for k in isa_keys:
        for ins in results[k]["instructions"]:
            row = by_name.setdefault(ins.name, {"name": ins.name, "family": ins.family})
            row[k] = ins.opcode
    rows = sorted(by_name.values(), key=lambda r: (r.get("family", ""), r["name"]))

    cross = OUT_DIR / "cross_version"
    cross.mkdir(exist_ok=True)
    (cross / "instruction_matrix.json").write_text(json.dumps(rows, indent=2) + "\n")

    def cell(r, k):
        return str(r[k]) if k in r else "—"

    lines = ["# RDNA Instruction Cross-Version Matrix",
             "",
             "Opcode number per ISA version (`—` = not present in that version).",
             "Generated by `tools/build_archive.py`; do not hand-edit.",
             "",
             "| Instruction | Family | RDNA3 | RDNA3.5 | RDNA4 |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['name']}` | {r.get('family','')} | "
                     f"{cell(r,'rdna3')} | {cell(r,'rdna35')} | {cell(r,'rdna4')} |")
    only4 = [r["name"] for r in rows if "rdna4" in r and "rdna3" not in r]
    gone = [r["name"] for r in rows if "rdna3" in r and "rdna4" not in r]
    lines += ["", f"**New in RDNA4 (vs RDNA3):** {len(only4)}",
              f"**In RDNA3 but not RDNA4:** {len(gone)}", ""]
    (cross / "instruction_matrix.md").write_text("\n".join(lines) + "\n")
    print(f"  cross-version: {len(rows)} distinct instructions "
          f"({len(only4)} new in RDNA4, {len(gone)} dropped)")


def main() -> None:
    print("Building RDNA ISA archive...")
    results: dict[str, dict] = {}
    for key, cfg in SOURCES.items():
        print(f"- {key}")
        results[key] = build_doc(key, cfg)

    build_cross_version(results)

    manifest = {"docs": []}
    for key, cfg in SOURCES.items():
        r = results.get(key)
        if not r:
            continue
        manifest["docs"].append({
            "key": key, "name": cfg["name"], "kind": cfg["kind"],
            "source_pdf": cfg["pdf"], "pages": r["meta"]["pages"],
            "sections": len(r["sections"]),
            "instructions": len(r["instructions"]),
            "encodings": len(r["encodings"]),
            "api_commands": len(r.get("mes_api", {}).get("commands", [])),
            "paths": {"meta": f"{key}/meta.json", "sections": f"{key}/sections.json",
                      "instructions": f"{key}/instructions.json" if cfg["kind"] == "isa" else None,
                      "encodings": f"{key}/encodings.json" if cfg["kind"] == "isa" else None,
                      "api": f"{key}/api.json" if key == "mes" else None},
        })
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Done. MANIFEST.json written.")


if __name__ == "__main__":
    main()
