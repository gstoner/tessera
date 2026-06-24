# RDNA ISA Data Archive

A clean, structured, version-controlled extraction of AMD's RDNA Instruction Set
Architecture reference guides (RDNA3 / RDNA3.5 / RDNA4) and the Micro Engine
Scheduler spec, built for Tessera ROCm-backend compiler development.

The source PDFs have a clean text layer and full bookmark outlines, so this is a
**programmatic extraction** (poppler `pdftotext` + `pypdf`) — not OCR/vision.
Everything here is regenerable from the PDFs by one script; treat the JSON as
machine truth and the markdown as human-readable mirror.

## What's here

```
docs/reference/isa/rdna/
├── README.md                       ← you are here
├── MANIFEST.json                   ← every doc + page/section/instruction counts + file paths
├── tools/build_archive.py          ← the regenerator (no network, no API)
├── cross_version/
│   ├── instruction_matrix.json     ← per-instruction opcode across RDNA3 / 3.5 / 4
│   └── instruction_matrix.md       ← same, as a readable table + new/dropped deltas
├── rdna3/  rdna35/  rdna4/         ← one dir per ISA doc
│   ├── meta.json                   ← title, page count, source sha256, section count
│   ├── sections.json               ← flattened bookmark outline (id, title, level, page range, file)
│   ├── sections/*.md               ← cleaned per-section text (running header/footer stripped)
│   ├── instructions.json           ← structured instruction DB (see schema below)
│   └── encodings.json              ← microcode bit-field layouts (Ch 15)
└── mes/                            ← Micro Engine Scheduler (GPU work scheduler)
    ├── SCHEDULER_OVERVIEW.md       ← synthesized "how the RDNA GPU scheduler works"
    ├── api.json                    ← MES command surface: 64-DWORD frame, opcode enum, 29 commands
    ├── meta.json / sections.json
    └── sections/*.md
```

## Coverage

| Doc | Pages | Sections | Instructions | Encoding formats |
|-----|------:|---------:|-------------:|-----------------:|
| RDNA3   | 609 | 236 | 1464 | 23 |
| RDNA3.5 | 653 | 238 | 1522 | 23 |
| RDNA4   | 707 | 255 | 1566 | 20 |
| Micro Engine Scheduler | 54 | 54 | — | — |

The Micro Engine Scheduler doc has no Ch 15/16 instruction surface; instead it
yields [`mes/SCHEDULER_OVERVIEW.md`](mes/SCHEDULER_OVERVIEW.md) (how the on-GPU
work scheduler maps application queues onto hardware queues + enforces priority)
and [`mes/api.json`](mes/api.json) (the 19 scheduling opcodes + 29 driver↔FW API
commands, each with purpose / page / C union).

Cross-version matrix: **1387 distinct instructions**, 245 new in RDNA4 vs RDNA3,
123 present in RDNA3 but not RDNA4 (some of those 123 are **renames**, e.g.
`GLOBAL_ATOMIC_MAX_F32` → `GLOBAL_ATOMIC_MAX_NUM_F32` — see caveats).

## Schemas

### `<doc>/instructions.json` — list of:

```json
{
  "name": "V_WMMA_F32_16X16X16_F16",
  "opcode": 64,
  "family": "VOP3P",
  "description": "Wave matrix-multiply-accumulate ...",
  "pseudocode": "  D0.f32 = ...",
  "notes": "",
  "page": 215
}
```

- `opcode` — the per-family opcode number (right-hand value next to the name in Ch 16).
- `family` — encoding family (SOP2, VOP1, VOP3 & VOP3SD, VOP3P, VDS, VBUFFER, FLAT/Scratch/Global, …). Cross-reference `encodings.json` for the bit layout of that family.
- `description` / `pseudocode` / `notes` — extracted prose, the indented HDL-ish semantics block, and any "Notes" block, separated from the entry body.
- `page` — approximate (the start page of Chapter 16, not the exact entry page).

### `<doc>/encodings.json` — list of:

```json
{
  "format": "VOP3P",
  "page": 172,
  "fields": [ {"name": "SRC0", "bits": "[8:0]", "desc": "..."}, ... ]
}
```

Bit-field tables from "Chapter 15. Microcode Formats" — what each instruction
field means and its operand encoding (SGPR/VGPR ranges, inline constants, etc.).

### `cross_version/instruction_matrix.json` — list of:

```json
{ "name": "V_WMMA_F32_16X16X16_FP8_FP8", "family": "VOP3P", "rdna4": 70 }
```

Keys `rdna3` / `rdna35` / `rdna4` are present only where that instruction exists,
with its opcode number as the value. Missing key = not in that ISA version.

### `mes/api.json` — the scheduler command surface

```json
{
  "frame_size_dwords": 64,
  "opcodes": { "MES_SCH_API_ADD_QUEUE": 2, ... },        // MES_SCH_API_OPCODE enum (0-18)
  "commands": [ {"name": "MES_SCH_API_ADD_QUEUE", "opcode": 2,
                 "union": "MESAPI__ADD_QUEUE", "purpose": "...",
                 "page_start": 22, "page_end": 24, "file": "sections/..."} ]
}
```

`opcode` is null for `MESAPI_MISC__*` sub-commands and `MES_API_QUERY_MES__*`
(they dispatch under opcode 14 `MISC` / 11 `QUERY_SCHEDULER_STATUS`, not the
top-level enum). See [`mes/SCHEDULER_OVERVIEW.md`](mes/SCHEDULER_OVERVIEW.md) for
the prose model the commands operate on.

## Using it during compiler work

```bash
# Which WMMA/SWMMAC ops does each RDNA gen expose, with opcodes?
python3 -c "import json;[print(r['name'],r.get('rdna3','-'),r.get('rdna35','-'),r.get('rdna4','-')) for r in json.load(open('docs/reference/isa/rdna/cross_version/instruction_matrix.json')) if 'WMMA' in r['name']]"

# Full semantics for one op (RDNA4)
python3 -c "import json;print(next(i for i in json.load(open('docs/reference/isa/rdna/rdna4/instructions.json')) if i['name']=='V_FMA_F32')['pseudocode'])"

# Bit layout of the VOP3P encoding (RDNA4)
python3 -c "import json;print(*[f for f in json.load(open('docs/reference/isa/rdna/rdna4/encodings.json')) if f['format']=='VOP3P'],sep='\n')"
```

For prose (addressing rules, hazards, LDS semantics, scheduling), grep
`*/sections/*.md` — each file is one bookmark section with the running
header/footer removed. `sections.json` maps section id → title → page range.

**gfx1151 (Strix Halo) note:** that part is RDNA3.5. Use `rdna35/` for what it
actually supports — notably it has WMMA F16/BF16/IU8/IU4 but **no** FP8/BF8 WMMA
(those are RDNA4-only, opcodes 70-74 / SWMMAC 87-90). The cross-version matrix is
the fastest way to confirm an op exists before emitting it.

## Caveats

- **Renames read as add+drop.** The cross-version delta keys on instruction name,
  so an ISA rename appears as one dropped + one new. Skim `instruction_matrix.md`
  before treating a "dropped" op as truly gone.
- **`encodings.json` is best-effort.** Multi-table sections (e.g. VBUFFER) can
  over-group fields from sibling sub-formats into one entry. The field
  name/bits/desc values are faithful; the grouping boundary may not be.
- **`page` on instructions is the chapter start**, not the exact page (entry-exact
  pages would need per-page re-extraction; not worth it for lookups).
- **`mes/api.json` `union` is best-effort.** Opcodes and purposes are exact; a few
  single-page command sections share a page boundary and pick up a neighbour's
  `union MESAPI__…` name. Trust `opcode`/`purpose`/`page`; confirm the exact union
  in the linked section file.
- **`pseudocode` is AMD's HDL-style notation verbatim** (`D0.f32`, `S0.u32[31]`,
  `1'1U`), not compilable — it's the semantic reference.

## Regenerate

```bash
python3 docs/reference/isa/rdna/tools/build_archive.py    # ~20s, needs pdftotext + pypdf
```

Source PDFs are read from `~/Downloads/` (paths in `SOURCES` at the top of the
script). `meta.json` records each source's sha256 so you can detect if an
upstream PDF changed. The PDFs themselves are **not** committed — only the
extracted archive.

## Provenance

| Key | Source PDF | sha256 in |
|-----|-----------|-----------|
| rdna3   | `rdna3-shader-instruction-set-architecture-feb-2023_0.pdf` | `rdna3/meta.json` |
| rdna35  | `rdna35_instruction_set_architecture.pdf` | `rdna35/meta.json` |
| rdna4   | `rdna4-instruction-set-architecture.pdf` | `rdna4/meta.json` |
| mes     | `micro_engine_scheduler.pdf` | `mes/meta.json` |

AMD RDNA ISA Reference Guides © Advanced Micro Devices, Inc. Extracted here for
engineering reference under fair use; redistribute per AMD's terms.
