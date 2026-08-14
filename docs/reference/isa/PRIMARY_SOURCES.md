---
last_updated: 2026-08-13
audit_role: reference
---

# AMD GPU ISA primary sources — assessment for agents

An assessment of the three AMD ISA reference PDFs held outside the repo at
`~/projects/AMD_GPU_ISA_DOCS/`: **which document answers which question**, and
the facts most likely to be assumed wrong. Companion index with per-document
chapter/page tables and full instruction inventories:
[`PRIMARY_SOURCES_INDEX.md`](PRIMARY_SOURCES_INDEX.md).

**Relationship to [`isa/rdna/`](rdna/README.md).** That directory is the
*extracted archive* — structured JSON/markdown produced from the RDNA PDFs by
`rdna/tools/build_archive.py`, and it is the thing to grep for
"does this opcode exist on my target". This file is different in kind: a
**cross-architecture assessment**, and it is the only place **CDNA 5** appears
at all — the archive covers RDNA3 / RDNA3.5 / RDNA4 and the MES spec only.
Adding CDNA 5 to the extractor is a scoped follow-up (see the end of this file).

Every claim below was derived from the PDFs on 2026-08-13; none is from memory.
The commands are at the bottom so any number can be re-derived.

| file | architecture | pages | PDF date |
|---|---|---:|---|
| `rdna35_instruction_set_architecture.pdf` | RDNA 3.5 | 653 | 2024-09-11 |
| `rdna4-instruction-set-architecture.pdf` | RDNA 4 | 707 | 2025-04-08 |
| `amd-instinct-cdna5-instruction-set-architecture.pdf` | CDNA 5 | 832 | 2026-07-29 |

---

## Read this before using these docs

**1. None of the three names a `gfx` target or a product number.** Searching all
three for `gfx[0-9]+` and for `MI###` returns **zero** hits. They are
*architecture* references. The architecture ↔ target mapping must come from
somewhere else (for Tessera: `docs/reference/isa/rdna/`, `rocminfo`, or the LLVM
target definitions). Do not expect to resolve "is this gfx1151?" from these PDFs.

**2. CDNA 5 has no MFMA instructions at all.** Zero occurrences of `MFMA` or
`SMFMAC` in the entire 832-page document. CDNA 5 uses the **`V_WMMA_*` /
`V_SWMMAC_*`** mnemonics — the RDNA naming. If your code branches on
"CDNA ⇒ MFMA", that branch is wrong for CDNA 5.

**3. CDNA 5 is wave32.** Quoting Chapter 1, Introduction (PDF p.12) — an
unqualified, document-wide statement, not a mode note:

> "This document may make reference to 'wave64' but this device supports only
> wave32."

This breaks the common assumption that CDNA is wave64 and RDNA is wave32.

**Concretely for Tessera.** `ROCMFragmentLayout.h:192` ends the CDNA branch with

```cpp
return make(family, familyName, "mfma", 64, inputElements, inputFormat, ...);
```

i.e. the CDNA families are hardwired to matrix-op `"mfma"` and `waveSize = 64`.
CDNA 5 is **neither**. Note this is not a live bug: that branch is reached only
by `gfx90a` / `gfx940` / `gfx942` / `gfx950`, so a CDNA 5 target resolves to
`std::nullopt` today and fails closed, which is correct. The point is about
*shape* — adding CDNA 5 is **not** "append an arch to the list", because the
`family ⇒ (matrixOp, waveSize)` mapping the function is built around does not
hold for it. Anyone scoping CDNA 5 support should budget for changing that
mapping, and `!tile.fragment`'s `family` parameter carries the same assumption
(Decision on `family` selecting a physical register ABI: wave 32 RDNA/WMMA vs
wave 64 CDNA/MFMA).

**4. CDNA 5 is compute-only.** It has **no** `IMAGE_*` instructions (0 distinct,
versus 92 in RDNA 3.5 and 108 in RDNA 4) and no export chapter. RDNA has both.
Conversely CDNA 5 is the only one with **Workgroup Clusters** (§2.3) and an
**Error Correction Codes (ECC)** chapter.

---

## Which document answers which question

| question | go to |
|---|---|
| Does target X have an FP8 matrix instruction? | the matrix inventory in [`INDEX.md`](INDEX.md) |
| Matrix fragment shape / K depth for a dtype | Instructions chapter (RDNA 3.5 ch.16, RDNA 4 ch.16, CDNA 5 ch.15) |
| Microcode bit fields for an encoding | "Microcode Formats" (RDNA 3.5 ch.15, RDNA 4 ch.15, CDNA 5 ch.14) |
| Wave size, EXEC/VCC width, VGPR allocation | ch.3 Wave State |
| LDS semantics, sizes, bank behaviour | RDNA "Data Share Operations"; CDNA 5 "Local Data Share Operations" |
| `s_waitcnt` / dependency rules | ch.5 "Data Dependency Resolution" |
| Buffer vs global vs flat addressing | ch.9 / ch.10-11 |
| Float atomics | RDNA 3.5 ch.13, RDNA 4 ch.13, CDNA 5 ch.12 |
| Texture/image ops, render export | RDNA only — **not in CDNA 5** |
| Workgroup clusters, ECC | CDNA 5 only |
| Matrix-op hazards / co-issue stalls | search `XDL` (CDNA 5), not `MFMA` |
| Is an SGPR/constant legal as a matrix operand? | VOP3P section, e.g. RDNA 3.5 p.77 |

---

## Matrix-core support, side by side

Counts are distinct `V_WMMA_*` / `V_SWMMAC_*` mnemonics; the full lists are in
[`INDEX.md`](INDEX.md).

| | RDNA 3.5 | RDNA 4 | CDNA 5 |
|---|---|---|---|
| distinct matrix mnemonics | 6 | 22 | 50 |
| dense `WMMA` | ✅ | ✅ | ✅ |
| sparse `SWMMAC` | ❌ | ✅ | ✅ |
| F16 / BF16 | ✅ | ✅ | ✅ |
| IU8 / IU4 | ✅ | ✅ | IU8 only — int4 exists, but as `V_DOT8_I32_IU4`, not WMMA |
| FP8 / BF8 | ❌ | ✅ | ✅ |
| FP4 / F8F6F4 mixed | ❌ | ❌ | ✅ |
| microscaling (`WMMA_SCALE*`) | ❌ | ❌ | ✅ |
| K depth at 16×16 | 16 | 16, 32 | 32, 64, 128 |
| non-square tile | ❌ | ❌ | ✅ `32X16X128_F4` |

Three consequences worth stating plainly:

* **RDNA 3.5 has no FP8 matrix path.** Its six mnemonics are F16/BF16 output
  from F16/BF16 input, plus `IU4`/`IU8`. Anything asking for an FP8 WMMA on
  RDNA 3.5 must be refused, not approximated.
* **RDNA 4 doubles K for the integer forms** (`WMMA_I32_16X16X32_IU4`) and adds
  the whole FP8/BF8 cross-product plus sparse `SWMMAC`.
* **CDNA 5 adds a scale operand class** (`V_WMMA_SCALE*`, `V_WMMA_SCALE16*`,
  `V_WMMA_LD_SCALE`) — block-scaled/microscaled formats are a first-class
  instruction family, not an emulation. It also reaches K=128 and introduces
  `F8F6F4` mixed-precision operands.

### The `V_DOT` family is separate from WMMA, and diverges

Dot-product ops are matrix-adjacent but are *not* WMMA and are easy to conflate.
They differ across all three:

| | RDNA 3.5 | RDNA 4 | CDNA 5 |
|---|---|---|---|
| `V_DOT2_*` (F16/BF16) | ✅ | ✅ | ❌ **absent** |
| `V_DOT4_F32_{FP8,BF8}` | ❌ | ✅ | ❌ **absent** |
| `V_DOT4_I32_IU8`, `V_DOT8_I32_IU4`, `V_DOT8_U32_U4` | ✅ | ✅ | ✅ |

So CDNA 5 keeps only the *integer* dot products and drops the float ones RDNA
has — while RDNA 4 is the only one of the three with an FP8 dot product.

### Encoding and operand constraints (RDNA 3.5, p.77)

WMMA is **VOP3P**-encoded and is explicitly *not* packed math — it is a single
MAD over mixed 16/32-bit inputs, grouped with VOP3P only because it shares the
encoding. The document states **`SRC` must be a VGPR** for WMMA. Check the
equivalent page per architecture before assuming an SGPR or inline constant is
legal as a matrix operand.

### Scheduling hazards: search for `XDL`, not `MFMA`

CDNA 5 calls the matrix pipeline **XDL** (26 occurrences), and the hazard and
stall rules are written in those terms — "XDL WMMA ops (16-bit data and smaller)
are tracked as if...", "Any XDL WMMA/SWMMAC instruction with Matrix...",
"Disable Multicycle XDL Stall". If you are looking for matrix-op dependency or
co-issue constraints, `XDL` is the search term. (`MAI`, AMD's older matrix term,
does **not** appear as a word in any of the three.)

---

## Instruction population, by class

Distinct mnemonics found in each document. Useful for sizing coverage work, and
for spotting that CDNA 5 is the largest ISA of the three on the vector side.

| class | RDNA 3.5 | RDNA 4 | CDNA 5 |
|---|---:|---:|---:|
| `S_*` | 312 | 340 | 346 |
| `V_*` | 566 | 583 | **758** |
| `DS_*` | 139 | 139 | 144 |
| `BUFFER_*` | 87 | 89 | 67 |
| `GLOBAL_*` | 55 | 67 | 89 |
| `IMAGE_*` | 92 | 108 | **0** |

`VOPD` (dual-issue VALU) appears in all three — 27 / 35 / 77 mentions — so
dual-issue is not an RDNA-only concept.

---

## Caveats on these numbers

* Counts are **distinct regex matches over extracted text**, not a parse of the
  opcode tables. They include mnemonics mentioned in prose and in encoding
  tables, so treat them as *upper bounds for the reference chapters* and as
  ratios rather than exact opcode counts. The matrix inventory is the one to
  trust most, because those mnemonics are long and unambiguous.
* **Anchor the mnemonic regex on `V_`.** An unanchored `WMMA_[A-Z0-9_]+` also
  matches line-wrapped fragments (`WMMA_F16_16`, `WMMA_16X16X16_IU4`) and
  inflates the count — it gave 7 and 26 for RDNA 3.5 / RDNA 4 where the anchored
  form gives 6 and 22. The counts above and in `INDEX.md` are the anchored ones.
* Page numbers are **PDF page numbers** (what a viewer jumps to), not the
  printed folio.

---

## Follow-up: teach the existing extractor about CDNA 5

`rdna/tools/build_archive.py` already searches
`~/projects/AMD_GPU_ISA_DOCS` and is format-generic. Adding CDNA 5 is one
`SOURCES` entry:

```python
    "cdna5": {
        "pdf": "amd-instinct-cdna5-instruction-set-architecture.pdf",
        "name": "CDNA5",
        "title": '"CDNA5" Instruction Set Architecture: Reference Guide',
        "kind": "isa",
    },
```

Deliberately **not** done in this PR: the output directory is named
`docs/reference/isa/rdna/`, so CDNA data would land under an RDNA path, and
the regenerated archive is a large binary-ish diff. Both deserve their own
change — the directory should probably become `docs/reference/isa/<arch>/`
first. Until then this assessment is the only CDNA 5 coverage in-tree.

## Reproducing the numbers here

Requires `pypdf`; the docs need no network access.

```bash
python3 -m venv /tmp/pdfenv && /tmp/pdfenv/bin/pip install -q pypdf

# Chapter outline with PDF page numbers
/tmp/pdfenv/bin/python - <<'PY'
from pypdf import PdfReader
import glob, os
for f in sorted(glob.glob("/home/gstoner/projects/AMD_GPU_ISA_DOCS/*.pdf")):
    r = PdfReader(f); print("##", os.path.basename(f))
    def walk(items, d=0):
        for it in items:
            if isinstance(it, list): walk(it, d+1)
            else:
                p = r.get_destination_page_number(it) + 1
                if d == 0: print(f"  {p}\t{it.title}")
    walk(r.outline)
PY

# Full text, then grep for anything
/tmp/pdfenv/bin/python - <<'PY'
from pypdf import PdfReader
import glob, os
for f in sorted(glob.glob("/home/gstoner/projects/AMD_GPU_ISA_DOCS/*.pdf")):
    r = PdfReader(f)
    open(f"/tmp/{os.path.basename(f)}.txt","w").write(
        "\n\f\n".join((p.extract_text() or "") for p in r.pages))
PY

grep -oE "\bV_(S?WMMA)_[A-Z0-9_]+" /tmp/*.txt | sort -u
```
