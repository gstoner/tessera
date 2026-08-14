---
last_updated: 2026-08-13
audit_role: reference
---

# Index — AMD GPU ISA primary-source PDFs

Generated from the PDFs themselves (outline destinations + extracted text),
not transcribed. Page numbers are **PDF page numbers**, which is what a
reader jumps to; they differ from the printed folio.

Assessment and findings: [`PRIMARY_SOURCES.md`](PRIMARY_SOURCES.md).
Regeneration commands are in that file.


---

## RDNA 3.5 — `rdna35_instruction_set_architecture.pdf`

* **653 PDF pages**
* **PDF created 2024-09-11**

### Chapters

| page | chapter |
|---:|---|
| 10 | Preface |
| 12 | Chapter 1. Introduction |
| 18 | Chapter 2. Shader Concepts |
| 22 | Chapter 3. Wave State |
| 44 | Chapter 4. Shader Instruction Set |
| 48 | Chapter 5. Program Flow Control |
| 56 | Chapter 6. Scalar ALU Operations |
| 65 | Chapter 7. Vector ALU Operations |
| 87 | Chapter 8. Scalar Memory Operations |
| 91 | Chapter 9. Vector Memory Buffer Instructions |
| 105 | Chapter 10. Vector Memory Image Instructions |
| 122 | Chapter 11. Global, Scratch and Flat Address Space |
| 129 | Chapter 12. Data Share Operations |
| 143 | Chapter 13. Float Memory Atomics |
| 150 | Chapter 14. Export: Position, Color/MRT |
| 153 | Chapter 15. Microcode Formats |
| 198 | Chapter 16. Instructions |

### Distinct mnemonics by class

| class | distinct |
|---|---:|
| `S_*` | 313 |
| `V_*` | 553 |
| `DS_*` | 139 |
| `BUFFER_*` | 88 |
| `GLOBAL_*` | 55 |
| `FLAT_*` | 55 |
| `SCRATCH_*` | 25 |
| `IMAGE_*` | 92 |
| `EXP*` | 4 |

### Matrix-core instructions (6)

```
WMMA_BF16_16X16X16_BF16
WMMA_F16_16X16X16_F16
WMMA_F32_16X16X16_BF16
WMMA_F32_16X16X16_F16
WMMA_I32_16X16X16_IU4
WMMA_I32_16X16X16_IU8
```

### Dot-product instructions (10)

Not WMMA — separate family, diverges across architectures (see README).

```
DOT2ACC_F32_F16
DOT2_BF16_BF16
DOT2_F16_F16
DOT2_F32_
DOT2_F32_BF16
DOT2_F32_F16
DOT4_I32_IU8
DOT4_U32_U8
DOT8_I32_IU4
DOT8_U32_U4
```

---

## RDNA 4 — `rdna4-instruction-set-architecture.pdf`

* **707 PDF pages**
* **PDF created 2025-04-08**

### Chapters

| page | chapter |
|---:|---|
| 11 | Preface |
| 13 | Chapter 1. Introduction |
| 19 | Chapter 2. Shader Concepts |
| 23 | Chapter 3. Wave State |
| 48 | Chapter 4. Shader Instruction Set |
| 53 | Chapter 5. Program Flow Control |
| 68 | Chapter 6. Scalar ALU Operations |
| 76 | Chapter 7. Vector ALU Operations |
| 107 | Chapter 8. Scalar Memory Operations |
| 112 | Chapter 9. Vector Memory Buffer Instructions |
| 126 | Chapter 10. Vector Memory Image Instructions |
| 145 | Chapter 11. Global, Scratch and Flat Address Space Operations |
| 154 | Chapter 12. Local Data Share Operations |
| 166 | Chapter 13. Float Memory Atomics |
| 169 | Chapter 14. Export: Position, Color/MRT |
| 172 | Chapter 15. Microcode Formats |
| 216 | Chapter 16. Instructions |

### Distinct mnemonics by class

| class | distinct |
|---|---:|
| `S_*` | 339 |
| `V_*` | 584 |
| `DS_*` | 139 |
| `BUFFER_*` | 89 |
| `GLOBAL_*` | 68 |
| `FLAT_*` | 55 |
| `SCRATCH_*` | 31 |
| `IMAGE_*` | 108 |
| `EXP*` | 6 |

### Matrix-core instructions (22)

```
SWMMAC_BF16_16X16X32_BF16
SWMMAC_F16_16X16X32_F16
SWMMAC_F32_16X16X32_BF16
SWMMAC_F32_16X16X32_BF8_BF8
SWMMAC_F32_16X16X32_BF8_FP8
SWMMAC_F32_16X16X32_F16
SWMMAC_F32_16X16X32_FP8_BF8
SWMMAC_F32_16X16X32_FP8_FP8
SWMMAC_I32_16X16X32_IU4
SWMMAC_I32_16X16X32_IU8
SWMMAC_I32_16X16X64_IU4
WMMA_BF16_16X16X16_BF16
WMMA_F16_16X16X16_F16
WMMA_F32_16X16X16_BF16
WMMA_F32_16X16X16_BF8_BF8
WMMA_F32_16X16X16_BF8_FP8
WMMA_F32_16X16X16_F16
WMMA_F32_16X16X16_FP8_BF8
WMMA_F32_16X16X16_FP8_FP8
WMMA_I32_16X16X16_IU4
WMMA_I32_16X16X16_IU8
WMMA_I32_16X16X32_IU4
```

### Dot-product instructions (12)

Not WMMA — separate family, diverges across architectures (see README).

```
DOT2_BF16_BF16
DOT2_F16_F16
DOT2_F32_BF16
DOT2_F32_F16
DOT4_F32_BF8_BF8
DOT4_F32_BF8_FP8
DOT4_F32_FP8_BF8
DOT4_F32_FP8_FP8
DOT4_I32_IU8
DOT4_U32_U8
DOT8_I32_IU4
DOT8_U32_U4
```

---

## CDNA 5 — `amd-instinct-cdna5-instruction-set-architecture.pdf`

* **832 PDF pages**
* **PDF created 2026-07-29**

### Chapters

| page | chapter |
|---:|---|
| 10 | Preface |
| 12 | Chapter 1. Introduction |
| 18 | Chapter 2. Shader Concepts |
| 21 | Chapter 3. Wave State |
| 41 | Chapter 4. Shader Instruction Set |
| 49 | Chapter 5. Program Flow Control |
| 68 | Chapter 6. Scalar ALU Operations |
| 76 | Chapter 7. Vector ALU Operations |
| 117 | Chapter 8. Scalar Memory Operations |
| 123 | Chapter 9. Vector Memory Buffer Instructions |
| 134 | Chapter 10. Global, Scratch and Flat Address Space Operations |
| 157 | Chapter 11. Local Data Share Operations |
| 165 | Chapter 12. Float Memory Atomics |
| 168 | Chapter 13. Error Correction Codes (ECC) |
| 170 | Chapter 14. Microcode Formats |
| 217 | Chapter 15. Instructions |

### Distinct mnemonics by class

| class | distinct |
|---|---:|
| `S_*` | 344 |
| `V_*` | 758 |
| `DS_*` | 144 |
| `BUFFER_*` | 67 |
| `GLOBAL_*` | 90 |
| `FLAT_*` | 69 |
| `SCRATCH_*` | 31 |
| `EXP*` | 1 |

### Matrix-core instructions (50)

```
SWMMAC_
SWMMAC_BF16F32_16X16X64_BF16
SWMMAC_BF16_16X16X64_BF16
SWMMAC_F16_16X16X128_BF8_BF8
SWMMAC_F16_16X16X128_BF8_FP8
SWMMAC_F16_16X16X128_FP8_BF8
SWMMAC_F16_16X16X128_FP8_FP8
SWMMAC_F16_16X16X64_F16
SWMMAC_F32_16X16X128_BF8_BF8
SWMMAC_F32_16X16X128_BF8_FP8
SWMMAC_F32_16X16X128_FP8_BF8
SWMMAC_F32_16X16X128_FP8_FP8
SWMMAC_F32_16X16X64_BF16
SWMMAC_F32_16X16X64_F16
SWMMAC_I32_16X16X128_IU8
WMMA_BF16F32_16X16X32_BF16
WMMA_BF16_16X16X32_BF16
WMMA_F16_16X16X128_BF8_BF8
WMMA_F16_16X16X128_BF8_FP8
WMMA_F16_16X16X128_FP8_BF8
WMMA_F16_16X16X128_FP8_FP8
WMMA_F16_16X16X32_F16
WMMA_F16_16X16X64_BF8_BF8
WMMA_F16_16X16X64_BF8_FP8
WMMA_F16_16X16X64_FP8_BF8
WMMA_F16_16X16X64_FP8_FP8
WMMA_F32_16X16X128_BF8_BF8
WMMA_F32_16X16X128_BF8_FP8
WMMA_F32_16X16X128_F8F6F4
WMMA_F32_16X16X128_FP8_BF8
WMMA_F32_16X16X128_FP8_FP8
WMMA_F32_16X16X32_BF16
WMMA_F32_16X16X32_F16
WMMA_F32_16X16X4_F32
WMMA_F32_16X16X64_BF8_BF8
WMMA_F32_16X16X64_BF8_FP8
WMMA_F32_16X16X64_FP8_BF8
WMMA_F32_16X16X64_FP8_FP8
WMMA_F32_32X16X128_F4
WMMA_I32_16X16X64_IU8
WMMA_LD_SCALE
WMMA_SCALE
WMMA_SCALE16_F32_16
WMMA_SCALE16_F32_16X16X128_F8F6F4
WMMA_SCALE16_F32_32
WMMA_SCALE16_F32_32X16X128_F4
WMMA_SCALE_F32_16X16X128_F
WMMA_SCALE_F32_16X16X128_F8F6F4
WMMA_SCALE_F32_32X16X128_F
WMMA_SCALE_F32_32X16X128_F4
```

### Dot-product instructions (4)

Not WMMA — separate family, diverges across architectures (see README).

```
DOT4_I32_IU8
DOT4_U32_U8
DOT8_I32_IU4
DOT8_U32_U4
```
