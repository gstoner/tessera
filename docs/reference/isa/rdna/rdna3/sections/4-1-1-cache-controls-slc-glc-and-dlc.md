# 4.1.1. Cache Controls: SLC, GLC and DLC

> RDNA3 ISA — pages 44–46

                                 Code        Meaning
Vector    Scalar    Scalar       0-105       SGPR 0 .. 105      SGPRs. One DWORD each.
Source    Source (8 Dest (7      106         VCC_LO             VCC[31:0]
(when 9   bits)     bits)        107         VCC_HI             VCC[63:32]
bits)
                                 108-123     ttmp0 .. ttmp15    Trap handler temporary SGPRs (privileged)
                                 124         NULL               Reads return zero, writes are ignored. When used
                                                                as a destination, nullifies the instruction.
                                 125         M0                 Temporary register, use for a variety of functions
                                 126         EXEC_LO            EXEC[31:0]
                                 127         EXEC_HI            EXEC[63:32]
                     Integer   128           0                  Inline constant zero
                     Inline    129-192       int 1 .. 64        Integer inline constants
                     Constants 193-208       int -1 .. -16
                                 209-232     Reserved           Reserved
                                 233         DPP8               8-lane DPP (only valid as SRC0)
                                 234         DPP8FI             8-lane DPP with Fetch-Invalid (only valid as SRC0)
                                 235         SHARED_BASE        Memory Aperture Definition
                                 236         SHARED_LIMIT
                                 237         PRIVATE_BASE
                                 238         PRIVATE_LIMIT
                                 239         Reserved           Reserved
                     Float     240           0.5                Inline floating point constants. Can be used in 16,
                     Inline    241           -0.5               32 and 64 bit floating point math. They may be
                     Constants 242           1.0                used with non-float instructions but the value
                                                                remains a float.
                                 243         -1.0
                                 244         2.0                1/(2*PI) is 0.15915494. The hex values are:
                                 245         -2.0               half: 0x3118
                                 246         4.0                single: 0x3e22f983
                                 247         -4.0               double: 0x3fc45f306dc9c882
                                 248         1.0 / (2 * PI)
                                 249         Reserved           Reserved
                                 250         DPP16              data parallel primitive
                                 251         Reserved           Reserved
                                 252         Reserved           Reserved
                                 253         SCC                { 31’b0, SCC }
                                 254         Reserved           Reserved
                                 255         Literal constant   32 bit constant from instruction stream
          Vector Src/Dst         256 - 511   VGPR 0 .. 255      Vector GPRs. One DWORD each.
          (8 bits)

4.1.1. Cache Controls: SLC, GLC and DLC
Scalar and vector memory instructions contain bits that control cache behavior. The SLC, GLC and DLC
instruction bits influence cache behavior for loads, stores, and atomics.

  GLC      controls the graphics first-level cache

  SLC        controls the graphics L2 cache

  DLC        controls the Memory-Attached Last-Level cache (MALL) if it is present (ignored otherwise)

Typically loads use GLC=0 (except for load-acquire). GLC=1 forces a miss in the first level cache and reads data
rom the L2 cache. If there was a line in the GPU L0 that matched, it is invalidated; L2 is reread.

Shader LOAD ops (load, sample, gather, etc…)

SRD                    ISA                        Resulting Policy in Cache                 SCOPE           Non-Temporal Hint
llc_        DLC SLC          GLC       MALL GL2              GL1            Tex(L0)                  MALL   GL2     GL1     Tex(L0)
noalloc                                (NOA)
0 or 1      0      0         0         0         LRU         HIT_LRU        HIT_LRU         CU       no     no      no      no
0 or 1      0      0         1         0         LRU         MISS_EVICT     MISS_EVICT      DEVICE   no     no      _NA_    _NA_
0 or 1      0      1         0         0         STREAM      HIT_EVICT      HIT_LRU         CU       no     yes     yes     no
0 or 1      0      1         1         0         STREAM      MISS_EVICT     MISS_EVICT      DEVICE   no     yes     _NA_    _NA_
0 or 1      1      0         0         1         LRU         HIT_LRU        HIT_LRU         CU       yes    no      no      no
0 or 1      1      0         1         1         LRU         MISS_EVICT     MISS_EVICT      DEVICE   yes    no      _NA_    _NA_
0 or 1      1      1         0         1         STREAM      HIT_EVICT      HIT_LRU         CU       yes    yes     yes     no
0 or 1      1      1         1         1         STREAM      MISS_EVICT     MISS_EVICT      DEVICE   yes    yes     _NA_    _NA_
2 or 3      0      0         0         1         LRU         HIT_LRU        HIT_LRU         CU       no     no      no      no
2 or 3      0      0         1         1         LRU         MISS_EVICT     MISS_EVICT      DEVICE   no     no      _NA_    _NA_
2 or 3      0      1         0         1         STREAM      HIT_EVICT      HIT_LRU         CU       no     yes     yes     no
2 or 3      0      1         1         1         STREAM      MISS_EVICT     MISS_EVICT      DEVICE   no     yes     _NA_    _NA_
2 or 3      1      0         0         1         LRU         HIT_LRU        HIT_LRU         CU       yes    no      no      no
2 or 3      1      0         1         1         LRU         MISS_EVICT     MISS_EVICT      DEVICE   yes    no      _NA_    _NA_
2 or 3      1      1         0         1         STREAM      HIT_EVICT      HIT_LRU         CU       yes    yes     yes     no
2 or 3      1      1         1         1         STREAM      MISS_EVICT     MISS_EVICT      DEVICE   yes    yes     _NA_    _NA_

  • For S_BUFFER_LOAD instructions, LLC_NOALLOC comes from V#.LLC_noalloc.
    For S_LOAD, LLC_NOALLOC is zero.
  • SMEM operations have SLC set to zero.

Shader STORE / ATOMIC ops (all are device scope)

SRD                    ISA                 Policy in Cache                Non-Temporal Hint
llc_         DLC        SLC        MALL         GL2             MALL                  GL2
noalloc                            (NOA)
0 or 2       0          0          0            LRU             no                    no
0 or 2       0          1          0            STREAM          no                    yes
0 or 2       1          0          1            LRU             yes                   no
0 or 2       1          1          1            STREAM          yes                   yes
1 or 3       0          0          1            LRU             no                    no
1 or 3       0          1          1            STREAM          no                    yes
1 or 3       1          0          1            LRU             no                    no
1 or 3       1          1          1            STREAM          no                    yes

    "Temporal Hint" = expect data to have temporal reuse.
    "SRD" = Shader Resource Descriptor

  • ISA.GLC ⇒ this is a scope bit for load operations (including sample, gather, etc…)
      ◦ 0 : CU (work-group) scope
         ◦ 1 : DEVICE scope

      ◦ All stores/atomic ops are device scope (GLC has non-perf related functionality)
  • ISA.SLC ⇒ Temporal Hint for graphic client caches
      ◦ 0 : Regular
      ◦ 1 : Stream (non-temporal)
  • ISA.DLC ⇒ Temporal Hint for Infinity Cache
      ◦ 0 : Regular
      ◦ 1 : Non-temporal

GLC is used by atomics to indicate:
  • 0: return nothing
  • 1: return pre-operation value from memory to VGPR
