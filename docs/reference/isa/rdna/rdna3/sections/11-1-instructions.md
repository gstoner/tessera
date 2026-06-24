# 11.1. Instructions

> RDNA3 ISA — pages 122–122

      Flat                              GLOBAL                             Scratch
      FLAT_ATOMIC_ADD_U32               GLOBAL_ATOMIC_ADD_U32              none
      FLAT_ATOMIC_ADD_F32               GLOBAL_ATOMIC_ADD_F32              none
      FLAT_ATOMIC_SUB_U32               GLOBAL_ATOMIC_SUB_U32              none
      FLAT_ATOMIC_MIN_I32               GLOBAL_ATOMIC_MIN_I32              none
      FLAT_ATOMIC_MIN_U32               GLOBAL_ATOMIC_MIN_U32              none
      FLAT_ATOMIC_MAX_I32               GLOBAL_ATOMIC_MAX_I32              none
      FLAT_ATOMIC_MAX_U32               GLOBAL_ATOMIC_MAX_U32              none
      FLAT_ATOMIC_AND_B32               GLOBAL_ATOMIC_AND_B32              none
      FLAT_ATOMIC_OR_B32                GLOBAL_ATOMIC_OR_B32               none
      FLAT_ATOMIC_XOR_B32               GLOBAL_ATOMIC_XOR_B32              none
      FLAT_ATOMIC_INC_U32               GLOBAL_ATOMIC_INC_U32              none
      FLAT_ATOMIC_DEC_U32               GLOBAL_ATOMIC_DEC_U32              none
      FLAT_ATOMIC_CMPSWAP_F32           GLOBAL_ATOMIC_CMPSWAP_F32          none
      FLAT_ATOMIC_MIN_F32               GLOBAL_ATOMIC_MIN_F32              none
      FLAT_ATOMIC_MAX_F32               GLOBAL_ATOMIC_MAX_F32              none
      FLAT_ATOMIC_SWAP_B64              GLOBAL_ATOMIC_SWAP_B64             none
      FLAT_ATOMIC_CMPSWAP_B64           GLOBAL_ATOMIC_CMPSWAP_B64          none
      FLAT_ATOMIC_ADD_U64               GLOBAL_ATOMIC_ADD_U64              none
      FLAT_ATOMIC_SUB_U64               GLOBAL_ATOMIC_SUB_U64              none
      FLAT_ATOMIC_MIN_I64               GLOBAL_ATOMIC_MIN_I64              none
      FLAT_ATOMIC_MIN_U64               GLOBAL_ATOMIC_MIN_U64              none
      FLAT_ATOMIC_MAX_I64               GLOBAL_ATOMIC_MAX_I64              none
      FLAT_ATOMIC_MAX_U64               GLOBAL_ATOMIC_MAX_U64              none
      FLAT_ATOMIC_AND_B64               GLOBAL_ATOMIC_AND_B64              none
      FLAT_ATOMIC_OR_B64                GLOBAL_ATOMIC_OR_B64               none
      FLAT_ATOMIC_XOR_B64               GLOBAL_ATOMIC_XOR_B64              none
      FLAT_ATOMIC_INC_U64               GLOBAL_ATOMIC_INC_U64              none
      FLAT_ATOMIC_DEC_U64               GLOBAL_ATOMIC_DEC_U64              none
      none                              GLOBAL_ATOMIC_CSUB_U32             none
                                        (GLC must be set to 1)

11.1. Instructions

11.1.1. FLAT
The Flat instruction set is nearly identical to the BUFFER instruction set, minus the FORMAT loads & stores.

Flat instructions do not use a resource constant (V#) or sampler (S#), but they do use a specific SGPR-pair
(FLAT_SCRATCH) to hold scratch-space information in case any threads' address resolves to scratch space. See
"Scratch" section below.

Since Flat instruction are executed as both an LDS and a Global instruction, Flat instructions increment both
VMcnt (or VScnt) and LGKMcnt and are not considered done until both have been decremented. There is no
way a priori to determine whether a Flat instruction uses only LDS or Global memory space.

When the address from a Flat instruction falls into scratch (private) space, a different addressing mechanism is
