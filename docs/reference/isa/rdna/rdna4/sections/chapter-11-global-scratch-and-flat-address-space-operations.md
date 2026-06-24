# Chapter 11. Global, Scratch and Flat Address Space Operations

> RDNA4 ISA — pages 145–147

Chapter 11. Global, Scratch and Flat Address
Space Operations
Flat, Global and Scratch are a collection of VMEM instructions that allow per-thread access to global memory,
shared memory and private memory. Unlike buffer and image instructions, these do not use a resource
constant.

Flat is the most generic of the 3 types where per-thread the address may map to global, private or shared
memory. Memory is addressed as a single flat address space, where certain memory address apertures map
these regions. The determination of the memory space to which an address maps is controlled by a set of
"memory aperture" base and size registers. Flat load/store/atomic instructions are effectively a simultaneous
issue of an LDS and GLOBAL instruction at the same time with the same address. The address per-thread is
read from the ADDR VGPR and then tested to see in which address space the data exists.

Flat Address Space ("flat") instructions allow load/store/atomic access to a generic memory address pointer that
can resolve to any of the following physical memories:
  • Global memory
  • Scratch ("private")
  • LDS ("shared")
  • Invalid
  • But not to: GPRs or LDS-parameters

GLOBAL is used when all of the address fall into global memory, not LDS or Scratch. This should be used when
possible (instead of "Flat") as Global does not tie up LDS resources. SCRATCH is similar, but is used to access
scratch (private) memory space.

Scratch (thread-private memory) is an allocation of memory private to a wave or workgroup in global memory.
It can be accessed either via SCRATCH memory instructions, or using FLAT instructions where the address
falls into an area of memory defined by the aperture registers. When an address falls in scratch space,
additional address computation is automatically performed by the hardware. For waves that are allocated
scratch memory space, the 64-bit SCRATCH_BASE register is initialized with the a pointer to that wave’s private
scratch memory. Waves that have no scratch memory have SCRATCH_BASE initialized to zero.
SCRATCH_BASE is a 64-bit byte address that is implicitly used by Flat and Scratch memory instructions, and
can be manually read via S_GETREG_B32.

The instruction specifies which VGPR supplies the address (per work-item), and that address for each work-
item may be in any one of those address spaces.

Instruction Fields

Field            Size Description
OP               8      Opcode: see next table
VADDR            8      VGPR that holds address or offset. For 64-bit addresses, ADDR has the LSB’s and ADDR+1 has the
                        MSBs. For offset a single VGPR has a 32 bit unsigned offset.
                        For FLAT_*: specifies an address.
                        For GLOBAL_* when SADDR is NULL: specifies an address.
                        For GLOBAL_* when SADDR is not NULL: specifies an unsigned byte offset.
                        For SCRATCH, specifies a signed byte offset if SVE=1
VSRC             8      VGPR that holds the first DWORD of store-data. Instructions can use 0-4 DWORDs.
VDST             8      VGPR destination for data returned to the shader, either from LOADs or Atomics that return the
                        pre-op value.
SCOPE            2      Memory Scope and Memory Temporal Hint. For atomics, TH indicates whether or not to return the
                        pre-op value.
TH               3
                        See Cache Controls: SCOPE and Temporal-Hint for more information about SCOPE and TH bits.
IOFFSET          24     Address immediate offset: 24-bit signed byte offset
SADDR            7      SGPR that provides an address or offset. To disable use, set this field to NULL. The meaning of this
                        field is different for Scratch and Global.
                        Flat: Unused
                        Scratch: use an SGPR for the signed 32-bit.
                        Global: use the SGPR to provide a 64-bit unsigned base address and the VGPR provides a 32-bit byte
                        offset.
SVE              1      Scratch VGPR Enable
                        When set to 1, scratch instructions include a 32-bit offset from a VGPR;
                        when set to 0, scratch instructions do not use a VGPR for addressing.

                                                  Table 67. Instructions
        Flat                                 GLOBAL                                 Scratch
        FLAT_LOAD_U8                         GLOBAL_LOAD_U8                         SCRATCH_LOAD_U8
        FLAT_LOAD_D16_U8                     GLOBAL_LOAD_D16_U8                     SCRATCH_LOAD_D16_U8
        FLAT_LOAD_D16_HI_U8                  GLOBAL_LOAD_D16_HI_U8                  SCRATCH_LOAD_D16_HI_U8
        FLAT_LOAD_I8                         GLOBAL_LOAD_I8                         SCRATCH_LOAD_I8
        FLAT_LOAD_D16_I8                     GLOBAL_LOAD_D16_I8                     SCRATCH_LOAD_D16_I8
        FLAT_LOAD_D16_HI_I8                  GLOBAL_LOAD_D16_HI_I8                  SCRATCH_LOAD_D16_HI_I8
        FLAT_LOAD_U16                        GLOBAL_LOAD_U16                        SCRATCH_LOAD_U16
        FLAT_LOAD_I16                        GLOBAL_LOAD_I16                        SCRATCH_LOAD_I16
        FLAT_LOAD_D16_B16                    GLOBAL_LOAD_D16_B16                    SCRATCH_LOAD_D16_B16
        FLAT_LOAD_D16_HI_B16                 GLOBAL_LOAD_D16_HI_B16                 SCRATCH_LOAD_D16_HI_B16
        FLAT_LOAD_B32                        GLOBAL_LOAD_B32                        SCRATCH_LOAD_B32
        FLAT_LOAD_B64                        GLOBAL_LOAD_B64                        SCRATCH_LOAD_B64
        FLAT_LOAD_B96                        GLOBAL_LOAD_B96                        SCRATCH_LOAD_B96
        FLAT_LOAD_B128                       GLOBAL_LOAD_B128                       SCRATCH_LOAD_B128
        FLAT_STORE_B8                        GLOBAL_STORE_B8                        SCRATCH_STORE_B8
        FLAT_STORE_D16_HI_B8                 GLOBAL_STORE_D16_HI_B8                 SCRATCH_STORE_D16_HI_B8
        FLAT_STORE_B16                       GLOBAL_STORE_B16                       SCRATCH_STORE_B16
        FLAT_STORE_D16_HI_B16                GLOBAL_STORE_D16_HI_B16                SCRATCH_STORE_D16_HI_B16
        FLAT_STORE_B32                       GLOBAL_STORE_B32                       SCRATCH_STORE_B32

      Flat                             GLOBAL                           Scratch
      FLAT_STORE_B64                   GLOBAL_STORE_B64                 SCRATCH_STORE_B64
      FLAT_STORE_B96                   GLOBAL_STORE_B96                 SCRATCH_STORE_B96
      FLAT_STORE_B128                  GLOBAL_STORE_B128                SCRATCH_STORE_B128
      none                             GLOBAL_LOAD_ADDTID_B32           none
      none                             GLOBAL_STORE_ADDTID_B32          none
      FLAT_ATOMIC_SWAP_B32             GLOBAL_ATOMIC_SWAP_B32           none
      FLAT_ATOMIC_CMPSWAP_B32          GLOBAL_ATOMIC_CMPSWAP_B32        none
      FLAT_ATOMIC_ADD_U32              GLOBAL_ATOMIC_ADD_U32            none
      FLAT_ATOMIC_ADD_F32              GLOBAL_ATOMIC_ADD_F32            none
      FLAT_ATOMIC_PK_ADD_F16           GLOBAL_ATOMIC_PK_ADD_F16         none
      FLAT_ATOMIC_PK_ADD_BF16          GLOBAL_ATOMIC_PK_ADD_BF16        none
      FLAT_ATOMIC_SUB_U32              GLOBAL_ATOMIC_SUB_U32            none
      FLAT_ATOMIC_MIN_I32              GLOBAL_ATOMIC_MIN_I32            none
      FLAT_ATOMIC_MIN_U32              GLOBAL_ATOMIC_MIN_U32            none
      FLAT_ATOMIC_MAX_I32              GLOBAL_ATOMIC_MAX_I32            none
      FLAT_ATOMIC_MAX_U32              GLOBAL_ATOMIC_MAX_U32            none
      FLAT_ATOMIC_AND_B32              GLOBAL_ATOMIC_AND_B32            none
      FLAT_ATOMIC_OR_B32               GLOBAL_ATOMIC_OR_B32             none
      FLAT_ATOMIC_XOR_B32              GLOBAL_ATOMIC_XOR_B32            none
      FLAT_ATOMIC_INC_U32              GLOBAL_ATOMIC_INC_U32            none
      FLAT_ATOMIC_DEC_U32              GLOBAL_ATOMIC_DEC_U32            none
      FLAT_ATOMIC_MIN_NUM_F32          GLOBAL_ATOMIC_MIN_NUM_F32        none
      FLAT_ATOMIC_MAX_NUM_F32          GLOBAL_ATOMIC_MAX_NUM_F32        none
      FLAT_ATOMIC_SWAP_B64             GLOBAL_ATOMIC_SWAP_B64           none
      FLAT_ATOMIC_ADD_U64              GLOBAL_ATOMIC_ADD_U64            none
      FLAT_ATOMIC_SUB_U64              GLOBAL_ATOMIC_SUB_U64            none
      FLAT_ATOMIC_MIN_I64              GLOBAL_ATOMIC_MIN_I64            none
      FLAT_ATOMIC_MIN_U64              GLOBAL_ATOMIC_MIN_U64            none
      FLAT_ATOMIC_MAX_I64              GLOBAL_ATOMIC_MAX_I64            none
      FLAT_ATOMIC_MAX_U64              GLOBAL_ATOMIC_MAX_U64            none
      FLAT_ATOMIC_AND_B64              GLOBAL_ATOMIC_AND_B64            none
      FLAT_ATOMIC_OR_B64               GLOBAL_ATOMIC_OR_B64             none
      FLAT_ATOMIC_XOR_B64              GLOBAL_ATOMIC_XOR_B64            none
      FLAT_ATOMIC_INC_U64              GLOBAL_ATOMIC_INC_U64            none
      FLAT_ATOMIC_DEC_U64              GLOBAL_ATOMIC_DEC_U64            none
      FLAT_ATOMIC_COND_SUB_U32         GLOBAL_ATOMIC_COND_SUB_U32       none
      (Supports only "return preOp")   (Supports only "return preOp")
      none                             GLOBAL_ATOMIC_SUB_CLAMP_U32 none
                                       (Supports only "return preOp")
      none                             GLOBAL_ATOMIC_ORDERED_ADD_ none
                                       B64
                                       (Supports only "return preOp")
      none                             GLOBAL_INV                       none
      none                             GLOBAL_WB                        none
      none                             GLOBAL_WBINV                     none
