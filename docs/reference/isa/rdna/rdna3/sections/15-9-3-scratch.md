# 15.9.3. SCRATCH

> RDNA3 ISA — pages 194–194

Opcode # Name                                   Opcode # Name
25         GLOBAL_STORE_B16                     64       GLOBAL_ATOMIC_DEC_U32
26         GLOBAL_STORE_B32                     65       GLOBAL_ATOMIC_SWAP_B64
27         GLOBAL_STORE_B64                     66       GLOBAL_ATOMIC_CMPSWAP_B64
28         GLOBAL_STORE_B96                     67       GLOBAL_ATOMIC_ADD_U64
29         GLOBAL_STORE_B128                    68       GLOBAL_ATOMIC_SUB_U64
30         GLOBAL_LOAD_D16_U8                   69       GLOBAL_ATOMIC_MIN_I64
31         GLOBAL_LOAD_D16_I8                   70       GLOBAL_ATOMIC_MIN_U64
32         GLOBAL_LOAD_D16_B16                  71       GLOBAL_ATOMIC_MAX_I64
33         GLOBAL_LOAD_D16_HI_U8                72       GLOBAL_ATOMIC_MAX_U64
34         GLOBAL_LOAD_D16_HI_I8                73       GLOBAL_ATOMIC_AND_B64
35         GLOBAL_LOAD_D16_HI_B16               74       GLOBAL_ATOMIC_OR_B64
36         GLOBAL_STORE_D16_HI_B8               75       GLOBAL_ATOMIC_XOR_B64
37         GLOBAL_STORE_D16_HI_B16              76       GLOBAL_ATOMIC_INC_U64
40         GLOBAL_LOAD_ADDTID_B32               77       GLOBAL_ATOMIC_DEC_U64
41         GLOBAL_STORE_ADDTID_B32              80       GLOBAL_ATOMIC_CMPSWAP_F32
51         GLOBAL_ATOMIC_SWAP_B32               81       GLOBAL_ATOMIC_MIN_F32
52         GLOBAL_ATOMIC_CMPSWAP_B32            82       GLOBAL_ATOMIC_MAX_F32
53         GLOBAL_ATOMIC_ADD_U32                86       GLOBAL_ATOMIC_ADD_F32
54         GLOBAL_ATOMIC_SUB_U32

15.9.3. SCRATCH
                          Table 110. SCRATCH Opcodes
Opcode # Name                          Opcode # Name
16         SCRATCH_LOAD_U8             27      SCRATCH_STORE_B64
17         SCRATCH_LOAD_I8             28      SCRATCH_STORE_B96
18         SCRATCH_LOAD_U16            29      SCRATCH_STORE_B128
19         SCRATCH_LOAD_I16            30      SCRATCH_LOAD_D16_U8
20         SCRATCH_LOAD_B32            31      SCRATCH_LOAD_D16_I8
21         SCRATCH_LOAD_B64            32      SCRATCH_LOAD_D16_B16
22         SCRATCH_LOAD_B96            33      SCRATCH_LOAD_D16_HI_U8
23         SCRATCH_LOAD_B128           34      SCRATCH_LOAD_D16_HI_I8
24         SCRATCH_STORE_B8            35      SCRATCH_LOAD_D16_HI_B16
25         SCRATCH_STORE_B16           36      SCRATCH_STORE_D16_HI_B8
26         SCRATCH_STORE_B32           37      SCRATCH_STORE_D16_HI_B16
