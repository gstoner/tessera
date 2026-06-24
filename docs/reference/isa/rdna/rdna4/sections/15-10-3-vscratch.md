# 15.10.3. VSCRATCH

> RDNA4 ISA — pages 214–214

Opcode # Name                                        Opcode # Name
22         GLOBAL_LOAD_B96                           64       GLOBAL_ATOMIC_DEC_U32
23         GLOBAL_LOAD_B128                          65       GLOBAL_ATOMIC_SWAP_B64
24         GLOBAL_STORE_B8                           66       GLOBAL_ATOMIC_CMPSWAP_B64
25         GLOBAL_STORE_B16                          67       GLOBAL_ATOMIC_ADD_U64
26         GLOBAL_STORE_B32                          68       GLOBAL_ATOMIC_SUB_U64
27         GLOBAL_STORE_B64                          69       GLOBAL_ATOMIC_MIN_I64
28         GLOBAL_STORE_B96                          70       GLOBAL_ATOMIC_MIN_U64
29         GLOBAL_STORE_B128                         71       GLOBAL_ATOMIC_MAX_I64
30         GLOBAL_LOAD_D16_U8                        72       GLOBAL_ATOMIC_MAX_U64
31         GLOBAL_LOAD_D16_I8                        73       GLOBAL_ATOMIC_AND_B64
32         GLOBAL_LOAD_D16_B16                       74       GLOBAL_ATOMIC_OR_B64
33         GLOBAL_LOAD_D16_HI_U8                     75       GLOBAL_ATOMIC_XOR_B64
34         GLOBAL_LOAD_D16_HI_I8                     76       GLOBAL_ATOMIC_INC_U64
35         GLOBAL_LOAD_D16_HI_B16                    77       GLOBAL_ATOMIC_DEC_U64
36         GLOBAL_STORE_D16_HI_B8                    79       GLOBAL_WBINV
37         GLOBAL_STORE_D16_HI_B16                   80       GLOBAL_ATOMIC_COND_SUB_U32
40         GLOBAL_LOAD_ADDTID_B32                    81       GLOBAL_ATOMIC_MIN_NUM_F32
41         GLOBAL_STORE_ADDTID_B32                   82       GLOBAL_ATOMIC_MAX_NUM_F32
43         GLOBAL_INV                                83       GLOBAL_LOAD_BLOCK
44         GLOBAL_WB                                 84       GLOBAL_STORE_BLOCK
51         GLOBAL_ATOMIC_SWAP_B32                    86       GLOBAL_ATOMIC_ADD_F32
52         GLOBAL_ATOMIC_CMPSWAP_B32                 87       GLOBAL_LOAD_TR_B128
53         GLOBAL_ATOMIC_ADD_U32                     88       GLOBAL_LOAD_TR_B64
54         GLOBAL_ATOMIC_SUB_U32                     89       GLOBAL_ATOMIC_PK_ADD_F16
55         GLOBAL_ATOMIC_SUB_CLAMP_U32               90       GLOBAL_ATOMIC_PK_ADD_BF16
56         GLOBAL_ATOMIC_MIN_I32                     115      GLOBAL_ATOMIC_ORDERED_ADD_B64
57         GLOBAL_ATOMIC_MIN_U32

15.10.3. VSCRATCH
                          Table 120. VSCRATCH Opcodes
Opcode # Name                             Opcode # Name
16         SCRATCH_LOAD_U8                28      SCRATCH_STORE_B96
17         SCRATCH_LOAD_I8                29      SCRATCH_STORE_B128
18         SCRATCH_LOAD_U16               30      SCRATCH_LOAD_D16_U8
19         SCRATCH_LOAD_I16               31      SCRATCH_LOAD_D16_I8
20         SCRATCH_LOAD_B32               32      SCRATCH_LOAD_D16_B16
21         SCRATCH_LOAD_B64               33      SCRATCH_LOAD_D16_HI_U8
22         SCRATCH_LOAD_B96               34      SCRATCH_LOAD_D16_HI_I8
23         SCRATCH_LOAD_B128              35      SCRATCH_LOAD_D16_HI_B16
24         SCRATCH_STORE_B8               36      SCRATCH_STORE_D16_HI_B8
25         SCRATCH_STORE_B16              37      SCRATCH_STORE_D16_HI_B16
26         SCRATCH_STORE_B32              83      SCRATCH_LOAD_BLOCK
27         SCRATCH_STORE_B64              84      SCRATCH_STORE_BLOCK
